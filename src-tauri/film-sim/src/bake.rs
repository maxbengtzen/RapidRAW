//! Bake a film look into the small, GPU-friendly tables RapidRAW's shader
//! consumes. (film-sim addition, not part of spektrafilm-rs.)
//!
//! The spectral chain is split at the film density:
//!
//! ```text
//!  scene linear RGB ──(A) tc LUT + 3×3──► film raw ──log10──► log raw
//!     ──(curves) 1-D LUT [+ DIR couplers]──► film density CMY
//!     ──(B) 3-D LUT: enlarger → paper → scanner → display──► output
//! ```
//!
//! (A) and the density curves are exact per-pixel maths. (B) is smooth in
//! density, so it is sampled on a 3-D grid through the verified CPU print +
//! scan stages (`Pipeline::render_from_film_density`). Grain and halation
//! are captured as parameters (see `scatter` for the light scattering).
//!
//! The parameters for (A), the curves and the spatial passes are captured
//! from `Pipeline::try_gpu_resident` through [`CaptureBackend`], i.e. they
//! are exactly what spektrafilm-rs hands its own GPU backend.

use std::path::Path;
use std::sync::Mutex;

use crate::backend::cpu_backend::CpuBackend;
use crate::backend::{
    ComputeBackend, FilmChainParams, FrontPass, GrainGpuParams, HalationGpuParams, Lut3D,
};
use crate::engine::params::{HalationParams, RuntimeParams};
use crate::engine::pipeline::Pipeline;
use crate::engine::profile;
use crate::math::image::ImageBuf;
use crate::math::precision::{Scalar, from_f32, to_f32};
use crate::math::spectral::TcLut;

/// Long edge (px) of the virtual image the pixel-space parameters are baked
/// for. Scale `*_px` values by `actual_long_edge / BAKE_REFERENCE_LONG_EDGE`.
pub const BAKE_REFERENCE_LONG_EDGE: u32 = 1000;

/// Number of samples in the uniform 1-D density-curve tables.
pub const CURVE_SAMPLES: usize = 1024;

/// A per-channel curve sampled on a uniform grid over `[x_min, x_max]`.
#[derive(Debug, Clone)]
pub struct Curve1D {
    pub x_min: f32,
    pub x_max: f32,
    /// `CURVE_SAMPLES` RGB triples.
    pub values: Vec<[f32; 3]>,
}

impl Curve1D {
    /// Linear interpolation with edge clamping (what the shader does).
    pub fn eval(&self, x: f32, c: usize) -> f32 {
        let n = self.values.len();
        let t = ((x - self.x_min) / (self.x_max - self.x_min)).clamp(0.0, 1.0) * (n - 1) as f32;
        let i0 = (t.floor() as usize).min(n - 2);
        let f = t - i0 as f32;
        self.values[i0][c] * (1.0 - f) + self.values[i0 + 1][c] * f
    }
}

#[derive(Debug, Clone)]
pub struct BakedDirCouplers {
    /// `couplers_matrix * amount`, row-major; correction[m] = Σ_k silver[k]·M[k][m].
    pub matrix: [[f32; 3]; 3],
    pub density_max: [f32; 3],
    pub is_positive: bool,
    /// "Density curves before DIR", re-evaluated on the corrected log raw.
    pub curves_0: Curve1D,
    /// Spatial diffusion of the inhibitor (0 = per-pixel only).
    pub diffusion_size_px: f32,
    pub diffusion_tail_px: f32,
    pub diffusion_tail_weight: f32,
}

/// Density → output RGB lattice. `data` is `size³` RGB triples, red index
/// fastest (`((b * size) + g) * size + r`), output already display-encoded.
#[derive(Debug, Clone)]
pub struct DensityLut {
    pub size: u32,
    pub density_min: [f32; 3],
    pub density_max: [f32; 3],
    pub data: Vec<[f32; 3]>,
}

/// Everything RapidRAW needs to render one film look.
#[derive(Debug, Clone)]
pub struct BakedFilm {
    pub tc_lut: TcLutF32,
    /// Linear input RGB → CAT-adapted XYZ, exposure folded in (row-major).
    pub rgb_to_adapted_xyz: [[f32; 3]; 3],
    /// Normalised film density curves over log raw (gamma applied).
    pub film_curves: Curve1D,
    pub dir_couplers: Option<BakedDirCouplers>,
    pub density_lut: DensityLut,
    pub halation: Option<HalationGpuParams>,
    pub grain: Option<GrainGpuParams>,
    /// Halation parameters in µm (stock-specific), for `scatter::scatter_field`.
    pub halation_params: HalationParams,
    /// Film format (long edge, mm): sets the physical pixel size.
    pub film_format_mm: f32,
}

/// f32 copy of the 2-D tc LUT (`size × size` RGB, row-major `[y][x]`).
#[derive(Debug, Clone)]
pub struct TcLutF32 {
    pub size: u32,
    pub data: Vec<[f32; 3]>,
}

#[derive(Debug, thiserror::Error)]
pub enum BakeError {
    #[error("profile: {0}")]
    Profile(String),
    #[error("pipeline: {0}")]
    Pipeline(String),
    #[error("film chain capture failed")]
    Capture,
}

/// Input colour space RapidRAW feeds the film chain: linear, sRGB primaries,
/// D65 (its processing "pipe" space).
pub const RAPIDRAW_INPUT_COLOR_SPACE: &str = "sRGB";

/// Build the pipeline for a film/paper pair with RapidRAW's I/O conventions.
pub fn build_pipeline(
    data_dir: &Path,
    film_stock: &str,
    paper_stock: Option<&str>,
    mut params: RuntimeParams,
) -> Result<Pipeline, BakeError> {
    let film = profile::load_profile_by_name(data_dir, film_stock)
        .map_err(|e| BakeError::Profile(format!("{film_stock}: {e}")))?;
    // Stocks without a target print (slides, some B&W) are scanned directly.
    if paper_stock.is_none() && film.info.target_print.is_none() {
        params.io.scan_film = true;
    }
    let paper_name = if params.io.scan_film {
        film_stock.to_string()
    } else if let Some(p) = paper_stock {
        p.to_string()
    } else {
        film.info.target_print.clone().unwrap_or_default()
    };
    let print = profile::load_profile_by_name(data_dir, &paper_name)
        .map_err(|e| BakeError::Profile(format!("{paper_name}: {e}")))?;
    params.io.input_color_space = RAPIDRAW_INPUT_COLOR_SPACE.into();
    params.io.input_cctf_decoding = false;
    params.io.output_color_space = "sRGB".into();
    params.io.output_cctf_encoding = true;
    // RapidRAW owns exposure; the film sees the image as edited.
    params.camera.auto_exposure = false;
    Pipeline::new_with_spectral(film, print, params, data_dir).map_err(BakeError::Pipeline)
}

/// Bake all tables for `pipeline`. `lut_size` is the density-LUT edge (33–65).
pub fn bake(pipeline: &Pipeline, lut_size: u32) -> Result<BakedFilm, BakeError> {
    let captured = capture_chain(pipeline)?;
    let density_lut = bake_density_lut(pipeline, &captured, lut_size);
    Ok(BakedFilm {
        tc_lut: captured.tc_lut,
        rgb_to_adapted_xyz: captured.rgb_to_adapted_xyz,
        film_curves: captured.film_curves,
        dir_couplers: captured.dir_couplers,
        density_lut,
        halation: captured.halation,
        grain: captured.grain,
        halation_params: pipeline.params.film_render.halation.clone(),
        film_format_mm: pipeline.params.camera.film_format_mm,
    })
}

// ── Capture ────────────────────────────────────────────────────────────────

struct Captured {
    tc_lut: TcLutF32,
    rgb_to_adapted_xyz: [[f32; 3]; 3],
    film_curves: Curve1D,
    dir_couplers: Option<BakedDirCouplers>,
    halation: Option<HalationGpuParams>,
    grain: Option<GrainGpuParams>,
}

/// A CPU backend whose fused-chain hook records its inputs instead of
/// rendering. `try_run_film_chain` returns `None`, so nothing is computed.
struct CaptureBackend {
    cpu: CpuBackend,
    slot: Mutex<Option<Captured>>,
}

impl ComputeBackend for CaptureBackend {
    fn colorspace_convert(&self, img: &ImageBuf, m: &[[f32; 3]; 3]) -> ImageBuf {
        self.cpu.colorspace_convert(img, m)
    }
    fn cctf_encode_srgb(&self, img: &ImageBuf) -> ImageBuf {
        self.cpu.cctf_encode_srgb(img)
    }
    fn cctf_decode_srgb(&self, img: &ImageBuf) -> ImageBuf {
        self.cpu.cctf_decode_srgb(img)
    }
    fn gaussian_blur(&self, img: &ImageBuf, sigma: f32) -> ImageBuf {
        self.cpu.gaussian_blur(img, sigma)
    }
    fn table_lookup(&self, img: &ImageBuf, x: &[f32], y: &[[f32; 3]]) -> ImageBuf {
        self.cpu.table_lookup(img, x, y)
    }
    fn lut3d_interp(&self, img: &ImageBuf, lut: &Lut3D) -> ImageBuf {
        self.cpu.lut3d_interp(img, lut)
    }
    fn name(&self) -> &str {
        "film-sim-capture"
    }

    fn try_run_film_chain(&self, p: &FilmChainParams<'_>) -> Option<ImageBuf> {
        let (tc, m) = match &p.front {
            FrontPass::Hanatos2025 {
                tc_lut,
                rgb_to_adapted_xyz,
            } => (*tc_lut, rgb_to_adapted_xyz),
            FrontPass::Mallett2019 { .. } => return None,
        };
        let film_curves = sample_curves(
            &self.cpu,
            p.film_log_exposure,
            p.film_density_curves_normalized,
            p.film_gamma,
        );
        let dir_couplers = p.dir_couplers.as_ref().map(|d| BakedDirCouplers {
            matrix: d.couplers_matrix_scaled,
            density_max: d.density_max,
            is_positive: d.is_positive,
            curves_0: sample_curves(
                &self.cpu,
                d.log_exposure,
                d.density_curves_0,
                d.gamma_factor,
            ),
            diffusion_size_px: d.diffusion_size_px,
            diffusion_tail_px: d.diffusion_tail_px,
            diffusion_tail_weight: d.diffusion_tail_weight,
        });
        *self.slot.lock().unwrap() = Some(Captured {
            tc_lut: tc_to_f32(tc),
            rgb_to_adapted_xyz: m.map(|r| r.map(|v| v as f32)),
            film_curves,
            dir_couplers,
            halation: p.halation,
            grain: p.grain,
        });
        None
    }
}

fn capture_chain(pipeline: &Pipeline) -> Result<Captured, BakeError> {
    let backend = CaptureBackend {
        cpu: CpuBackend,
        slot: Mutex::new(None),
    };
    // Only the dimensions matter (pixel size in µm); auto-exposure is off.
    let probe = ImageBuf::new(BAKE_REFERENCE_LONG_EDGE, BAKE_REFERENCE_LONG_EDGE * 2 / 3);
    let _ = pipeline.process_resident_borrowed(&probe, &backend);
    backend.slot.into_inner().unwrap().ok_or(BakeError::Capture)
}

fn tc_to_f32(tc: &TcLut) -> TcLutF32 {
    assert_eq!(tc.channels, 3);
    TcLutF32 {
        size: tc.size as u32,
        data: tc
            .data
            .chunks_exact(3)
            .map(|c| [c[0] as f32, c[1] as f32, c[2] as f32])
            .collect(),
    }
}

/// Resample `curves(log_exposure / gamma)` on a uniform grid through the
/// backend's own `density_curve_interp`, so the semantics (clamping, NaN
/// handling) are exactly those of the CPU reference.
fn sample_curves(
    cpu: &CpuBackend,
    log_exposure: &[f64],
    curves: &[[f64; 3]],
    gamma: f64,
) -> Curve1D {
    let lo = log_exposure.iter().cloned().fold(f64::INFINITY, f64::min) / gamma;
    let hi = log_exposure
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        / gamma;
    let (lo, hi) = (lo.min(hi), lo.max(hi));
    let n = CURVE_SAMPLES;
    let mut ramp = ImageBuf::new(n as u32, 1);
    for i in 0..n {
        let x = lo + (hi - lo) * i as f64 / (n - 1) as f64;
        let v: Scalar = crate::math::precision::from_f64(x);
        ramp.data[i * 3] = v;
        ramp.data[i * 3 + 1] = v;
        ramp.data[i * 3 + 2] = v;
    }
    let out = cpu.density_curve_interp(&ramp, log_exposure, curves, gamma);
    Curve1D {
        x_min: lo as f32,
        x_max: hi as f32,
        values: out
            .data
            .chunks_exact(3)
            .map(|c| [to_f32(c[0]), to_f32(c[1]), to_f32(c[2])])
            .collect(),
    }
}

// ── Density → output LUT ───────────────────────────────────────────────────

fn bake_density_lut(pipeline: &Pipeline, cap: &Captured, size: u32) -> DensityLut {
    // Density range the film can reach, over both curve sets, with margin
    // for grain excursions.
    let mut dmin = [f32::INFINITY; 3];
    let mut dmax = [f32::NEG_INFINITY; 3];
    let mut scan = |curve: &Curve1D| {
        for v in &curve.values {
            for c in 0..3 {
                if v[c].is_finite() {
                    dmin[c] = dmin[c].min(v[c]);
                    dmax[c] = dmax[c].max(v[c]);
                }
            }
        }
    };
    scan(&cap.film_curves);
    if let Some(d) = &cap.dir_couplers {
        scan(&d.curves_0);
    }
    for c in 0..3 {
        let span = (dmax[c] - dmin[c]).max(1e-3);
        dmin[c] -= span * 0.02;
        dmax[c] += span * 0.05;
    }

    // Spatial effects in the print/scan stages run as separate passes.
    let mut lut_pipeline = pipeline.clone();
    {
        let p = &mut lut_pipeline.params;
        p.print_render.glare.active = false;
        p.film_render.glare.active = false;
        p.scanner.unsharp_mask = [0.0, 0.0];
        p.scanner.lens_blur = 0.0;
        p.enlarger.lens_blur = 0.0;
        p.enlarger.diffusion_filter.active = false;
    }

    let n = size as usize;
    let mut grid = ImageBuf::new(size, size * size);
    let step = |c: usize, i: usize| dmin[c] + (dmax[c] - dmin[c]) * i as f32 / (n - 1) as f32;
    for b in 0..n {
        for g in 0..n {
            for r in 0..n {
                let idx = ((b * n + g) * n + r) * 3;
                grid.data[idx] = from_f32(step(0, r));
                grid.data[idx + 1] = from_f32(step(1, g));
                grid.data[idx + 2] = from_f32(step(2, b));
            }
        }
    }
    let out = lut_pipeline.render_from_film_density(&grid, &CpuBackend);
    DensityLut {
        size,
        density_min: dmin,
        density_max: dmax,
        data: out
            .data
            .chunks_exact(3)
            .map(|c| [to_f32(c[0]), to_f32(c[1]), to_f32(c[2])])
            .collect(),
    }
}

// ── CPU reference of the shader ────────────────────────────────────────────

impl BakedFilm {
    /// Apply the per-pixel part of the baked chain exactly as RapidRAW's
    /// shader does (no spatial effects, DIR without diffusion). Reference
    /// for the WGSL port and for measuring the baking error.
    pub fn apply_pixelwise(&self, rgb: [f32; 3]) -> [f32; 3] {
        self.apply_from_raw(self.front(rgb))
    }

    /// The part of the chain after the front pass: film raw exposure →
    /// density (curves, DIR couplers) → print/scan LUT.
    pub fn apply_from_raw(&self, raw: [f32; 3]) -> [f32; 3] {
        let log_raw = raw.map(|v| (v.max(0.0) + 1e-10).log10());
        let mut density = [0.0f32; 3];
        for c in 0..3 {
            density[c] = self.film_curves.eval(log_raw[c], c);
        }
        if let Some(d) = &self.dir_couplers {
            let silver = if d.is_positive {
                [
                    d.density_max[0] - density[0],
                    d.density_max[1] - density[1],
                    d.density_max[2] - density[2],
                ]
            } else {
                density
            };
            for m in 0..3 {
                let corr = silver[0] * d.matrix[0][m]
                    + silver[1] * d.matrix[1][m]
                    + silver[2] * d.matrix[2][m];
                density[m] = d.curves_0.eval(log_raw[m] - corr, m);
            }
        }
        self.density_lut.sample(density)
    }

    /// Linear input RGB → film raw exposure (Hanatos2025 front pass).
    pub fn front(&self, rgb: [f32; 3]) -> [f32; 3] {
        let m = &self.rgb_to_adapted_xyz;
        let xyz = [
            m[0][0] * rgb[0] + m[0][1] * rgb[1] + m[0][2] * rgb[2],
            m[1][0] * rgb[0] + m[1][1] * rgb[1] + m[1][2] * rgb[2],
            m[2][0] * rgb[0] + m[2][1] * rgb[1] + m[2][2] * rgb[2],
        ];
        let b = xyz[0] + xyz[1] + xyz[2];
        if b <= 1e-10 {
            return [0.0; 3];
        }
        let (xc, yc) = (xyz[0] / b, xyz[1] / b);
        let omx = (1.0 - xc).max(1e-10);
        let tx = ((1.0 - xc) * (1.0 - xc)).clamp(0.0, 1.0);
        let ty = (yc / omx).clamp(0.0, 1.0);
        let scale = (self.tc_lut.size - 1) as f32;
        let v = self.tc_lut.sample_bicubic(ty * scale, tx * scale);
        [v[0] * b, v[1] * b, v[2] * b]
    }
}

impl TcLutF32 {
    /// Mitchell-Netravali (B=C=1/3) bicubic with reflected borders — port of
    /// `sample_lut_bicubic` in spektrafilm-rs `hanatos2025_rgb_to_raw.wgsl`.
    pub fn sample_bicubic(&self, lut_x: f32, lut_y: f32) -> [f32; 3] {
        fn mitchell(t: f32) -> f32 {
            let at = t.abs();
            let (b, c) = (1.0 / 3.0, 1.0 / 3.0);
            if at < 1.0 {
                ((12.0 - 9.0 * b - 6.0 * c) * at * at * at
                    + (-18.0 + 12.0 * b + 6.0 * c) * at * at
                    + (6.0 - 2.0 * b))
                    / 6.0
            } else if at < 2.0 {
                ((-b - 6.0 * c) * at * at * at
                    + (6.0 * b + 30.0 * c) * at * at
                    + (-12.0 * b - 48.0 * c) * at
                    + (8.0 * b + 24.0 * c))
                    / 6.0
            } else {
                0.0
            }
        }
        fn reflect(i: i32, n: i32) -> usize {
            let mut idx = i.abs();
            let period = 2 * (n - 1);
            if period > 0 {
                idx %= period;
                if idx >= n {
                    idx = period - idx;
                }
            } else {
                idx = 0;
            }
            idx.clamp(0, n - 1) as usize
        }
        let n = self.size as i32;
        let max_xy = (n - 1) as f32;
        let xf = lut_x.clamp(0.0, max_xy);
        let yf = lut_y.clamp(0.0, max_xy);
        let xi = (xf.floor() as i32).min(n - 2);
        let yi = (yf.floor() as i32).min(n - 2);
        let (fx, fy) = (xf - xi as f32, yf - yi as f32);
        let mut sum = [0.0f32; 3];
        let mut wsum = 0.0f32;
        for dy in 0..4 {
            let sy = reflect(yi + dy - 1, n);
            let wy = mitchell(fy - (dy - 1) as f32);
            for dx in 0..4 {
                let sx = reflect(xi + dx - 1, n);
                let w = mitchell(fx - (dx - 1) as f32) * wy;
                wsum += w;
                let v = self.data[sy * self.size as usize + sx];
                for c in 0..3 {
                    sum[c] += w * v[c];
                }
            }
        }
        if wsum != 0.0 {
            for v in &mut sum {
                *v /= wsum;
            }
        }
        sum
    }
}

impl DensityLut {
    /// Trilinear lookup with edge clamping.
    pub fn sample(&self, d: [f32; 3]) -> [f32; 3] {
        let n = self.size as usize;
        let mut i0 = [0usize; 3];
        let mut f = [0.0f32; 3];
        for c in 0..3 {
            let t = ((d[c] - self.density_min[c]) / (self.density_max[c] - self.density_min[c]))
                .clamp(0.0, 1.0)
                * (n - 1) as f32;
            i0[c] = (t.floor() as usize).min(n - 2);
            f[c] = t - i0[c] as f32;
        }
        let at = |r: usize, g: usize, b: usize| self.data[(b * n + g) * n + r];
        let mut out = [0.0f32; 3];
        for (db, wb) in [(0, 1.0 - f[2]), (1, f[2])] {
            for (dg, wg) in [(0, 1.0 - f[1]), (1, f[1])] {
                for (dr, wr) in [(0, 1.0 - f[0]), (1, f[0])] {
                    let v = at(i0[0] + dr, i0[1] + dg, i0[2] + db);
                    let w = wr * wg * wb;
                    for c in 0..3 {
                        out[c] += w * v[c];
                    }
                }
            }
        }
        out
    }
}
