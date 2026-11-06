//! Spatial light scattering and metering on a low-resolution proxy of the
//! scene. (film-sim addition, not part of spektrafilm-rs.)
//!
//! spektrafilm applies the camera diffusion filter and the film halation as
//! linear operators on the film raw exposure:
//!
//! ```text
//!   raw_d = (1 − p) · raw + p · PSF ∗ raw                  (diffusion filter)
//!   raw_h = (raw_d + a ⊙ Σ_k w_k G_k ∗ raw_d) / (1 + a)    (halation)
//! ```
//!
//! Both are a sharp part `k ⊙ raw` plus a smooth, low-frequency part. The
//! smooth part is computed here, with spektrafilm's own CPU code, on a small
//! proxy of the scene and upsampled by the GPU:
//!
//! ```text
//!   raw_final(x) ≈ k ⊙ raw(x) + upsample(raw_h_small − k ⊙ raw_small)(x)
//! ```
//!
//! The front pass is homogeneous (`raw(s·rgb) = s·raw(rgb)`), so the field
//! scales linearly with exposure.

use rayon::prelude::*;

use crate::backend::cpu_backend::CpuBackend;
use crate::bake::BakedFilm;
use crate::engine::params::DiffusionFilterParams;
use crate::engine::stages::filming;
use crate::math::image::ImageBuf;
use crate::math::precision::{from_f32, to_f32};
use crate::model::diffusion;

/// Smooth scattered-light field in film raw units at exposure 0.
#[derive(Debug, Clone)]
pub struct ScatterField {
    /// Per-channel weight of the sharp (unscattered) light.
    pub sharp: [f32; 3],
    pub width: u32,
    pub height: u32,
    /// `width × height` RGB raw-exposure triples, row-major.
    pub data: Vec<[f32; 3]>,
}

/// Linear scene proxy (RapidRAW pipe space: linear sRGB primaries).
pub struct SceneProxy<'a> {
    pub width: u32,
    pub height: u32,
    pub rgb: &'a [[f32; 3]],
}

/// spektrafilm auto-exposure (`measure_autoexposure_ev`) on the proxy: the
/// EV that brings the metered luminance to mid-grey. `method` is one of
/// `average`, `median`, `center_weighted`, `partial`, `matrix`,
/// `multi_zone`, `highlight_weighted`.
pub fn auto_exposure_ev(proxy: &SceneProxy, method: &str) -> f32 {
    let img = to_image(proxy);
    let rgb_to_xyz = filming::input_colorspace_to_xyz(crate::bake::RAPIDRAW_INPUT_COLOR_SPACE);
    filming::measure_autoexposure_ev(&img, &rgb_to_xyz, method)
}

/// Compute the scattered-light field for `film` from the light reaching it,
/// with the optional camera diffusion filter and the stock halation scaled
/// by `halation_amount`.
pub fn scatter_field(
    film: &BakedFilm,
    proxy: &SceneProxy,
    diffusion_filter: Option<&DiffusionFilterParams>,
    halation_amount: f32,
) -> Option<ScatterField> {
    let df = diffusion_filter.filter(|d| d.active && d.strength > 0.0);
    let halation_on = halation_amount > 0.0 && film.halation.is_some();
    if df.is_none() && !halation_on {
        return None;
    }
    let scene = to_image(proxy);
    let pixel_um = filming::pixel_size_um(film.film_format_mm, proxy.width, proxy.height);

    // Film raw exposure of the proxy (exposure 0).
    let mut raw = ImageBuf::new(proxy.width, proxy.height);
    raw.data
        .par_chunks_exact_mut(3)
        .zip(scene.data.par_chunks_exact(3))
        .for_each(|(out, px)| {
            let r = film.front([to_f32(px[0]), to_f32(px[1]), to_f32(px[2])]);
            for c in 0..3 {
                out[c] = from_f32(r[c]);
            }
        });

    let mut sharp = [1.0f32; 3];
    let mut scattered = raw.clone();

    if let Some(df) = df {
        let model = df.to_model();
        let p = diffusion::diffusion_scatter_fraction(&model) as f32;
        scattered = diffusion::apply_diffusion_filter_um(&scattered, &model, pixel_um as f64);
        for s in &mut sharp {
            *s *= 1.0 - p;
        }
    }

    if halation_on {
        let h = &film.halation_params;
        let amount = h.halation_amount * halation_amount as f64;
        scattered = diffusion::apply_halation_um(
            &scattered,
            pixel_um,
            0.0, // emulsion scatter (2–10 µm) is below the proxy resolution
            h.scatter_spatial_scale,
            h.scatter_core_um,
            h.scatter_tail_um,
            h.scatter_tail_weight,
            amount,
            h.halation_spatial_scale,
            h.halation_strength,
            h.halation_first_sigma_um,
            h.halation_n_bounces,
            h.halation_bounce_decay,
            h.halation_renormalize,
            &CpuBackend,
        );
        if h.halation_renormalize {
            for c in 0..3 {
                sharp[c] /= 1.0 + (h.halation_strength[c] * amount) as f32;
            }
        }
    }

    let data = scattered
        .data
        .par_chunks_exact(3)
        .zip(raw.data.par_chunks_exact(3))
        .map(|(s, r)| {
            let mut o = [0.0f32; 3];
            for c in 0..3 {
                o[c] = (to_f32(s[c]) - sharp[c] * to_f32(r[c])).max(0.0);
            }
            o
        })
        .collect();
    Some(ScatterField {
        sharp,
        width: proxy.width,
        height: proxy.height,
        data,
    })
}

fn to_image(proxy: &SceneProxy) -> ImageBuf {
    let mut img = ImageBuf::new(proxy.width, proxy.height);
    img.data
        .par_chunks_exact_mut(3)
        .zip(proxy.rgb.par_iter())
        .for_each(|(out, p)| {
            for c in 0..3 {
                out[c] = from_f32(p[c].max(0.0));
            }
        });
    img
}
