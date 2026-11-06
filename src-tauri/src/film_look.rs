//! Spectral film simulation ("Film" tone mapper) using the `film-sim` crate
//! (derived from spektrafilm by Andrea Volpato, see `film-sim/NOTICE.md`).
//!
//! Film looks are baked into small tables on the CPU and evaluated per pixel
//! in the main shader. Exposure is metered on the unedited scene (auto
//! exposure, or matched to the camera's JPEG), so RapidRAW's exposure acts as
//! compensation. The diffusion filter and halation are computed on the CPU
//! from a low-resolution `FILM_PROBE` render of the edited light reaching the
//! film, and reused while a slider is dragged.

use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use film_sim::bake::{self, BakedFilm};
use film_sim::look::{FilmLookSettings, StockInfo};
use film_sim::placement;
use film_sim::scatter::{self, ScatterField, SceneProxy};
use image::DynamicImage;
use serde_json::Value;
use tauri::Manager;

use crate::image_processing::AllAdjustments;

/// 49³ keeps the baked chain within 1/255 of the full spectral pipeline.
const DENSITY_LUT_SIZE: u32 = 49;
const CACHE_ENTRIES: usize = 8;
const FIELD_ENTRIES: usize = 4;
const PROXY_LONG_EDGE: u32 = 640;

pub const TONEMAPPER_MODE_FILM: u32 = 2;

pub struct FilmTables {
    pub key: String,
    pub baked: BakedFilm,
}

pub struct FilmLook {
    pub tables: Arc<FilmTables>,
    pub settings: FilmLookSettings,
    pub source_path: String,
    /// While a slider is dragged, the previous scattering field is reused.
    pub interactive: bool,
    /// The Basic panel's exposure (EV), without the camera's baseline.
    pub user_exposure_ev: f32,
}

static DATA_DIR: OnceLock<PathBuf> = OnceLock::new();
static CACHE: Mutex<Vec<Arc<FilmTables>>> = Mutex::new(Vec::new());

pub fn init(app: &tauri::AppHandle) {
    let resolved = app
        .path()
        .resolve("film-sim/data", tauri::path::BaseDirectory::Resource)
        .ok()
        .filter(|p| p.join("profiles").is_dir());
    let dir = resolved.unwrap_or_else(|| {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("film-sim")
            .join("data")
    });
    log::info!("Film simulation data: {}", dir.display());
    let _ = DATA_DIR.set(dir);
}

fn data_dir() -> PathBuf {
    DATA_DIR.get().cloned().unwrap_or_else(|| {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("film-sim")
            .join("data")
    })
}

pub fn is_film_tonemapper(adjustments: &Value) -> bool {
    adjustments["toneMapper"].as_str() == Some("film")
        && crate::image_processing::is_tool_active(adjustments, "film")
}

pub fn settings_from_adjustments(adjustments: &Value) -> FilmLookSettings {
    adjustments
        .get("filmSim")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default()
}

/// `None` when the film tone mapper is off or the look cannot be built.
pub fn resolve(adjustments: &Value, source_path: &str, interactive: bool) -> Option<Arc<FilmLook>> {
    if !is_film_tonemapper(adjustments) {
        return None;
    }
    let settings = settings_from_adjustments(adjustments);
    let tables = tables_for(&settings)?;
    Some(Arc::new(FilmLook {
        tables,
        settings,
        source_path: source_path.to_string(),
        interactive,
        user_exposure_ev: crate::image_processing::user_exposure_ev(adjustments),
    }))
}

fn tables_for(settings: &FilmLookSettings) -> Option<Arc<FilmTables>> {
    let key = settings.bake_key();
    if let Some(hit) = CACHE.lock().unwrap().iter().find(|t| t.key == key) {
        return Some(Arc::clone(hit));
    }

    let started = std::time::Instant::now();
    let baked = bake::build_pipeline(
        &data_dir(),
        &settings.stock,
        settings.paper.as_deref(),
        settings.runtime_params(),
    )
    .and_then(|pipeline| bake::bake(&pipeline, DENSITY_LUT_SIZE));
    let baked = match baked {
        Ok(b) => b,
        Err(e) => {
            log::error!("Film simulation bake failed for {}: {e}", settings.stock);
            return None;
        }
    };
    log::info!(
        "Baked film look {} in {} ms",
        settings.stock,
        started.elapsed().as_millis()
    );

    let tables = Arc::new(FilmTables { key, baked });
    let mut cache = CACHE.lock().unwrap();
    if let Some(hit) = cache.iter().find(|t| t.key == tables.key) {
        return Some(Arc::clone(hit));
    }
    if cache.len() >= CACHE_ENTRIES {
        cache.remove(0);
    }
    cache.push(Arc::clone(&tables));
    Some(tables)
}

#[tauri::command]
pub fn list_film_stocks() -> Vec<StockInfo> {
    film_sim::look::list_stocks(&data_dir())
}

struct PlacementEntry {
    path: String,
    tables_key: String,
    exposure_ev: Option<f32>,
}

const PLACEMENT_ENTRIES: usize = 16;
static PLACEMENTS: Mutex<Vec<PlacementEntry>> = Mutex::new(Vec::new());

/// The solved camera-brightness exposure: `None` = not solved yet,
/// `Some(None)` = the camera preview cannot be used.
fn placement_ev(look: &FilmLook) -> Option<Option<f32>> {
    PLACEMENTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .find(|e| e.path == look.source_path && e.tables_key == look.tables.key)
        .map(|e| e.exposure_ev)
}

pub fn ensure_camera_placement(look: &FilmLook, image: &DynamicImage, is_raw: bool) {
    if !look.settings.match_camera_brightness
        || look.source_path.is_empty()
        || placement_ev(look).is_some()
    {
        return;
    }
    let started = std::time::Instant::now();
    let exposure_ev = match camera_placement(look, image, is_raw) {
        Ok(ev) => {
            log::info!(
                "Film: camera brightness matched at {ev:+.2} EV in {} ms",
                started.elapsed().as_millis()
            );
            Some(ev)
        }
        Err(e) => {
            log::warn!(
                "Film: cannot match camera brightness for {} ({e}); using auto exposure",
                look.source_path
            );
            None
        }
    };
    let mut cache = PLACEMENTS.lock().unwrap_or_else(|e| e.into_inner());
    if cache.len() >= PLACEMENT_ENTRIES {
        cache.remove(0);
    }
    cache.push(PlacementEntry {
        path: look.source_path.clone(),
        tables_key: look.tables.key.clone(),
        exposure_ev,
    });
}

/// Area-average `image` down to at most `edge` pixels. (`DynamicImage::
/// thumbnail` does not average float images correctly.)
fn thumbnail_rgb(image: &DynamicImage, edge: u32) -> (u32, u32, Vec<[f32; 3]>) {
    let converted;
    let rgb = match image.as_rgb32f() {
        Some(rgb) => rgb,
        None => {
            converted = image.to_rgb32f();
            &converted
        }
    };
    let (w, h) = rgb.dimensions();
    placement::downsample(w, h, bytemuck::cast_slice(rgb.as_raw()), edge)
}

fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn camera_placement(look: &FilmLook, image: &DynamicImage, is_raw: bool) -> Result<f32, String> {
    let (sw, sh, pixels) = thumbnail_rgb(image, placement::EDGE);
    // The camera's rendering of the frame: a RAW's embedded preview, or the
    // display-referred file itself.
    let (rw, rh, reference) = if is_raw {
        let (file, _) = crate::file_management::parse_virtual_path(&look.source_path);
        let bytes = std::fs::read(&file).map_err(|e| e.to_string())?;
        let preview =
            crate::image_loader::safe_embedded_preview_fallback(&bytes, &look.source_path)
                .ok_or("the RAW has no camera preview")?;
        thumbnail_rgb(&preview, 2 * placement::EDGE)
    } else {
        (sw, sh, pixels.clone())
    };
    let linear: Vec<[f32; 3]> = if is_raw {
        pixels
    } else {
        pixels.iter().map(|p| p.map(srgb_to_linear)).collect()
    };
    let source = SceneProxy {
        width: sw,
        height: sh,
        rgb: &linear,
    };
    let center = scatter::auto_exposure_ev(&source, &look.settings.auto_exposure_method);
    let reference = SceneProxy {
        width: rw,
        height: rh,
        rgb: &reference,
    };
    placement::place_exposure(&look.tables.baked, &source, &reference, center as f64)
        .map(|p| p.exposure_ev as f32)
}

/// Uniform block mirrored by `FilmParams` in `shader.wgsl`.
#[repr(C)]
#[derive(Debug, Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FilmParamsGpu {
    /// Rows of the linear-sRGB → CAT-adapted XYZ matrix (exposure 0).
    rgb_to_xyz: [[f32; 4]; 3],
    /// (film x_min, film x_max, DIR x_min, DIR x_max) of the curve rows.
    curve_range: [f32; 4],
    density_min: [f32; 4],
    density_max: [f32; 4],
    /// Rows k of the DIR matrix: correction[m] = Σ_k silver[k] · M[k][m].
    dir_matrix: [[f32; 4]; 3],
    dir_density_max: [f32; 4],
    /// (tc LUT size, density LUT size, has DIR couplers, is positive film).
    info: [u32; 4],
    /// Grain (spektrafilm particle model, normal approximation):
    /// .xyz density_min, .w user amount (0 = off).
    grain_dmin: [f32; 4],
    /// .xyz density_max (incl. density_min), .w number of sub-layers.
    grain_dmax: [f32; 4],
    /// .xyz particles per pixel per sub-layer at `grain_ref.w`, .w monochrome.
    grain_npp: [f32; 4],
    /// .xyz grain uniformity, .w long edge (px) the particle counts are for.
    grain_ref: [f32; 4],
    /// .xyz weight of the unscattered light per channel, .w 1 = scatter
    /// texture present.
    scatter: [f32; 4],
    /// .x film exposure scale (2^(film exposure + auto-exposure)).
    exposure: [f32; 4],
}

pub struct FilmTableTextures {
    tc_lut_view: wgpu::TextureView,
    curves_view: wgpu::TextureView,
    density_lut_view: wgpu::TextureView,
}

pub struct FilmBindings {
    tables: Arc<FilmTableTextures>,
    scatter_view: wgpu::TextureView,
    params_buffer: wgpu::Buffer,
}

pub const FIRST_BINDING: u32 = 16;

pub fn layout_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
    let texture = |binding: u32, dim: wgpu::TextureViewDimension| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: dim,
            multisampled: false,
        },
        count: None,
    };
    vec![
        texture(FIRST_BINDING, wgpu::TextureViewDimension::D2),
        texture(FIRST_BINDING + 1, wgpu::TextureViewDimension::D2),
        texture(FIRST_BINDING + 2, wgpu::TextureViewDimension::D3),
        wgpu::BindGroupLayoutEntry {
            binding: FIRST_BINDING + 3,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        texture(FIRST_BINDING + 4, wgpu::TextureViewDimension::D2),
    ]
}

impl FilmBindings {
    pub fn bind_group_entries(&self) -> [wgpu::BindGroupEntry<'_>; 5] {
        [
            wgpu::BindGroupEntry {
                binding: FIRST_BINDING,
                resource: wgpu::BindingResource::TextureView(&self.tables.tc_lut_view),
            },
            wgpu::BindGroupEntry {
                binding: FIRST_BINDING + 1,
                resource: wgpu::BindingResource::TextureView(&self.tables.curves_view),
            },
            wgpu::BindGroupEntry {
                binding: FIRST_BINDING + 2,
                resource: wgpu::BindingResource::TextureView(&self.tables.density_lut_view),
            },
            wgpu::BindGroupEntry {
                binding: FIRST_BINDING + 3,
                resource: self.params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: FIRST_BINDING + 4,
                resource: wgpu::BindingResource::TextureView(&self.scatter_view),
            },
        ]
    }
}

fn params_for(look: &FilmLook, exposure_ev: f32, field: Option<&ScatterField>) -> FilmParamsGpu {
    let b = &look.tables.baked;
    let s = &look.settings;
    let m = &b.rgb_to_adapted_xyz;
    let row4 = |r: [f32; 3], w: f32| [r[0], r[1], r[2], w];
    let (dir_matrix, dir_dmax, has_dir, positive, dir_lo, dir_hi) = match &b.dir_couplers {
        Some(d) => (
            d.matrix.map(|r| row4(r, 0.0)),
            row4(d.density_max, 0.0),
            1,
            d.is_positive as u32,
            d.curves_0.x_min,
            d.curves_0.x_max,
        ),
        None => (
            [[0.0; 4]; 3],
            [0.0; 4],
            0,
            0,
            b.film_curves.x_min,
            b.film_curves.x_max,
        ),
    };
    let grain = s.grain.clamp(0.0, 2.0);
    let (grain_dmin, grain_dmax, grain_npp, grain_ref) = match (&b.grain, grain > 0.0) {
        (Some(g), true) => (
            row4(g.density_min, grain),
            row4(g.density_max, g.n_sub_layers.max(1) as f32),
            row4(g.n_particles_per_pixel, g.monochrome as u32 as f32),
            row4(g.grain_uniformity, bake::BAKE_REFERENCE_LONG_EDGE as f32),
        ),
        _ => ([0.0; 4], [1.0; 4], [1.0; 4], [0.0; 4]),
    };
    let scatter = match field {
        Some(f) => row4(f.sharp, 1.0),
        None => [1.0, 1.0, 1.0, 0.0],
    };
    FilmParamsGpu {
        rgb_to_xyz: [row4(m[0], 0.0), row4(m[1], 0.0), row4(m[2], 0.0)],
        curve_range: [b.film_curves.x_min, b.film_curves.x_max, dir_lo, dir_hi],
        density_min: row4(b.density_lut.density_min, 0.0),
        density_max: row4(b.density_lut.density_max, 0.0),
        dir_matrix,
        dir_density_max: dir_dmax,
        info: [b.tc_lut.size, b.density_lut.size, has_dir, positive],
        grain_dmin,
        grain_dmax,
        grain_npp,
        grain_ref,
        scatter,
        exposure: [2f32.powf(exposure_ev), 0.0, 0.0, 0.0],
    }
}

fn texture_view(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    size: (u32, u32, u32),
    dimension: wgpu::TextureDimension,
    data: &[[f32; 4]],
) -> wgpu::TextureView {
    use wgpu::util::{DeviceExt, TextureDataOrder};
    device
        .create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size.0,
                    height: size.1,
                    depth_or_array_layers: size.2,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            TextureDataOrder::MipMajor,
            bytemuck::cast_slice(data),
        )
        .create_view(&Default::default())
}

fn rgba(v: &[[f32; 3]]) -> Vec<[f32; 4]> {
    v.iter().map(|p| [p[0], p[1], p[2], 1.0]).collect()
}

impl FilmTableTextures {
    fn upload(device: &wgpu::Device, queue: &wgpu::Queue, tables: &FilmTables) -> Self {
        let b = &tables.baked;
        // Row 0: film density curves; row 1: DIR "curves before couplers".
        let n = b.film_curves.values.len();
        let mut curves = rgba(&b.film_curves.values);
        match &b.dir_couplers {
            Some(d) => {
                assert_eq!(d.curves_0.values.len(), n);
                curves.extend(rgba(&d.curves_0.values));
            }
            None => curves.extend(rgba(&b.film_curves.values)),
        }
        let d2 = wgpu::TextureDimension::D2;
        Self {
            tc_lut_view: texture_view(
                device,
                queue,
                "Film tc LUT",
                (b.tc_lut.size, b.tc_lut.size, 1),
                d2,
                &rgba(&b.tc_lut.data),
            ),
            curves_view: texture_view(device, queue, "Film curves", (n as u32, 2, 1), d2, &curves),
            density_lut_view: texture_view(
                device,
                queue,
                "Film density LUT",
                (b.density_lut.size, b.density_lut.size, b.density_lut.size),
                wgpu::TextureDimension::D3,
                &rgba(&b.density_lut.data),
            ),
        }
    }

    fn dummy(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let one = [[0.0f32; 4]];
        let d2 = wgpu::TextureDimension::D2;
        Self {
            tc_lut_view: texture_view(device, queue, "Film dummy", (1, 1, 1), d2, &one),
            curves_view: texture_view(device, queue, "Film dummy", (1, 1, 1), d2, &one),
            density_lut_view: texture_view(
                device,
                queue,
                "Film dummy",
                (1, 1, 1),
                wgpu::TextureDimension::D3,
                &one,
            ),
        }
    }
}

pub struct Proxy {
    width: u32,
    height: u32,
    rgb: Vec<[f32; 3]>,
}

impl Proxy {
    pub fn from_f16_bits(bits: &[u16], width: u32, height: u32) -> Self {
        use rayon::prelude::*;
        let factor = width.max(height).max(1).div_ceil(PROXY_LONG_EDGE).max(1);
        let (w, h) = (width.div_ceil(factor), height.div_ceil(factor));
        let px = |x: u32, y: u32, c: usize| {
            let v = half::f16::from_bits(bits[(y * width + x) as usize * 4 + c]).to_f32();
            if v.is_finite() { v.max(0.0) } else { 0.0 }
        };
        let rgb = (0..h)
            .into_par_iter()
            .flat_map_iter(|oy| {
                (0..w).map(move |ox| {
                    let mut sum = [0.0f32; 3];
                    let mut n = 0.0f32;
                    for y in oy * factor..((oy + 1) * factor).min(height) {
                        for x in ox * factor..((ox + 1) * factor).min(width) {
                            for (c, s) in sum.iter_mut().enumerate() {
                                *s += px(x, y, c);
                            }
                            n += 1.0;
                        }
                    }
                    sum.map(|s| s / n.max(1.0))
                })
            })
            .collect();
        Self {
            width: w,
            height: h,
            rgb,
        }
    }

    fn scene(&self) -> SceneProxy<'_> {
        SceneProxy {
            width: self.width,
            height: self.height,
            rgb: &self.rgb,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ProxyParams {
    factor: u32,
    is_raw: u32,
    out_width: u32,
    out_height: u32,
}

struct ProxyPipeline {
    bgl: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl ProxyPipeline {
    fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Film Proxy Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/film_proxy.wgsl").into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Film Proxy BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Film Proxy Layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Film Proxy Pipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { bgl, pipeline }
    }

    fn run(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &crate::GpuImageEntry,
    ) -> Result<Proxy, String> {
        use wgpu::util::DeviceExt;
        let long = input.width.max(input.height).max(1);
        let factor = long.div_ceil(PROXY_LONG_EDGE).max(1);
        let (w, h) = (input.width.div_ceil(factor), input.height.div_ceil(factor));
        let size = wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Film Proxy"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let params = ProxyParams {
            factor,
            is_raw: input.is_raw,
            out_width: w,
            out_height: h,
        };
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Film Proxy Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Film Proxy BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&input.texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
        }
        queue.submit(Some(encoder.finish()));
        let bytes = crate::gpu_processing::read_texture_data_roi(
            device,
            queue,
            &texture,
            wgpu::Origin3d::ZERO,
            size,
            16,
        )?;
        let px: &[[f32; 4]] = bytemuck::cast_slice(&bytes);
        Ok(Proxy {
            width: w,
            height: h,
            rgb: px.iter().map(|p| [p[0], p[1], p[2]]).collect(),
        })
    }
}

#[derive(PartialEq)]
struct FieldKey {
    scene: u64,
    tables_key: String,
    diffusion: String,
    strength: u32,
    halation: u32,
}

struct CachedField {
    key: FieldKey,
    edits: u64,
    field: Option<Arc<ScatterField>>,
    view: Option<wgpu::TextureView>,
}

/// The framed source, independent of the preview resolution.
fn scene_id(input: &crate::GpuImageEntry) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    (input.source_identity, input.transform_key).hash(&mut hasher);
    hasher.finish()
}

fn edits_hash(adjustments: &AllAdjustments, mask_cache_key: Option<u64>) -> u64 {
    let mut a = *adjustments;
    a.tile_offset_x = 0;
    a.tile_offset_y = 0;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    bytemuck::bytes_of(&a).hash(&mut hasher);
    mask_cache_key.hash(&mut hasher);
    hasher.finish()
}

#[derive(Default)]
struct RendererCache {
    tables: Option<(Arc<FilmTables>, Arc<FilmTableTextures>)>,
    proxy: Option<(u64, Arc<Proxy>)>,
    auto_exposure: Vec<((u64, String), f32)>,
    /// Most recently used last; the editor and the thumbnails share it.
    fields: Vec<CachedField>,
}

pub struct FilmRenderer {
    dummy_tables: Arc<FilmTableTextures>,
    dummy_scatter: wgpu::TextureView,
    dummy_params: wgpu::Buffer,
    proxy_pipeline: OnceLock<ProxyPipeline>,
    cache: Mutex<RendererCache>,
}

impl FilmRenderer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        use wgpu::util::DeviceExt;
        let one = [[0.0f32; 4]];
        Self {
            dummy_tables: Arc::new(FilmTableTextures::dummy(device, queue)),
            dummy_scatter: texture_view(
                device,
                queue,
                "Film dummy",
                (1, 1, 1),
                wgpu::TextureDimension::D2,
                &one,
            ),
            dummy_params: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Film params (dummy)"),
                contents: bytemuck::bytes_of(&FilmParamsGpu::default()),
                usage: wgpu::BufferUsages::UNIFORM,
            }),
            proxy_pipeline: OnceLock::new(),
            cache: Mutex::new(RendererCache::default()),
        }
    }

    pub fn dummy(&self) -> FilmBindings {
        FilmBindings {
            tables: Arc::clone(&self.dummy_tables),
            scatter_view: self.dummy_scatter.clone(),
            params_buffer: self.dummy_params.clone(),
        }
    }

    /// `probe` renders the light reaching the film; it is only called when
    /// the scattering field is stale.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        look: &FilmLook,
        input: &crate::GpuImageEntry,
        adjustments: &AllAdjustments,
        mask_cache_key: Option<u64>,
        probe: impl FnOnce() -> Result<Proxy, String>,
    ) -> FilmBindings {
        use wgpu::util::DeviceExt;
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        let s = &look.settings;

        let tables = match &cache.tables {
            Some((t, tex)) if Arc::ptr_eq(t, &look.tables) => Arc::clone(tex),
            _ => {
                let tex = Arc::new(FilmTableTextures::upload(device, queue, &look.tables));
                cache.tables = Some((Arc::clone(&look.tables), Arc::clone(&tex)));
                tex
            }
        };

        // Metered exposure replaces the camera's baseline exposure, which is
        // part of `global.exposure`.
        let metering = s.auto_exposure || s.match_camera_brightness;
        let placed = if s.match_camera_brightness {
            placement_ev(look).flatten()
        } else {
            None
        };
        let metered_ev = match placed {
            Some(ev) => Some(ev),
            None if metering => self.proxy(&mut cache, device, queue, input).map(|proxy| {
                Self::auto_exposure(&mut cache, scene_id(input), &proxy, &s.auto_exposure_method)
            }),
            None => None,
        };
        let baseline_ev = adjustments.global.exposure - look.user_exposure_ev;
        let exposure_ev = s.film_exposure + metered_ev.map_or(0.0, |ev| ev - baseline_ev);

        let diffusion = s.diffusion_params();
        let wants_scatter =
            diffusion.is_some() || (s.halation > 0.0 && look.tables.baked.halation.is_some());
        let (field, scatter_view) = if wants_scatter {
            let key = FieldKey {
                scene: scene_id(input),
                tables_key: look.tables.key.clone(),
                diffusion: diffusion
                    .as_ref()
                    .map(|d| d.filter_family.clone())
                    .unwrap_or_default(),
                strength: diffusion.as_ref().map_or(0.0, |d| d.strength).to_bits(),
                halation: s.halation.to_bits(),
            };
            let edits = edits_hash(adjustments, mask_cache_key);
            let slot = cache.fields.iter().position(|f| f.key == key);
            let current = slot.is_some_and(|i| cache.fields[i].edits == edits || look.interactive);
            if current {
                let entry = cache.fields.remove(slot.unwrap());
                cache.fields.push(entry);
            } else {
                let started = std::time::Instant::now();
                let field = match probe() {
                    Ok(proxy) => {
                        // Stored at exposure 0; the shader scales it by the
                        // current exposure (exact for the probed edits).
                        let to_ev0 = (-adjustments.global.exposure).exp2();
                        let rgb: Vec<[f32; 3]> =
                            proxy.rgb.iter().map(|p| p.map(|v| v * to_ev0)).collect();
                        scatter::scatter_field(
                            &look.tables.baked,
                            &SceneProxy {
                                width: proxy.width,
                                height: proxy.height,
                                rgb: &rgb,
                            },
                            diffusion.as_ref(),
                            s.halation.clamp(0.0, 3.0),
                        )
                        .map(Arc::new)
                    }
                    Err(e) => {
                        log::error!("Film probe render failed: {e}");
                        None
                    }
                };
                let view = field.as_ref().map(|f| {
                    texture_view(
                        device,
                        queue,
                        "Film scatter field",
                        (f.width, f.height, 1),
                        wgpu::TextureDimension::D2,
                        &rgba(&f.data),
                    )
                });
                log::info!("Film scatter field in {} ms", started.elapsed().as_millis());
                if let Some(i) = slot {
                    cache.fields.remove(i);
                }
                if cache.fields.len() >= FIELD_ENTRIES {
                    cache.fields.remove(0);
                }
                cache.fields.push(CachedField {
                    key,
                    edits,
                    field,
                    view,
                });
            }
            let cached = cache.fields.last().unwrap();
            (
                cached.field.clone(),
                cached
                    .view
                    .clone()
                    .unwrap_or_else(|| self.dummy_scatter.clone()),
            )
        } else {
            (None, self.dummy_scatter.clone())
        };

        let params = params_for(look, exposure_ev, field.as_deref());
        FilmBindings {
            tables,
            scatter_view,
            params_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Film params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            }),
        }
    }

    fn proxy(
        &self,
        cache: &mut RendererCache,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &crate::GpuImageEntry,
    ) -> Option<Arc<Proxy>> {
        if let Some((id, p)) = &cache.proxy
            && *id == scene_id(input)
        {
            return Some(Arc::clone(p));
        }
        let pipeline = self
            .proxy_pipeline
            .get_or_init(|| ProxyPipeline::new(device));
        match pipeline.run(device, queue, input) {
            Ok(p) => {
                let p = Arc::new(p);
                cache.proxy = Some((scene_id(input), Arc::clone(&p)));
                cache.auto_exposure.clear();
                Some(p)
            }
            Err(e) => {
                log::error!("Film proxy readback failed: {e}");
                None
            }
        }
    }

    fn auto_exposure(cache: &mut RendererCache, scene: u64, proxy: &Proxy, method: &str) -> f32 {
        let key = (scene, method.to_string());
        if let Some((_, ev)) = cache.auto_exposure.iter().find(|(k, _)| *k == key) {
            return *ev;
        }
        let ev = scatter::auto_exposure_ev(&proxy.scene(), method);
        cache.auto_exposure.push((key, ev));
        ev
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thumbnail_keeps_float_image_distribution() {
        let (w, h) = (1000u32, 600u32);
        let image = image::Rgb32FImage::from_fn(w, h, |x, _| {
            let v = 0.001 * 2f32.powf(10.0 * x as f32 / w as f32);
            image::Rgb([v, v, v])
        });
        let (tw, th, rgb) = thumbnail_rgb(&DynamicImage::ImageRgb32F(image), 256);
        assert_eq!((tw, th), (256, 154));
        let first = rgb[0][0];
        let last = rgb[tw as usize - 1][0];
        assert!(first < 0.0011 && last > 0.9, "{first} .. {last}");
        let mean = |v: &[[f32; 3]]| v.iter().map(|p| p[1] as f64).sum::<f64>() / v.len() as f64;
        let expected = 0.001 * (1023.0f64 / (10.0 * std::f64::consts::LN_2));
        assert!((mean(&rgb) / expected - 1.0).abs() < 0.01);
    }
}
