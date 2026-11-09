//! Checks for the RapidRAW-specific parts of film-sim (baking, scatter field,
//! look settings) against spektrafilm's full CPU pipeline.

use std::path::{Path, PathBuf};

use film_sim::backend::cpu_backend::CpuBackend;
use film_sim::bake::{self, BakedFilm};
use film_sim::engine::params::RuntimeParams;
use film_sim::look::{self, FilmLookSettings};
use film_sim::math::image::ImageBuf;
use film_sim::math::precision::{from_f32, to_f32};
use film_sim::placement;
use film_sim::scatter::{self, SceneProxy};

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("data")
}

/// Linear test scene: a hue/brightness sweep (deep shadows to ~3 stops over
/// mid-grey) with a small, very bright highlight for the scatter tests.
fn test_scene(w: u32, h: u32) -> Vec<[f32; 3]> {
    let mut px = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let fx = x as f32 / (w - 1) as f32;
            let fy = y as f32 / (h - 1) as f32;
            let level = 0.18 * 2f32.powf(-6.0 + 9.0 * fx);
            let hue = fy * std::f32::consts::TAU;
            let rgb = [0.0f32, 2.094, 4.189].map(|o| level * (1.0 + 0.6 * (hue + o).cos()));
            let dx = x as i32 - (w / 2) as i32;
            let dy = y as i32 - (h / 2) as i32;
            px.push(if dx * dx + dy * dy <= 4 {
                [40.0; 3]
            } else {
                rgb
            });
        }
    }
    px
}

fn to_buf(w: u32, h: u32, px: &[[f32; 3]]) -> ImageBuf {
    ImageBuf::from_data(w, h, px.iter().flatten().map(|&v| from_f32(v)).collect())
}

/// Spatial effects off: what the per-pixel baked chain models on its own.
fn pixelwise_params() -> RuntimeParams {
    let mut p = FilmLookSettings::default().runtime_params();
    p.film_render.halation.active = false;
    p.film_render.grain.active = false;
    p.film_render.dir_couplers.diffusion_size_um = 0.0;
    p.film_render.glare.active = false;
    p.print_render.glare.active = false;
    p.scanner.unsharp_mask = [0.0, 0.0];
    p
}

fn max_error_255(reference: &ImageBuf, ours: &[f32]) -> f32 {
    assert_eq!(reference.data.len(), ours.len());
    reference
        .data
        .iter()
        .zip(ours)
        .map(|(r, o)| (to_f32(*r) - o).abs())
        .fold(0.0, f32::max)
        * 255.0
}

fn check_baked_matches_pipeline(stock: &str) {
    let (w, h) = (48, 24);
    let scene = test_scene(w, h);
    let pipeline = bake::build_pipeline(&data_dir(), stock, None, pixelwise_params()).unwrap();
    let baked = bake::bake(&pipeline, 49).unwrap();
    let reference = pipeline.process(to_buf(w, h, &scene), &CpuBackend);
    let ours: Vec<f32> = scene
        .iter()
        .flat_map(|p| baked.apply_pixelwise(*p))
        .collect();
    let err = max_error_255(&reference, &ours);
    assert!(err < 1.5, "{stock}: baked chain off by {err:.2}/255");
}

#[test]
fn baked_negative_matches_full_pipeline() {
    check_baked_matches_pipeline("kodak_portra_400");
}

#[test]
fn baked_slide_matches_full_pipeline() {
    check_baked_matches_pipeline("fujifilm_velvia_100");
}

#[test]
fn baked_bw_matches_full_pipeline() {
    check_baked_matches_pipeline("kodak_trix");
}

/// The shader scales the scatter field by exposure instead of recomputing it;
/// that relies on the front pass being homogeneous.
#[test]
fn front_pass_is_homogeneous() {
    let pipeline =
        bake::build_pipeline(&data_dir(), "kodak_portra_400", None, pixelwise_params()).unwrap();
    let baked = bake::bake(&pipeline, 33).unwrap();
    for rgb in [[0.18, 0.18, 0.18], [0.02, 0.3, 0.9], [1.5, 0.4, 0.05]] {
        let base = baked.front(rgb);
        for k in [0.25f32, 4.0] {
            let scaled = baked.front(rgb.map(|v| v * k));
            for c in 0..3 {
                let rel = (scaled[c] - k * base[c]).abs() / (k * base[c]).max(1e-6);
                assert!(rel < 1e-4, "front({rgb:?}·{k})[{c}] off by {rel:e}");
            }
        }
    }
}

/// Sharp part + scattered field reproduces spektrafilm's diffusion filter and
/// halation (both evaluated at the same resolution).
#[test]
fn scatter_split_matches_full_pipeline() {
    let (w, h) = (64, 48);
    let scene = test_scene(w, h);
    let settings = FilmLookSettings::default();
    let df = settings.diffusion_params().expect("default filter");

    let mut params = pixelwise_params();
    params.film_render.halation.active = true;
    params.film_render.halation.scatter_amount = 0.0; // below proxy resolution
    let mut ref_params = params.clone();
    ref_params.camera.diffusion_filter = df.clone();

    let reference = bake::build_pipeline(&data_dir(), &settings.stock, None, ref_params)
        .unwrap()
        .process(to_buf(w, h, &scene), &CpuBackend);

    let pipeline = bake::build_pipeline(&data_dir(), &settings.stock, None, params).unwrap();
    let baked = bake::bake(&pipeline, 49).unwrap();
    let proxy = SceneProxy {
        width: w,
        height: h,
        rgb: &scene,
    };
    let field = scatter::scatter_field(&baked, &proxy, Some(&df), 1.0).expect("field");
    let ours: Vec<f32> = scene
        .iter()
        .zip(&field.data)
        .flat_map(|(p, s)| {
            let r = baked.front(*p);
            baked.apply_from_raw([0, 1, 2].map(|c| field.sharp[c] * r[c] + s[c]))
        })
        .collect();
    let err = max_error_255(&reference, &ours);
    assert!(err < 1.5, "split scatter model off by {err:.2}/255");

    // And the effect itself is visible, so the comparison means something.
    let plain: Vec<f32> = scene
        .iter()
        .flat_map(|p| baked.apply_pixelwise(*p))
        .collect();
    assert!(max_error_255(&reference, &plain) > 5.0);
}

#[test]
fn no_scatter_field_without_filter_or_halation() {
    let baked = default_baked();
    let scene = test_scene(8, 8);
    let proxy = SceneProxy {
        width: 8,
        height: 8,
        rgb: &scene,
    };
    assert!(scatter::scatter_field(&baked, &proxy, None, 0.0).is_none());
}

#[test]
fn auto_exposure_of_mid_grey_is_zero() {
    let grey = vec![[0.184f32; 3]; 32 * 32];
    let proxy = SceneProxy {
        width: 32,
        height: 32,
        rgb: &grey,
    };
    for method in [
        "center_weighted",
        "average",
        "median",
        "partial",
        "matrix",
        "multi_zone",
        "highlight_weighted",
    ] {
        let ev = scatter::auto_exposure_ev(&proxy, method);
        assert!(ev.abs() < 0.05, "{method}: {ev}");
        let dark: Vec<[f32; 3]> = grey.iter().map(|p| p.map(|v| v / 4.0)).collect();
        let ev_dark = scatter::auto_exposure_ev(
            &SceneProxy {
                rgb: &dark,
                ..proxy
            },
            method,
        );
        assert!((ev_dark - 2.0).abs() < 0.05, "{method}: {ev_dark}");
    }
}

/// A camera reference rendered by the same film at a known exposure is
/// matched back to that exposure, from either side of the search centre.
#[test]
fn match_camera_brightness_recovers_known_exposure() {
    let baked = default_baked();
    let (w, h) = (96, 64);
    let scene: Vec<[f32; 3]> = test_scene(w, h)
        .into_iter()
        .map(|p| p.map(|v| v.min(2.0)))
        .collect();
    let source = SceneProxy {
        width: w,
        height: h,
        rgb: &scene,
    };
    for (truth, center) in [(0.75f32, 0.0), (-1.3, 0.5), (0.2, -1.0)] {
        let gain = 2f32.powf(truth);
        let camera: Vec<[f32; 3]> = scene
            .iter()
            .map(|p| {
                baked
                    .apply_pixelwise(p.map(|v| v * gain))
                    .map(|v| v.clamp(0.0, 1.0))
            })
            .collect();
        let reference = SceneProxy {
            width: w,
            height: h,
            rgb: &camera,
        };
        let p = placement::place_exposure(&baked, &source, &reference, center).unwrap();
        assert!(
            (p.exposure_ev - truth as f64).abs() < 0.02,
            "truth {truth}, centre {center}: {p:?}"
        );
    }
}

#[test]
fn match_camera_brightness_refuses_other_framing() {
    let baked = default_baked();
    let scene = test_scene(96, 64);
    let tall = test_scene(64, 96);
    let p = placement::place_exposure(
        &baked,
        &SceneProxy {
            width: 96,
            height: 64,
            rgb: &scene,
        },
        &SceneProxy {
            width: 64,
            height: 96,
            rgb: &tall,
        },
        0.0,
    );
    assert!(p.is_err());
}

#[test]
fn settings_default_from_empty_json() {
    let s: FilmLookSettings = serde_json::from_str("{}").unwrap();
    assert_eq!(s.bake_key(), FilmLookSettings::default().bake_key());
    let s: FilmLookSettings =
        serde_json::from_str(r#"{"stock":"kodak_ektar_100","grain":0.5}"#).unwrap();
    assert_eq!(s.stock, "kodak_ektar_100");
    assert_eq!(s.grain, 0.5);
    assert_eq!(s.diffusion_filter, "black_pro_mist");
}

/// Render-time settings must not force a re-bake; table settings must.
#[test]
fn bake_key_tracks_only_baked_settings() {
    let base = FilmLookSettings::default();
    let runtime_only = FilmLookSettings {
        grain: 1.7,
        halation: 0.2,
        film_exposure: 1.0,
        auto_exposure: false,
        auto_exposure_method: "median".into(),
        diffusion_filter: "glimmerglass".into(),
        diffusion_strength: 1.2,
        ..base.clone()
    };
    assert_eq!(base.bake_key(), runtime_only.bake_key());
    for changed in [
        FilmLookSettings {
            stock: "kodak_ektar_100".into(),
            ..base.clone()
        },
        FilmLookSettings {
            print_exposure: 0.5,
            ..base.clone()
        },
        FilmLookSettings {
            couplers: 0.5,
            ..base.clone()
        },
    ] {
        assert_ne!(base.bake_key(), changed.bake_key());
    }
}

#[test]
fn diffusion_none_or_zero_is_off() {
    for (filter, strength) in [("none", 0.5), ("", 0.5), ("pro_mist", 0.0)] {
        let s = FilmLookSettings {
            diffusion_filter: filter.into(),
            diffusion_strength: strength,
            ..Default::default()
        };
        assert!(s.diffusion_params().is_none(), "{filter} {strength}");
    }
}

#[test]
fn stock_list_classifies_profiles() {
    let stocks = look::list_stocks(&data_dir());
    let kind = |id: &str| {
        stocks
            .iter()
            .find(|s| s.id == id)
            .unwrap_or_else(|| panic!("{id} missing"))
            .kind
            .as_str()
    };
    assert_eq!(kind("kodak_portra_400"), "negative");
    assert_eq!(kind("fujifilm_velvia_100"), "positive");
    assert_eq!(kind("kodak_portra_endura"), "print");
    assert_eq!(kind("kodak_2383"), "print");
    assert!(stocks.iter().any(|s| s.id == "kodak_trix" && s.is_bw));
    // Every camera stock must bake (it is selectable in the UI).
    let default = FilmLookSettings::default();
    assert!(stocks.iter().any(|s| s.id == default.stock));
}

#[test]
fn every_camera_stock_builds() {
    for s in look::list_stocks(&data_dir()) {
        if s.kind == "print" {
            continue;
        }
        let settings = FilmLookSettings {
            stock: s.id.clone(),
            ..Default::default()
        };
        bake::build_pipeline(&data_dir(), &s.id, None, settings.runtime_params())
            .unwrap_or_else(|e| panic!("{}: {e:?}", s.id));
    }
}

fn default_baked() -> BakedFilm {
    let s = FilmLookSettings::default();
    let pipeline = bake::build_pipeline(&data_dir(), &s.stock, None, s.runtime_params()).unwrap();
    bake::bake(&pipeline, 17).unwrap()
}
