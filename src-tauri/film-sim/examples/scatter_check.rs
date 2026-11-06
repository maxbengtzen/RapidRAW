//! Compare the split "sharp + scattered field" model used by RapidRAW with
//! spektrafilm's full CPU pipeline, with a camera diffusion filter and
//! halation active (grain, DIR diffusion, emulsion scatter, glare off).
//! Both run at the same resolution, so this checks the decomposition, not
//! the GPU upsampling.
//!
//! cargo run --release --example scatter_check -- <in.png> <out_dir> <film> <data_dir> [filter] [strength]

use std::path::Path;

use film_sim::backend::select_backend;
use film_sim::bake;
use film_sim::engine::params::DiffusionFilterParams;
use film_sim::look::FilmLookSettings;
use film_sim::math::image::ImageBuf;
use film_sim::math::precision::{from_f32, srgb_decode, to_f32};
use film_sim::scatter::{self, SceneProxy};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (input, out_dir, film, data_dir) = (&a[1], Path::new(&a[2]), &a[3], Path::new(&a[4]));
    let family = a.get(5).cloned().unwrap_or("black_pro_mist".into());
    let strength: f32 = a.get(6).map(|s| s.parse().unwrap()).unwrap_or(0.5);

    let settings = FilmLookSettings {
        stock: film.clone(),
        ..Default::default()
    };
    let mut params = settings.runtime_params();
    params.film_render.grain.active = false;
    params.film_render.dir_couplers.diffusion_size_um = 0.0;
    params.film_render.halation.scatter_amount = 0.0;
    params.film_render.glare.active = false;
    params.print_render.glare.active = false;
    params.scanner.unsharp_mask = [0.0, 0.0];
    let df = DiffusionFilterParams {
        active: family != "none",
        filter_family: family.clone(),
        strength,
        ..Default::default()
    };

    let rgb = image::open(input).expect("open").to_rgb32f();
    let (w, h) = (rgb.width(), rgb.height());
    let lin: Vec<[f32; 3]> = rgb
        .into_raw()
        .chunks_exact(3)
        .map(|p| [0, 1, 2].map(|c| to_f32(srgb_decode(from_f32(p[c])))))
        .collect();

    // Reference: full pipeline with the diffusion filter in the camera.
    let mut ref_params = params.clone();
    ref_params.camera.diffusion_filter = df.clone();
    let ref_pipeline = bake::build_pipeline(data_dir, film, None, ref_params).unwrap();
    let img = ImageBuf::from_data(w, h, lin.iter().flatten().map(|&v| from_f32(v)).collect());
    let reference = ref_pipeline.process(img, select_backend().as_ref());

    // Ours: baked tables + scatter field at full resolution.
    let pipeline = bake::build_pipeline(data_dir, film, None, params).unwrap();
    let baked = bake::bake(&pipeline, 49).unwrap();
    let proxy = SceneProxy {
        width: w,
        height: h,
        rgb: &lin,
    };
    let t = std::time::Instant::now();
    let field =
        scatter::scatter_field(&baked, &proxy, df.active.then_some(&df), 1.0).expect("field");
    eprintln!(
        "scatter field {}x{}: {} ms, sharp {:?}",
        w,
        h,
        t.elapsed().as_millis(),
        field.sharp
    );
    let ours: Vec<f32> = lin
        .iter()
        .zip(&field.data)
        .flat_map(|(p, s)| {
            let r = baked.front(*p);
            baked.apply_from_raw([0, 1, 2].map(|c| field.sharp[c] * r[c] + s[c]))
        })
        .collect();
    let plain: Vec<f32> = lin.iter().flat_map(|p| baked.apply_pixelwise(*p)).collect();

    let stats = |x: &[f32]| {
        let (mut max, mut sum) = (0.0f32, 0.0f64);
        for (r, o) in reference.data.iter().zip(x) {
            let d = (to_f32(*r) - o).abs();
            max = max.max(d);
            sum += d as f64;
        }
        (max * 255.0, sum / x.len() as f64 * 255.0)
    };
    let (m, e) = stats(&ours);
    let (pm, pe) = stats(&plain);
    println!("{family} {strength}: split model vs full pipeline: max {m:.2}/255 mean {e:.3}/255");
    println!(
        "  (without any scattering: max {pm:.2}/255 mean {pe:.3}/255 — the size of the effect)"
    );
    let save = |name: &str, data: &[f32]| {
        let u8s: Vec<u8> = data
            .iter()
            .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect();
        image::RgbImage::from_raw(w, h, u8s)
            .unwrap()
            .save(out_dir.join(name))
            .unwrap();
    };
    let r: Vec<f32> = reference.data.iter().map(|v| to_f32(*v)).collect();
    save("scatter_reference.png", &r);
    save("scatter_ours.png", &ours);
}
