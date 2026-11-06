//! Measure the error of the baked, per-pixel film chain (what RapidRAW's
//! shader computes) against the full CPU spectral pipeline, with spatial
//! effects disabled on both sides.
//!
//! cargo run --release --example bake_check -- <in.png> <out_dir> <film> <data_dir> [lut_size]

use std::path::Path;
use std::time::Instant;

use film_sim::backend::select_backend;
use film_sim::bake;
use film_sim::engine::params::RuntimeParams;
use film_sim::math::image::ImageBuf;
use film_sim::math::precision::{Scalar, from_f32, srgb_decode, to_f32};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (input, out_dir, film, data_dir) = (&a[1], Path::new(&a[2]), &a[3], Path::new(&a[4]));
    let lut_size: u32 = a.get(5).map(|s| s.parse().unwrap()).unwrap_or(33);

    let mut params = RuntimeParams::default();
    params.film_render.halation.active = false;
    params.film_render.grain.active = false;
    params.film_render.dir_couplers.diffusion_size_um = 0.0;
    params.film_render.glare.active = false;
    params.print_render.glare.active = false;
    params.scanner.unsharp_mask = [0.0, 0.0];

    let t = Instant::now();
    let pipeline = bake::build_pipeline(data_dir, film, None, params).expect("pipeline");
    eprintln!("pipeline build: {} ms", t.elapsed().as_millis());
    let t = Instant::now();
    let baked = bake::bake(&pipeline, lut_size).expect("bake");
    eprintln!(
        "bake ({}³ LUT, tc {}²): {} ms",
        lut_size,
        baked.tc_lut.size,
        t.elapsed().as_millis()
    );
    eprintln!(
        "film curve x: [{:.3}, {:.3}]",
        baked.film_curves.x_min, baked.film_curves.x_max
    );
    eprintln!(
        "density range: {:?} .. {:?}",
        baked.density_lut.density_min, baked.density_lut.density_max
    );
    eprintln!("dir couplers: {}", baked.dir_couplers.is_some());

    let rgb = image::open(input).expect("open").to_rgb32f();
    let (w, h) = (rgb.width(), rgb.height());
    let lin: Vec<Scalar> = rgb
        .into_raw()
        .into_iter()
        .map(|v| srgb_decode(from_f32(v)))
        .collect();
    let img = ImageBuf::from_data(w, h, lin.clone());

    let t = Instant::now();
    let reference = pipeline.process(img, select_backend().as_ref());
    eprintln!("full CPU pipeline: {} ms", t.elapsed().as_millis());

    let t = Instant::now();
    let ours: Vec<f32> = lin
        .chunks_exact(3)
        .flat_map(|p| baked.apply_pixelwise([to_f32(p[0]), to_f32(p[1]), to_f32(p[2])]))
        .collect();
    eprintln!(
        "baked pixelwise (single thread): {} ms",
        t.elapsed().as_millis()
    );

    let mut max = 0.0f32;
    let mut sum = 0.0f64;
    let mut over_1_255 = 0usize;
    for (r, o) in reference.data.iter().zip(&ours) {
        let d = (to_f32(*r) - o).abs();
        max = max.max(d);
        sum += d as f64;
        if d > 1.0 / 255.0 {
            over_1_255 += 1;
        }
    }
    let n = ours.len();
    println!(
        "error (display-encoded 0..1): max {:.5} ({:.2}/255), mean {:.6} ({:.3}/255), >1/255: {:.3}%",
        max,
        max * 255.0,
        sum / n as f64,
        sum / n as f64 * 255.0,
        100.0 * over_1_255 as f64 / n as f64
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
    save("bake_reference.png", &r);
    save("bake_ours.png", &ours);
}
