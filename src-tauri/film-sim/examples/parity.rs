//! Parity harness: run the film-sim CPU pipeline on a PNG exactly the way the
//! spektrafilm-rs CLI does (`spektrafilm process in.png -o out.png --raw-out`),
//! so the two raw f64 dumps can be compared byte for byte.
//!
//! cargo run --release --example parity -- <in.png> <out.png> <raw_out.bin> <film> <data_dir>

use std::io::Write;
use std::path::Path;

use film_sim::backend::select_backend;
use film_sim::engine::params::RuntimeParams;
use film_sim::engine::pipeline::Pipeline;
use film_sim::engine::profile;
use film_sim::math::image::ImageBuf;
use film_sim::math::precision::{Scalar, from_f32, srgb_decode};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 6 {
        eprintln!("usage: parity <in.png> <out.png> <raw_out.bin> <film> <data_dir>");
        std::process::exit(2);
    }
    let (input, output, raw_out, film_name, data_dir) =
        (&args[1], &args[2], &args[3], &args[4], Path::new(&args[5]));

    let film = profile::load_profile_by_name(data_dir, film_name).expect("film profile");
    let print_stock = film
        .info
        .target_print
        .clone()
        .expect("film has target_print");
    let print = profile::load_profile_by_name(data_dir, &print_stock).expect("print profile");

    let mut params = RuntimeParams::default();
    params.io.input_color_space = "sRGB".to_string();
    params.io.input_cctf_decoding = false;

    let rgb = image::open(input).expect("open input").to_rgb32f();
    let (w, h) = (rgb.width(), rgb.height());
    let data: Vec<Scalar> = rgb
        .into_raw()
        .into_iter()
        .map(|v| srgb_decode(from_f32(v)))
        .collect();
    let img = ImageBuf::from_data(w, h, data);

    let backend = select_backend();
    let pipeline =
        Pipeline::new_with_spectral(film, print, params, data_dir).expect("spectral pipeline");
    let t = std::time::Instant::now();
    let result = pipeline.process(img, backend.as_ref());
    eprintln!("film-sim pipeline: {} ms", t.elapsed().as_millis());

    let bytes: Vec<u8> = result
        .data
        .iter()
        .flat_map(|&v| (v as f64).to_le_bytes())
        .collect();
    std::fs::File::create(raw_out)
        .and_then(|mut f| f.write_all(&bytes))
        .expect("write raw dump");

    let u8s: Vec<u8> = result
        .data
        .iter()
        .map(|&v| ((v.clamp(0.0, 1.0) * 255.0).round_ties_even()) as u8)
        .collect();
    image::RgbImage::from_raw(result.width, result.height, u8s)
        .expect("buffer")
        .save(output)
        .expect("save png");
}
