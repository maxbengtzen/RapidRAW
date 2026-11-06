//! Check the closed-form grain noise used by RapidRAW's shader,
//!   sd = dmax * sqrt(p * sat / (npp * n_sub)),  p = (d + dmin) / dmax,
//!   sat = 1 - p * uniformity,
//! against spektrafilm's sampled particle model on flat density patches.
use film_sim::backend::cpu_backend::CpuBackend;
use film_sim::math::image::ImageBuf;
use film_sim::math::precision::{from_f64, to_f32};
use film_sim::model::grain::apply_grain_to_density;

fn main() {
    let (w, h) = (256u32, 256u32);
    let dmin = [0.03f64; 3];
    let dmax_curves = [2.2f64, 2.0, 1.8];
    let uniformity = [0.97f64, 0.99, 0.97];
    let area = 0.2f64;
    let scale = [1.6f64, 1.6, 3.2];
    for pix_um in [35.0f32, 12.0, 6.0] {
        for d in [0.1f64, 0.6, 1.2, 1.7] {
            let data = vec![from_f64(d); (w * h * 3) as usize];
            let img = ImageBuf::from_data(w, h, data);
            let out = apply_grain_to_density(
                &img,
                pix_um,
                area,
                scale,
                dmin,
                dmax_curves,
                uniformity,
                0.0,
                1,
                false,
                &CpuBackend,
            );
            let mut line = format!("pix {pix_um:>4} um  d {d:.1}:");
            for c in 0..3 {
                let v: Vec<f64> = out.pixels().map(|p| to_f32(p[c]) as f64).collect();
                let mean = v.iter().sum::<f64>() / v.len() as f64;
                let sd =
                    (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
                let dmax = dmax_curves[c] + dmin[c];
                let p = ((d + dmin[c]) / dmax).clamp(1e-6, 1.0 - 1e-6);
                let sat = 1.0 - p * uniformity[c] * (1.0 - 1e-6);
                let npp = (pix_um as f64).powi(2) / (area * scale[c]);
                let pred = dmax * (p * sat / npp).sqrt();
                line += &format!(
                    "  c{c} mean {mean:.4} sd {sd:.5} pred {pred:.5} ({:+.1}%)",
                    100.0 * (sd / pred - 1.0)
                );
            }
            println!("{line}");
        }
    }
}
