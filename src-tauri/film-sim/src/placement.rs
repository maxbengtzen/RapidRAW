//! "Match camera brightness": place the scene exposure so that the film
//! render's mid-tones match the camera's own JPEG rendering.
//!
//! Ported from negs (https://github.com/noahhaasis/negs, GPL-3.0),
//! `spektrafilm-core/src/exposure_placement.rs`, which credits yosoyelchris's
//! Negs fork. Changes: trials run through the baked per-pixel film chain
//! (`BakedFilm::apply_pixelwise`) instead of a full pipeline, the solve is
//! synchronous, and the EV range is centred on a caller-supplied exposure.
//!
//! The camera preview supplies luminance intent only: the distributions of the
//! central 80 % of both images are compared at the 10–90 % quantiles that fall
//! in the mid-tones, and the exposure is backed off if it would push more of
//! the frame to the paper white than the camera preview has there.

use rayon::prelude::*;

use crate::bake::BakedFilm;
use crate::scatter::SceneProxy;

/// Long edge of the images compared.
pub const EDGE: u32 = 256;
const MAX_EV: f64 = 3.0;
const MIN_SAMPLES: usize = 128;
const QUANTILES: [f64; 9] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
const EXTRA_WHITE_AREA: f64 = 0.01;

/// Result of [`place_exposure`].
#[derive(Debug, Clone, Copy)]
pub struct Placement {
    /// Scene exposure (EV, absolute) that matches the camera preview.
    pub exposure_ev: f64,
    /// The mid-tone match before the highlight limit.
    pub requested_ev: f64,
    pub highlight_limited: bool,
    /// Median log2 mid-tone ratio render / camera at `exposure_ev`.
    pub residual_ev: f64,
}

#[derive(Clone, Copy, Debug)]
struct WhiteThresholds {
    waveform: f64,
    channels: [f64; 3],
}

#[derive(Clone, Debug)]
struct Distribution {
    quantiles: [f64; 9],
    near_white_fraction: f64,
}

fn inverse_srgb(x: f64) -> f64 {
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

fn y(rgb: [f64; 3]) -> f64 {
    rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722
}

fn median(values: &mut [f64]) -> f64 {
    let middle = values.len() / 2;
    *values.select_nth_unstable_by(middle, f64::total_cmp).1
}

/// Distribution of the central 80 % of a display-referred (sRGB-encoded)
/// image.
fn distribution(
    pixels: &[[f32; 3]],
    width: u32,
    height: u32,
    thresholds: WhiteThresholds,
) -> Result<Distribution, String> {
    if width == 0 || height == 0 || (width as usize * height as usize) != pixels.len() {
        return Err("invalid image dimensions".into());
    }
    let mut luminances = Vec::new();
    let mut near_white = 0;
    let (mx, my) = (width / 10, height / 10);
    for row in my..height - my {
        for col in mx..width - mx {
            let p = pixels[(row * width + col) as usize];
            if p.iter().any(|v| !v.is_finite()) {
                continue;
            }
            let rgb = p.map(f64::from);
            let luma = y(rgb.map(inverse_srgb));
            if !luma.is_finite() || luma < 0.0 {
                continue;
            }
            luminances.push(luma);
            if y(rgb) >= thresholds.waveform || (0..3).any(|c| rgb[c] >= thresholds.channels[c]) {
                near_white += 1;
            }
        }
    }
    if luminances.len() < MIN_SAMPLES {
        return Err("too few usable pixels".into());
    }
    luminances.sort_unstable_by(f64::total_cmp);
    Ok(Distribution {
        quantiles: QUANTILES
            .map(|q| luminances[((luminances.len() - 1) as f64 * q).round() as usize]),
        near_white_fraction: near_white as f64 / luminances.len() as f64,
    })
}

fn calibration_thresholds(white: [f64; 3]) -> Result<WhiteThresholds, String> {
    if white.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err("the film has no usable print white".into());
    }
    // Relative to the paper white, so a paper white below 1 cannot evade
    // the highlight guard by never clipping digitally.
    Ok(WhiteThresholds {
        waveform: (0.97 * y(white)).min(0.97),
        channels: white.map(|v| (0.985 * v).min(0.985)),
    })
}

/// Solve the exposure offset (EV, within ±[`MAX_EV`]) at which `render`'s
/// mid-tones match `target`.
fn solve(
    target: &Distribution,
    mut render: impl FnMut(f64) -> Result<Distribution, String>,
) -> Result<Placement, String> {
    // Only the camera's mid-tone quantiles are matched: night scenes are not
    // driven to a mid-grey histogram.
    let selected: Vec<_> = (0..QUANTILES.len())
        .filter(|&i| {
            let v = target.quantiles[i];
            v >= inverse_srgb(0.15) && v <= inverse_srgb(0.85)
        })
        .collect();
    if selected.len() < 3 {
        return Err("the camera preview has too few mid-tones".into());
    }
    let residual = |d: &Distribution| -> f64 {
        let mut v: Vec<_> = selected
            .iter()
            .map(|&i| (d.quantiles[i].max(1e-9) / target.quantiles[i]).log2())
            .collect();
        median(&mut v)
    };
    let baseline = render(0.0)?;
    let budget =
        (baseline.near_white_fraction.max(target.near_white_fraction) + EXTRA_WHITE_AREA).min(1.0);
    let points: Vec<(f64, f64)> = (0..=12)
        .map(|i| {
            let ev = -MAX_EV + i as f64 * 0.5;
            render(ev).map(|d| (ev, residual(&d)))
        })
        .collect::<Result<_, _>>()?;
    if points.windows(2).any(|p| p[1].1 + 0.025 < p[0].1) {
        return Err("the film response is not monotone here".into());
    }
    if points[0].1 > 0.0 || points[12].1 < 0.0 {
        return Err("the camera brightness is outside ±3 EV".into());
    }
    let (mut lo, mut hi) = points
        .windows(2)
        .find(|p| p[0].1 <= 0.0 && p[1].1 >= 0.0)
        .map_or((-MAX_EV, MAX_EV), |p| (p[0].0, p[1].0));
    for _ in 0..8 {
        let mid = (lo + hi) * 0.5;
        if residual(&render(mid)?) < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let requested = (((lo + hi) * 0.5) * 100.0).round() / 100.0;
    let mut ev = requested;
    let mut after = render(ev)?;
    let limited = ev > 0.0 && after.near_white_fraction > budget;
    if limited {
        let (mut lo, mut hi) = (0.0, ev);
        for _ in 0..10 {
            let mid = (lo + hi) * 0.5;
            if render(mid)?.near_white_fraction <= budget {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        ev = (lo * 100.0).floor() / 100.0;
        after = render(ev)?;
    }
    Ok(Placement {
        exposure_ev: ev,
        requested_ev: requested,
        highlight_limited: limited,
        residual_ev: residual(&after),
    })
}

/// Area-average `rgb` (`width × height`) down to at most `long_edge`.
pub fn downsample(
    width: u32,
    height: u32,
    rgb: &[[f32; 3]],
    long_edge: u32,
) -> (u32, u32, Vec<[f32; 3]>) {
    let scale = (long_edge as f64 / width.max(height) as f64).min(1.0);
    let w = ((width as f64 * scale).round() as u32).max(1);
    let h = ((height as f64 * scale).round() as u32).max(1);
    let data = (0..h)
        .into_par_iter()
        .flat_map_iter(|oy| {
            let y0 = (oy as u64 * height as u64 / h as u64) as u32;
            let y1 = (((oy + 1) as u64 * height as u64).div_ceil(h as u64) as u32).max(y0 + 1);
            (0..w).map(move |ox| {
                let x0 = (ox as u64 * width as u64 / w as u64) as u32;
                let x1 = (((ox + 1) as u64 * width as u64).div_ceil(w as u64) as u32).max(x0 + 1);
                let mut sum = [0.0f64; 3];
                for y in y0..y1.min(height) {
                    for x in x0..x1.min(width) {
                        let p = rgb[(y * width + x) as usize];
                        for c in 0..3 {
                            sum[c] += p[c] as f64;
                        }
                    }
                }
                let n = ((y1.min(height) - y0) * (x1.min(width) - x0)) as f64;
                sum.map(|s| (s / n) as f32)
            })
        })
        .collect();
    (w, h, data)
}

/// The scene exposure (absolute EV applied to `source`) at which `film`
/// renders `source` with the brightness of the camera's `reference`.
///
/// * `source`: the scene in linear RapidRAW pipe space, as framed by the
///   camera (no crop), at most [`EDGE`] pixels on the long edge.
/// * `reference`: the camera's rendering of the same frame, sRGB-encoded
///   0–1, same orientation; it is reduced to the size of `source`.
/// * `center_ev`: the search is `center_ev ± 3 EV` (use auto-exposure).
pub fn place_exposure(
    film: &BakedFilm,
    source: &SceneProxy,
    reference: &SceneProxy,
    center_ev: f64,
) -> Result<Placement, String> {
    let aspect = |p: &SceneProxy| p.width as f64 / p.height.max(1) as f64;
    if (aspect(source) / aspect(reference) - 1.0).abs() > 0.03 {
        return Err("the camera preview is framed differently".into());
    }
    if source.rgb.iter().flatten().any(|v| !v.is_finite()) {
        return Err("the image has non-finite pixels".into());
    }
    let (rw, rh, reference_rgb) = downsample(
        reference.width,
        reference.height,
        reference.rgb,
        source.width.max(source.height),
    );

    let white = film
        .apply_pixelwise([65536.0; 3])
        .map(|v| v.clamp(0.0, 1.0) as f64);
    let thresholds = calibration_thresholds(white)?;
    let target = distribution(&reference_rgb, rw, rh, thresholds)?;

    let mut cache: Vec<(u64, Distribution)> = Vec::new();
    let render = |ev: f64| -> Result<Distribution, String> {
        let key = (ev as f32).to_bits() as u64;
        if let Some((_, d)) = cache.iter().find(|(k, _)| *k == key) {
            return Ok(d.clone());
        }
        let gain = (2f64.powf(center_ev + ev as f32 as f64)) as f32;
        let out: Vec<[f32; 3]> = source
            .rgb
            .par_iter()
            .map(|p| {
                film.apply_pixelwise(p.map(|v| v.max(0.0) * gain))
                    .map(|v| v.clamp(0.0, 1.0))
            })
            .collect();
        let d = distribution(&out, source.width, source.height, thresholds)?;
        cache.push((key, d.clone()));
        Ok(d)
    };
    let mut p = solve(&target, render)?;
    p.exposure_ev += center_ev;
    p.requested_ev += center_ev;
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn srgb(x: f64) -> f64 {
        if x <= 0.0031308 {
            12.92 * x
        } else {
            1.055 * x.powf(1.0 / 2.4) - 0.055
        }
    }
    fn threshold() -> WhiteThresholds {
        calibration_thresholds([1.0; 3]).unwrap()
    }
    fn simple_frame(ev: f64) -> Vec<[f32; 3]> {
        (0..1024)
            .map(|i| {
                let x = (0.018 + 0.30 * (i % 32) as f64 / 31.0) * ev.exp2();
                [srgb(x.min(1.0)) as f32; 3]
            })
            .collect()
    }

    #[test]
    fn scalar_fit_recovers_camera_brightness() {
        for expected in [-1.2, 0.0, 0.73, 1.1] {
            let target = distribution(&simple_frame(expected), 32, 32, threshold()).unwrap();
            let a = solve(&target, |ev| {
                distribution(&simple_frame(ev), 32, 32, threshold())
            })
            .unwrap();
            assert!(
                (a.exposure_ev - expected).abs() <= 0.01,
                "{expected}: {a:?}"
            );
            assert!(!a.highlight_limited);
            assert!(a.residual_ev.abs() < 0.02);
        }
    }

    #[test]
    fn paper_white_guard_operates_below_digital_clipping() {
        let t = calibration_thresholds([0.90, 0.94, 0.86]).unwrap();
        let dim = vec![[0.75f32; 3]; 1024];
        let white = vec![[0.89f32, 0.93, 0.85]; 1024];
        assert_eq!(
            distribution(&dim, 32, 32, t).unwrap().near_white_fraction,
            0.0
        );
        assert_eq!(
            distribution(&white, 32, 32, t).unwrap().near_white_fraction,
            1.0
        );
    }

    #[test]
    fn highlight_backoff_respects_white_budget() {
        let target = Distribution {
            quantiles: [0.2; 9],
            near_white_fraction: 0.0,
        };
        let a = solve(&target, |ev| {
            Ok(Distribution {
                quantiles: [0.1 * ev.exp2(); 9],
                near_white_fraction: (ev - 0.4).max(0.0) * 0.2,
            })
        })
        .unwrap();
        assert!((a.requested_ev - 1.0).abs() < 0.01);
        assert!(a.highlight_limited);
        assert!((a.exposure_ev - 0.45).abs() < 0.011);
        assert!(a.residual_ev < -0.5);
    }

    #[test]
    fn missing_midtones_unbracketed_and_nonmonotonic_are_errors() {
        let dark = Distribution {
            quantiles: [0.0001; 9],
            near_white_fraction: 0.0,
        };
        assert!(solve(&dark, |_| Ok(dark.clone())).is_err());
        let target = Distribution {
            quantiles: [0.2; 9],
            ..dark
        };
        assert!(
            solve(&target, |_| Ok(Distribution {
                quantiles: [0.001; 9],
                ..target.clone()
            }))
            .is_err()
        );
        assert!(
            solve(&target, |ev| Ok(Distribution {
                quantiles: [0.2 * (-ev).exp2(); 9],
                ..target.clone()
            }))
            .is_err()
        );
        assert!(distribution(&[[f32::NAN; 3]; 1024], 32, 32, threshold()).is_err());
    }

    #[test]
    fn downsample_preserves_mean() {
        let rgb: Vec<[f32; 3]> = (0..300 * 200).map(|i| [(i % 7) as f32; 3]).collect();
        let (w, h, small) = downsample(300, 200, &rgb, 64);
        assert_eq!((w, h), (64, 43));
        let mean = |v: &[[f32; 3]]| v.iter().map(|p| p[0] as f64).sum::<f64>() / v.len() as f64;
        assert!((mean(&rgb) - mean(&small)).abs() < 0.05);
    }
}
