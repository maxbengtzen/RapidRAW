//! User-facing film look settings, as stored in RapidRAW's adjustments
//! (`adjustments.filmSim`), and their mapping onto spektrafilm's
//! `RuntimeParams`. (film-sim addition, not part of spektrafilm-rs.)

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::engine::params::RuntimeParams;
use crate::engine::profile;

pub const DEFAULT_STOCK: &str = "kodak_portra_400";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct FilmLookSettings {
    /// Film profile name (file stem in `data/profiles`).
    pub stock: String,
    /// Print paper profile; `None` = the film's own target print.
    pub paper: Option<String>,
    /// Scan the developed film directly instead of printing it.
    pub scan_film: bool,
    /// Camera exposure onto the film, in EV.
    pub film_exposure: f32,
    /// Enlarger exposure, in EV (0 = the auto-normalised print).
    pub print_exposure: f32,
    /// Enlarger filtration shifts, in CC units (colour balance of the print).
    pub yellow_shift: f32,
    pub magenta_shift: f32,
    /// DIR-coupler strength (0 = off, 1 = stock, up to 2).
    pub couplers: f32,
    /// Grain strength (0 = off, 1 = the stock's particle model, up to 2).
    /// Applied at render time; does not change the baked tables.
    pub grain: f32,
    /// Halation strength (0 = off, 1 = stock, up to 3).
    /// Applied at render time; does not change the baked tables.
    pub halation: f32,
    /// Meter the scene and place it on the film like spektrafilm's
    /// auto-exposure; `film_exposure` and RapidRAW's exposure then act as
    /// exposure compensation.
    pub auto_exposure: bool,
    /// spektrafilm metering pattern, see `scatter::auto_exposure_ev`.
    pub auto_exposure_method: String,
    /// Place the exposure so the film render matches the brightness of the
    /// camera's own JPEG (`placement`); falls back to auto-exposure when the
    /// camera preview cannot be used.
    pub match_camera_brightness: bool,
    /// Film format, long edge in mm (35 = full frame). Sets the physical
    /// pixel size, so grain, halation and diffusion scale with it.
    pub film_format: f32,
    /// Camera diffusion filter family ("none", "black_pro_mist", "pro_mist",
    /// "glimmerglass", "cinebloom").
    pub diffusion_filter: String,
    /// Diffusion filter strength in filter stops (1/8 … 2).
    pub diffusion_strength: f32,
}

impl Default for FilmLookSettings {
    fn default() -> Self {
        Self {
            stock: DEFAULT_STOCK.into(),
            paper: None,
            scan_film: false,
            film_exposure: 0.0,
            print_exposure: 0.0,
            yellow_shift: 0.0,
            magenta_shift: 0.0,
            couplers: 1.0,
            grain: 1.0,
            halation: 1.0,
            auto_exposure: true,
            auto_exposure_method: "center_weighted".into(),
            match_camera_brightness: false,
            film_format: 35.0,
            diffusion_filter: "black_pro_mist".into(),
            diffusion_strength: 0.5,
        }
    }
}

impl FilmLookSettings {
    /// spektrafilm runtime parameters for these settings.
    pub fn runtime_params(&self) -> RuntimeParams {
        let mut p = RuntimeParams::default();
        p.io.scan_film = self.scan_film;
        // Film exposure is applied at render time (the front pass is
        // homogeneous), so it is not baked.
        p.camera.exposure_compensation_ev = 0.0;
        p.camera.film_format_mm = self.film_format.clamp(4.0, 200.0);
        p.enlarger.print_exposure = 2f32.powf(self.print_exposure);
        p.enlarger.y_filter_shift = self.yellow_shift;
        p.enlarger.m_filter_shift = self.magenta_shift;
        let dir = &mut p.film_render.dir_couplers;
        dir.active = self.couplers > 0.0;
        dir.amount = self.couplers.clamp(0.0, 2.0) as f64;
        // Grain and halation are always baked at stock strength; the user
        // amounts scale them at render time (see `grain` / `halation`).
        p.film_render.grain.active = true;
        p.film_render.halation.active = true;
        p
    }

    /// Cache key of the baked tables: only the settings that change them.
    pub fn bake_key(&self) -> String {
        let s = Self {
            grain: 0.0,
            halation: 0.0,
            film_exposure: 0.0,
            auto_exposure: false,
            auto_exposure_method: String::new(),
            match_camera_brightness: false,
            diffusion_filter: String::new(),
            diffusion_strength: 0.0,
            ..self.clone()
        };
        serde_json::to_string(&s).unwrap_or_default()
    }

    /// The camera diffusion filter, if one is selected.
    pub fn diffusion_params(&self) -> Option<crate::engine::params::DiffusionFilterParams> {
        if self.diffusion_filter.is_empty()
            || self.diffusion_filter == "none"
            || self.diffusion_strength <= 0.0
        {
            return None;
        }
        Some(crate::engine::params::DiffusionFilterParams {
            active: true,
            filter_family: self.diffusion_filter.clone(),
            strength: self.diffusion_strength.clamp(0.0, 2.0),
            ..Default::default()
        })
    }
}

/// One entry of the stock list shown in the UI.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StockInfo {
    pub id: String,
    pub name: String,
    /// "negative", "positive" (slide) or "print" (paper / print film).
    pub kind: String,
    pub is_bw: bool,
    pub target_print: Option<String>,
}

/// List the bundled film and paper profiles.
pub fn list_stocks(data_dir: &Path) -> Vec<StockInfo> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(data_dir.join("profiles")) else {
        return out;
    };
    let mut names: Vec<String> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let p = e.path();
            (p.extension()? == "json").then(|| p.file_stem()?.to_str().map(str::to_string))?
        })
        .collect();
    names.sort();
    for id in names {
        let Ok(prof) = profile::load_profile_by_name(data_dir, &id) else {
            continue;
        };
        // Print media (papers and cine print films) are not camera stocks.
        let kind = if !prof.is_filming() {
            "print"
        } else if prof.is_positive() {
            "positive"
        } else {
            "negative"
        };
        out.push(StockInfo {
            name: prof.info.name.clone().unwrap_or_else(|| id.clone()),
            id,
            kind: kind.into(),
            is_bw: prof.is_bw(),
            target_print: prof.info.target_print.clone(),
        });
    }
    out
}
