//! film-sim: spectral simulation of analogue colour film, print and scan.
//!
//! Derived from spektrafilm-rs (https://github.com/turbasvin/spektrafilm-rs,
//! GPL-3.0), a Rust port of spektrafilm by Andrea Volpato
//! (https://github.com/andreavolpato/spektrafilm, GPL-3.0). See NOTICE.md
//! for provenance and the licence of the bundled profile data.
//!
//! Module layout mirrors the upstream crates so fixes can be ported:
//! - `math`    ← spektrafilm-math   (spectral maths, interpolation, RNG)
//! - `model`   ← spektrafilm-model  (grain, halation, DIR couplers, glare)
//! - `engine`  ← spektrafilm-core   (profiles, params, pipeline stages)
//! - `backend` ← spektrafilm-gpu    (ComputeBackend trait + CPU backend only)

pub mod backend;
pub mod bake;
pub mod engine;
pub mod look;
pub mod math;
pub mod model;
pub mod placement;
pub mod scatter;
