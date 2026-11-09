# film-sim — provenance and licences

## Code (`src/`) — GPL-3.0

`film-sim` is derived from **spektrafilm-rs** by turbasvin
(https://github.com/turbasvin/spektrafilm-rs, GPL-3.0), vendored at commit
`9dd59b0380194b93686aaa230a8bb9680aa270a4` (2026-06-15). spektrafilm-rs is
itself a Rust port of **spektrafilm** by Andrea Volpato
(https://github.com/andreavolpato/spektrafilm, GPL-3.0).

Module mapping from the upstream crates:

| film-sim module | upstream crate      |
|-----------------|---------------------|
| `math`          | `spektrafilm-math`  |
| `model`         | `spektrafilm-model` |
| `engine`        | `spektrafilm-core`  |
| `backend`       | `spektrafilm-gpu` (trait + CPU backend only) |

Modifications made for RapidRAW:

- Merged the crates into one library and rewrote module paths.
- Removed the wgpu 24, CUDA and egui front-ends; RapidRAW renders the preview
  in its own wgpu pipeline.
- Removed the BLAS / OpenBLAS / Accelerate dependency from the CPU backend
  (pure-Rust matrix product on every platform).
- Removed the Python debug-comparison test module.
- Removed the `arctic2026alpha02` spectral upsampler (and its 8 MB LUT);
  RapidRAW uses `hanatos2025`.
- Added `Pipeline::render_from_film_density` (printing + scanning on a
  film-density image; the unchanged second half of `Pipeline::process`).
- Added `diffusion::diffusion_scatter_fraction`.
- Removed code left unused by the above (GPU-only test, unused parameter
  defaults, variables and imports) and pointed the tests at `data/`.
- Formatted with `cargo fmt`.

New code written for RapidRAW:

- `src/bake.rs` bakes a film look into the tables RapidRAW's shader uses
  (tc LUT, density curves, DIR couplers and a density → output 3-D LUT)
  and contains a CPU reference of that shader.
- `src/look.rs` holds the settings stored in RapidRAW's adjustments and the
  stock list.
- `src/scatter.rs` computes the auto-exposure and the low-resolution
  scattered-light field (diffusion filter + halation) with the upstream code.
- `src/placement.rs` ("match camera brightness") is ported from **negs** by
  Noah Haasis (https://github.com/noahhaasis/negs, GPL-3.0),
  `spektrafilm-core/src/exposure_placement.rs`, which credits yosoyelchris's
  Negs fork. Its trials run through `bake.rs`'s per-pixel chain instead of
  a full pipeline.
- `tests/rapidraw.rs` checks the above against the full CPU pipeline.

GPL-3.0 code may be combined with RapidRAW's AGPL-3.0 code (section 13 of
both licences). The full GPL-3.0 text is in `LICENSE`.

Some numerical routines are ported from numpy, scipy, scikit-image and
colour-science (PCHIP 3D LUT, MT19937 binomial sampler, CIE 1931 observer),
as credited in the upstream projects.

## Data (`data/`) — CC BY-SA 4.0

The film and paper profiles, filter data and spectral LUTs in `data/` are

    spektrafilm by Andrea Volpato
    https://github.com/andreavolpato/spektrafilm
    Licensed CC BY-SA 4.0

The licence and its preamble are in `data/SPEKTRAFILM_LICENSE.txt`. Changes
to these files must be recorded in `data/CHANGELOG.txt`.

The spectral-upsampling LUT (`hanatos_*`) was contributed to spektrafilm by
Johannes Hanatos (https://github.com/hanatos), author of vkdt.
