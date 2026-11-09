//! cargo run --example list_stocks -- <data_dir>
fn main() {
    let dir = std::env::args().nth(1).expect("data dir");
    for s in film_sim::look::list_stocks(std::path::Path::new(&dir)) {
        println!(
            "{:<32} {:<9} bw={:<5} print={:<24} {}",
            s.id,
            s.kind,
            s.is_bw,
            s.target_print.unwrap_or_default(),
            s.name
        );
    }
}
