use std::fs;

use examples::zklp::ZKLPExecutor;
use spain::simulate::{SpainConfig, stateful_simulate};

pub fn main() {
    let content = fs::read_to_string("loc2index64.txt")
        .unwrap()
        .split_whitespace()
        .map(|v| u64::from_str_radix(v, 16).unwrap())
        .collect::<Vec<_>>();
    // for i in 0..content.len() / 6 {
    for i in 0..1 {
        let &[lat, lng, res, result_i, result_j, result_k] =
            content[i * 6..(i + 1) * 6].try_into().unwrap();
        let lat = f64::from_bits(lat) as f64;
        let lng = f64::from_bits(lng) as f64;
        let alpha_lat = (lat / 2.).tan();
        let gamma_lat = (lat / 2.).sin();
        let delta_lat = (lat / 2.).cos();
        let beta_lat = 2. * gamma_lat * delta_lat;
        let alpha_lng = (lng / 2.).tan();
        let gamma_lng = (lng / 2.).sin();
        let delta_lng = (lng / 2.).cos();
        let beta_lng = 2. * gamma_lng * delta_lng;
        let exec = ZKLPExecutor::new(
            lat, lng, res, result_i, result_j, result_k, alpha_lat, beta_lat, gamma_lat, delta_lat,
            alpha_lng, beta_lng, gamma_lng, delta_lng,
        );
        let mut config = SpainConfig::default();
        config.batch_size = 256;
        dbg!(stateful_simulate(exec, Some(config)));
    }
}
