// SPDX-License-Identifier: AGPL-3.0-or-later

//! LTEE B1: Mutation accumulation validation (Barrick et al. 2009).
//!
//! Validates the mutation accumulation time series from the Long-Term
//! Evolution Experiment. Reproduces the Python baseline's linear and
//! power-law fits, component-wise mutation rates, and interpolation
//! using pure Rust math.
//!
//! Paper: Barrick et al. "Genome evolution and adaptation in a long-term
//! experiment with *Escherichia coli*" Nature 461:1243-1247 (2009).
//!
//! Expected values: `control/ltee_mutation_accumulation/expected_values.json`

#![expect(clippy::cast_precision_loss, reason = "LTEE data indexing")]

use neural_spring::rng::Rng;
use neural_spring::tolerances;
use neural_spring::validation::ValidationHarness;

const GENERATIONS: [f64; 6] = [0.0, 2000.0, 5000.0, 10000.0, 15000.0, 20000.0];
const POINT_MUTATIONS: [f64; 6] = [0.0, 8.0, 17.0, 29.0, 38.0, 45.0];
const IS_INSERTIONS: [f64; 6] = [0.0, 2.0, 5.0, 9.0, 14.0, 17.0];
const DELETIONS: [f64; 6] = [0.0, 1.0, 3.0, 6.0, 8.0, 10.0];

const EXPECTED_RATE: f64 = 3.592_307_692_307_692_4e-3;
const EXPECTED_POWER_EXPONENT: f64 = 0.821_279_838_189_791_9;
const EXPECTED_INTERCEPT: f64 = 4.200_000_000_000_008;
const EXPECTED_INTERP_7500: f64 = 34.5;
const EXPECTED_POINT_RATE: f64 = 2.220_879_120_879_121e-3;
const EXPECTED_IS_RATE: f64 = 8.626_373_626_373_626e-4;
const EXPECTED_DEL_RATE: f64 = 5.087_912_087_912_089e-4;

fn total_mutations() -> [f64; 6] {
    let mut total = [0.0; 6];
    for i in 0..6 {
        total[i] = POINT_MUTATIONS[i] + IS_INSERTIONS[i] + DELETIONS[i];
    }
    total
}

fn polyfit_1(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let sx: f64 = x.iter().sum();
    let sy: f64 = y.iter().sum();
    let sxx: f64 = x.iter().map(|&v| v * v).sum();
    let sxy: f64 = x.iter().zip(y).map(|(&xi, &yi)| xi * yi).sum();
    let denom = n.mul_add(sxx, -(sx * sx));
    let slope = n.mul_add(sxy, -(sx * sy)) / denom;
    let intercept = slope.mul_add(-sx, sy) / n;
    (slope, intercept)
}

fn interp_linear(xp: &[f64], fp: &[f64], x: f64) -> f64 {
    if x <= xp[0] {
        return fp[0];
    }
    if x >= xp[xp.len() - 1] {
        return fp[fp.len() - 1];
    }
    for i in 0..xp.len() - 1 {
        if x >= xp[i] && x <= xp[i + 1] {
            let t = (x - xp[i]) / (xp[i + 1] - xp[i]);
            return t.mul_add(fp[i + 1] - fp[i], fp[i]);
        }
    }
    fp[fp.len() - 1]
}

fn lstm_forward(gen_norm: &[f64], seed: u64) -> Vec<f64> {
    let hidden_size = 8;
    let mut rng = Rng::new(seed);

    let mut w_h = vec![0.0_f64; hidden_size * hidden_size];
    let mut w_x = vec![0.0_f64; hidden_size];
    let mut w_o = vec![0.0_f64; hidden_size];

    for v in &mut w_h {
        *v = rng.normal() * 0.1;
    }
    for v in &mut w_x {
        *v = rng.normal() * 0.1;
    }
    for v in &mut w_o {
        *v = rng.normal() * 0.1;
    }

    let mut h = vec![0.0_f64; hidden_size];
    let mut predictions = Vec::with_capacity(gen_norm.len());

    for &x_t in gen_norm {
        let mut new_h = vec![0.0; hidden_size];
        for i in 0..hidden_size {
            let mut sum = w_x[i] * x_t;
            for j in 0..hidden_size {
                sum += w_h[i * hidden_size + j] * h[j];
            }
            new_h[i] = sum.tanh();
        }
        h = new_h;

        let mut y_pred = 0.0;
        for i in 0..hidden_size {
            y_pred += w_o[i] * h[i];
        }
        predictions.push(y_pred);
    }

    predictions
}

struct SimpleLogger;
impl log::Log for SimpleLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Info
    }
    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            eprintln!("{}", record.args());
        }
    }
    fn flush(&self) {}
}
static LOGGER: SimpleLogger = SimpleLogger;

fn run_checks(h: &mut ValidationHarness) {
    let total = total_mutations();

    // ── Check 1: Data monotonicity ──────────────────────────────────
    let monotonic = total.windows(2).all(|w| w[0] <= w[1]);
    h.check_bool("B1-001: total mutations monotonically increasing", monotonic);

    // ── Check 2: Mutation rate estimation ───────────────────────────
    let (rate, intercept) = polyfit_1(&GENERATIONS, &total);

    h.check_bool("B1-002: mutation rate positive", rate > 0.0);
    h.check_abs(
        "B1-003: mutation rate matches Python",
        rate,
        EXPECTED_RATE,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_abs(
        "B1-004: intercept matches Python",
        intercept,
        EXPECTED_INTERCEPT,
        tolerances::CROSS_LANGUAGE,
    );

    // ── Check 3: Power-law fit ──────────────────────────────────────
    let log_gen: Vec<f64> = GENERATIONS[1..].iter().map(|&g| g.ln()).collect();
    let log_mut: Vec<f64> = total[1..].iter().map(|&m| m.ln()).collect();
    let (power_exp, _) = polyfit_1(&log_gen, &log_mut);

    h.check_abs(
        "B1-005: power-law exponent matches Python",
        power_exp,
        EXPECTED_POWER_EXPONENT,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_bool(
        "B1-006: sublinear accumulation (exponent < 1.0)",
        power_exp < 1.0,
    );

    // ── Check 4: LSTM forward pass ──────────────────────────────────
    let gen_max = GENERATIONS[5];
    let gen_norm: Vec<f64> = GENERATIONS.iter().map(|&g| g / gen_max).collect();
    let predictions = lstm_forward(&gen_norm, 42);

    h.check_bool(
        "B1-007: LSTM predictions all finite",
        predictions.iter().all(|p| p.is_finite()),
    );

    // ── Check 5: Neutral model fit ──────────────────────────────────
    let max_residual = GENERATIONS
        .iter()
        .zip(total.iter())
        .map(|(&g, &m)| (m - rate.mul_add(g, intercept)).abs())
        .fold(0.0_f64, f64::max);
    let relative_residual = max_residual / total[5];

    h.check_bool(
        "B1-008: neutral model relative residual < 15%",
        relative_residual < 0.15,
    );

    // ── Check 6: Component-wise rates ───────────────────────────────
    let (point_rate, _) = polyfit_1(&GENERATIONS, &POINT_MUTATIONS);
    let (is_rate, _) = polyfit_1(&GENERATIONS, &IS_INSERTIONS);
    let (del_rate, _) = polyfit_1(&GENERATIONS, &DELETIONS);

    h.check_bool(
        "B1-009: point mutations dominate",
        point_rate > is_rate && point_rate > del_rate,
    );
    h.check_abs(
        "B1-010: point rate matches Python",
        point_rate,
        EXPECTED_POINT_RATE,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_abs(
        "B1-011: IS rate matches Python",
        is_rate,
        EXPECTED_IS_RATE,
        tolerances::CROSS_LANGUAGE,
    );
    h.check_abs(
        "B1-012: deletion rate matches Python",
        del_rate,
        EXPECTED_DEL_RATE,
        tolerances::CROSS_LANGUAGE,
    );

    // ── Check 7: Interpolation at 7,500 generations ─────────────────
    let interp_7500 = interp_linear(&GENERATIONS, &total, 7500.0);
    h.check_abs(
        "B1-013: interpolation at 7,500 gen matches Python",
        interp_7500,
        EXPECTED_INTERP_7500,
        tolerances::CROSS_LANGUAGE,
    );

    // ── Check 8: Mutation rate in expected range ────────────────────
    h.check_bool(
        "B1-014: rate in biological range [1e-4, 1e-2]",
        (1e-4..1e-2).contains(&rate),
    );

}

fn main() {
    let _ = log::set_logger(&LOGGER);
    log::set_max_level(log::LevelFilter::Info);

    let mut h = ValidationHarness::new("LTEE B1: Mutation Accumulation (Barrick 2009)");
    run_checks(&mut h);
    h.finish();
}
