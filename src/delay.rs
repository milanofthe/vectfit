//! The phase of each entry of a sampled response, unwrapped over frequency: how many turns it
//! makes over the band (the order a fit needs at least, [`phase_turns`]) and the slope of its
//! straight line, the propagation delay [`crate::Options::delays`] takes out of the samples before
//! the fit ([`phase_slope`]).

use std::f64::consts::TAU;

use super::C;

/// The samples of `s` (on the `jω` axis) in order of frequency.
fn by_frequency(s: &[C]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..s.len()).collect();
    order.sort_by(|&a, &b| s[a].im.total_cmp(&s[b].im));
    order
}

/// The entry `e` of `data` at the samples `order` as `(ω, phase, magnitude)`, the phase unwrapped
/// from sample to sample (taking it to turn by less than half a turn between two neighbouring
/// samples); the samples where it is zero are left out.
fn unwrapped(s: &[C], data: &[Vec<C>], order: &[usize], e: usize) -> Vec<(f64, f64, f64)> {
    let mut prev: Option<f64> = None;
    order
        .iter()
        .filter_map(|&k| {
            let v = data[k][e];
            let w = v.norm();
            if w == 0.0 {
                return None;
            }
            let mut phase = v.arg();
            if let Some(p) = prev {
                phase += TAU * ((p - phase) / TAU).round();
            }
            prev = Some(phase);
            Some((s[k].im, phase, w))
        })
        .collect()
}

/// The largest number of turns the phase of an entry of `data` spans over the samples `s` (on the
/// `jω` axis), unwrapped as [`phase_slope`] does. A rational function needs at least one c.c. pole
/// pair a turn: the phase of a pair and its zeros spans at most one turn (the argument principle),
/// so a response that turns `n` times over the band (a propagation delay, a chain of resonances)
/// is not fitted by fewer than `n` pairs. On samples too sparse for the phase the count is too
/// small, as [`phase_slope`]'s delay is.
pub fn phase_turns(s: &[C], data: &[Vec<C>]) -> f64 {
    let nf = data.first().map_or(0, |row| row.len());
    let order = by_frequency(s);
    (0..nf)
        .map(|e| {
            let u = unwrapped(s, data, &order, e);
            let (lo, hi) = u.iter().fold(
                (f64::INFINITY, f64::NEG_INFINITY),
                |(lo, hi), &(_, p, _)| (lo.min(p), hi.max(p)),
            );
            if u.len() < 2 {
                0.0
            } else {
                (hi - lo) / TAU
            }
        })
        .fold(0.0, f64::max)
}

/// Per entry of `data` (`data[k][e]` the entry `e` at the sample `s[k]` on the `jω` axis), the
/// delay `τ_e >= 0` of the straight line through its unwrapped phase over `ω` (least squares,
/// each sample weighted by its magnitude): `exp(s·τ_e)·data_e` has no linear phase trend left.
///
/// The phase is unwrapped from sample to sample in order of frequency, which takes the phase
/// between two neighbouring samples to turn by less than half a turn: on samples too sparse for
/// the delay the estimate is too small, which costs poles, not accuracy. An entry that is zero
/// throughout, or sampled at one frequency, has no delay; a rising phase (a negative slope of the
/// delay) none either.
pub fn phase_slope(s: &[C], data: &[Vec<C>]) -> Vec<f64> {
    let nf = data.first().map_or(0, |row| row.len());
    let order = by_frequency(s);
    (0..nf)
        .map(|e| {
            let (mut sw, mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for (x, phase, w) in unwrapped(s, data, &order, e) {
                sw += w;
                sx += w * x;
                sy += w * phase;
                sxx += w * x * x;
                sxy += w * x * phase;
            }
            let var = sw * sxx - sx * sx;
            if var <= f64::EPSILON * sxx * sw {
                return 0.0;
            }
            (-(sw * sxy - sx * sy) / var).max(0.0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A delayed rational entry: the delay is its phase slope where the rational part is flat in
    /// phase (a constant), and a fit with it needs no pole for the propagation.
    #[test]
    fn the_phase_slope_of_a_delay_is_the_delay() {
        let tau = 7.0;
        let s: Vec<C> = (0..40)
            .map(|i| C::new(0.0, 0.01 + i as f64 / 39.0))
            .collect();
        let data: Vec<Vec<C>> = s
            .iter()
            .map(|&z| vec![(-z * tau).exp() * 0.8, C::new(0.0, 0.0)])
            .collect();
        let d = phase_slope(&s, &data);
        assert!((d[0] - tau).abs() < 1e-9, "delay {}", d[0]);
        assert_eq!(d[1], 0.0);
        let first: Vec<Vec<C>> = data.iter().map(|r| vec![r[0]]).collect();
        let m = crate::fit_auto_opts(
            &s,
            &first,
            1e-9,
            20,
            &crate::Options {
                proportional: false,
                relative: false,
                delays: vec![d[0]],
                ..crate::Options::default()
            },
        );
        assert!(m.order() <= 3, "{} poles", m.order());
        for (z, row) in s.iter().zip(&data) {
            assert!((m.eval(*z)[0] - row[0]).norm() < 1e-8);
        }
    }
}
