//! The propagation delay of each entry of a sampled response, from the slope of its phase: the
//! delays [`crate::Options::delays`] takes out of the samples before the fit.

use std::f64::consts::TAU;

use super::C;

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
    let mut order: Vec<usize> = (0..s.len()).collect();
    order.sort_by(|&a, &b| s[a].im.total_cmp(&s[b].im));
    (0..nf)
        .map(|e| {
            let mut prev: Option<f64> = None;
            let (mut sw, mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for &k in &order {
                let v = data[k][e];
                let w = v.norm();
                if w == 0.0 {
                    continue;
                }
                let mut phase = v.arg();
                if let Some(p) = prev {
                    phase += TAU * ((p - phase) / TAU).round();
                }
                prev = Some(phase);
                let x = s[k].im;
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
