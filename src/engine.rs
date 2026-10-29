//! The relaxed vector-fitting engine: the `Vf` least-squares workspace (σ identification → pole
//! relocation → residue fit), the fixed-order [`fit`], and the automatic-order [`fit_auto`] with
//! Grivet-Talocia/Bandinu-style *adding and skimming* — grow the pole set at the worst-error band,
//! drop spurious poles by their band-limited energy, until the response is met with a minimal set.
//!
//! [`fit_auto_opts`] takes the basis, the weighting, per-entry delays, the poles to start from and
//! the parameters of the order search as [`Options`]. With absolute weights the residue basis is
//! the same for every entry (one factorisation for all), and a response of more entries than real
//! equations a sample identifies its poles on the real combinations of its entries that span it
//! (the many-port compression: the least-squares residual of any pole set is the same).

use faer::linalg::solvers::SolveLstsq;
use faer::Mat;
use rayon::prelude::*;

use super::model::VfModel;
use super::C;

/// Partial-fraction basis at `s = jω`, written into caller-owned slices (no allocation):
/// `xs` = the σ part [real, cpx-re, cpx-im]; `xf` = [const?, diff?] ++ `xs`. A free function so
/// the caller can fill its scratch buffers while holding the (disjoint) pole fields by ref.
#[allow(clippy::too_many_arguments)]
fn fill_x_into(
    s: C,
    pr: &[f64],
    pc: &[C],
    fit_const: bool,
    fit_diff: bool,
    fit_sqrt: bool,
    xf: &mut [C],
    xs: &mut [C],
) {
    let nr = pr.len();
    for (i, &p) in pr.iter().enumerate() {
        xs[i] = C::new(1.0, 0.0) / (s - C::new(p, 0.0));
    }
    for (i, &p) in pc.iter().enumerate() {
        xs[nr + i] = C::new(1.0, 0.0) / (s - p) + C::new(1.0, 0.0) / (s - p.conj());
        xs[nr + pc.len() + i] = C::new(0.0, 1.0) / (s - p) - C::new(0.0, 1.0) / (s - p.conj());
    }
    let mut o = 0;
    if fit_const {
        xf[o] = C::new(1.0, 0.0);
        o += 1;
    }
    if fit_diff {
        xf[o] = s;
        o += 1;
    }
    if fit_sqrt {
        // Skin-effect branch term √s (principal branch): on the jω axis
        // √(jω) = (1+j)·√(ω/2) — one REAL coefficient captures the
        // Kramers-Kronig-consistent resistance+inductance pair that a rational
        // basis can only chase with spurious poles.
        xf[o] = s.sqrt();
        o += 1;
    }
    xf[o..].copy_from_slice(xs);
}

/// The least-squares weight of every equation, `wt[sample][entry]`: the one definition of the
/// error a fit is judged on (the fit itself, [`crate::ratmodel::RationalModel::sample_error`]).
///
/// Relative (`relative`): the inverse magnitude of the sample. A matrix response spans a huge
/// magnitude range across entries (a spiral's off-diagonal Y is ~1000× its diagonal); plain
/// absolute least-squares fits the large entries and leaves the small ones at ~10% relative
/// error — invisible to an absolute residual, so the order growth stops too early. Weighting
/// every equation by 1/|data| makes each entry fit to its OWN relative accuracy and the
/// convergence metric track relative error. The magnitude is floored per entry at a share of
/// its MEDIAN magnitude, not the band peak: a sharp resonance peak would otherwise raise the
/// floor so far that every off-peak sample of the entry is under-weighted — the fit then reports
/// convergence while the off-peak band (where L/Q curves live) is wrong. The small peak term
/// keeps true transmission zeros from being weighted as noise.
///
/// Absolute: every equation alike, the error an absolute one (a response with an absolute
/// tolerance, as S, whose -80 dB couplings would otherwise be fitted to the relative accuracy
/// of the others, noise and all).
/// The magnitude floor of a relative weight, as a share of the entry's median magnitude
/// ([`weights`]). Source: not measured; set with the relative weighting, low enough that the
/// off-resonance band of an entry keeps its own relative accuracy.
const REL_FLOOR_MEDIAN: f64 = 0.05;

/// The magnitude floor of a relative weight, as a share of the entry's peak magnitude
/// ([`weights`]): a true transmission zero is weighted as at most this far below the peak, not
/// as noise to be fitted exactly. Source: not measured; set with the relative weighting.
const REL_FLOOR_PEAK: f64 = 1e-6;

pub(crate) fn weights(data: &[Vec<C>], relative: bool) -> Vec<Vec<f64>> {
    let nf = data[0].len();
    if !relative {
        return vec![vec![1.0; nf]; data.len()];
    }
    let floor: Vec<f64> = (0..nf)
        .map(|e| {
            let mut mags: Vec<f64> = data.iter().map(|row| row[e].norm()).collect();
            mags.sort_by(f64::total_cmp);
            let (med, peak) = (mags[mags.len() / 2], mags[mags.len() - 1]);
            (REL_FLOOR_MEDIAN * med)
                .max(REL_FLOOR_PEAK * peak)
                .max(f64::MIN_POSITIVE)
        })
        .collect();
    data.iter()
        .map(|row| (0..nf).map(|e| 1.0 / row[e].norm().max(floor[e])).collect())
        .collect()
}

/// What a fit holds besides its poles, how it weighs the samples, where it starts and how it
/// searches the order ([`fit_auto_opts`]).
#[derive(Clone, Debug)]
pub struct Options {
    /// The proportional term `s·h` per entry: an admittance or impedance grows with frequency
    /// (a capacitance, an inductance); a bounded response (scattering parameters) has none.
    pub proportional: bool,
    /// The √s skin-effect branch term per entry ([`fit_auto_with`]).
    pub sqrt: bool,
    /// Relative weights, `tol` a relative error; off, absolute ones ([`weights`]).
    pub relative: bool,
    /// Per-entry delays on the normalised axis (empty: none): entry `e` is modelled as
    /// `exp(-s·delays[e])` times a rational function, so a propagation delay costs no poles. The
    /// fit takes the delays out of the samples, fits the rational parts and puts the delays back
    /// into the model ([`VfModel::eval`]); [`crate::phase_slope`] estimates them from the samples.
    pub delays: Vec<f64>,
    /// The poles to start from, as [`VfModel::poles`] lists them (the real ones, then each c.c.
    /// pair as the pole with `Im > 0` and its conjugate): a refit after a few more samples starts
    /// from the poles of the fit before. Empty, or more than the order may take: one c.c. pair
    /// and one real pole spread over the band.
    pub start: Vec<C>,
    /// Skim threshold: a pole whose band energy is below this share of the mean pole energy in
    /// every entry is spurious (the scikit-rf default).
    pub skim: f64,
    /// A skim is kept where the rms error grows by no more than this factor (else the pole
    /// was load-bearing and comes back).
    pub skim_slack: f64,
    /// Relocation steps per order.
    pub relocations: usize,
    /// Pairs added in a row without lowering the rms error before the order is final.
    pub patience: usize,
    /// The smallest magnitude of the relaxation constant `d`: below it `d` is fixed at this
    /// magnitude (its sign kept) and σ solved again without the relaxation row (Gustavsen).
    pub relax_floor: f64,
    /// `tol` bounds the largest weighted error over the samples and entries instead of the rms,
    /// and the order grows while that falls: a response of many entries whose tolerance holds
    /// for every entry, where a few entries off by more than it hardly move the rms.
    pub max_error: bool,
}

impl Default for Options {
    /// The admittance fit: the proportional term, no √s term, relative weights, no delays, the
    /// spread start, and the order search of [`fit_auto`].
    fn default() -> Self {
        Options {
            proportional: true,
            sqrt: false,
            relative: true,
            delays: Vec::new(),
            start: Vec::new(),
            skim: 0.03,
            skim_slack: 1.05,
            relocations: 8,
            patience: 3,
            relax_floor: 1e-12,
            max_error: false,
        }
    }
}

struct Vf<'a> {
    s: &'a [C],         // jω samples (normalised)
    data: &'a [Vec<C>], // [sample][entry]
    nf: usize,
    // the LS weights wt[sample][entry] ([`weights`]); all one where `uniform`
    wt: Vec<Vec<f64>>,
    uniform: bool,
    // the entries the rms error is averaged over (`nf`, or the entries a compressed response
    // stands for)
    norm_entries: usize,
    relax_floor: f64,
    fit_const: bool,
    fit_diff: bool,
    fit_sqrt: bool,
    pr: Vec<f64>, // real poles (negative)
    pc: Vec<C>,   // complex poles, one per c.c. pair (Re<0, Im>0)
    // F residues per entry, and const/diff
    rr: Vec<Vec<C>>, // [entry][real pole]
    rc: Vec<Vec<C>>, // [entry][cpx pole]
    cst: Vec<C>,
    dif: Vec<C>,
    sqt: Vec<C>,
    // σ residues + relaxation
    sr: Vec<f64>,
    sc: Vec<C>,
    d_relax: f64,
    // reused scratch for the cached partial fractions (xf: n·nrf, xs: n·nrs)
    xf: Vec<C>,
    xs: Vec<C>,
}

impl<'a> Vf<'a> {
    /// New workspace over samples `s` / response `data` seeded with the given initial poles,
    /// weighted relatively per entry (`relative`) or absolutely ([`weights`]).
    fn new(s: &'a [C], data: &'a [Vec<C>], pc: Vec<C>, pr: Vec<f64>, relative: bool) -> Self {
        let nf = data[0].len();
        Vf {
            s,
            data,
            nf,
            wt: weights(data, relative),
            uniform: !relative,
            norm_entries: nf,
            relax_floor: Options::default().relax_floor,
            fit_const: true,
            fit_diff: true,
            fit_sqrt: false,
            pr,
            pc,
            rr: Vec::new(),
            rc: Vec::new(),
            cst: Vec::new(),
            dif: Vec::new(),
            sqt: Vec::new(),
            sr: Vec::new(),
            sc: Vec::new(),
            d_relax: 1.0,
            xf: Vec::new(),
            xs: Vec::new(),
        }
    }

    fn ncd(&self) -> usize {
        self.fit_const as usize + self.fit_diff as usize + self.fit_sqrt as usize
    }
    fn nrf(&self) -> usize {
        self.ncd() + self.pr.len() + 2 * self.pc.len()
    }
    fn nrs(&self) -> usize {
        self.pr.len() + 2 * self.pc.len()
    }

    /// Cache the partial fractions for all samples into the reused scratch buffers.
    fn cache_x(&mut self) {
        let (n, nrf, nrs) = (self.s.len(), self.nrf(), self.nrs());
        self.xf.resize(n * nrf, C::new(0.0, 0.0));
        self.xs.resize(n * nrs, C::new(0.0, 0.0));
        for k in 0..n {
            let (xf, xs) = (
                &mut self.xf[k * nrf..(k + 1) * nrf],
                &mut self.xs[k * nrs..(k + 1) * nrs],
            );
            fill_x_into(
                self.s[k],
                &self.pr,
                &self.pc,
                self.fit_const,
                self.fit_diff,
                self.fit_sqrt,
                xf,
                xs,
            );
        }
    }

    /// Relaxed σ identification: solve for the σ residues and the relaxation constant `d_relax`
    /// jointly over ALL entries (common poles), with a normalisation row that forbids the
    /// trivial σ≡0. The per-entry F residues are nuisance unknowns here (refit later). Where
    /// `|d_relax|` comes out below the floor, it is fixed there (its sign kept) and σ is solved
    /// again without the normalisation row, the `d` column on the right-hand side (Gustavsen).
    fn compute_sigma(&mut self) {
        let (n, nf) = (self.s.len(), self.nf);
        let (nrf, nrs) = (self.nrf(), self.nrs());
        let nu = nrs + 1; // σ residues + d_relax (the only shared unknowns)
        if 2 * n <= nrf {
            return; // a per-entry block cannot determine its F residues — keep current poles
        }
        self.cache_x();
        // FAST (Deschrijver): the per-entry F residues are nuisance unknowns. For each entry,
        // QR `[X_F | −X_S·D | −D]` and keep only the σ-block R22 (the rows orthogonal to X_F's
        // column space) — this eliminates the F residues. Stacking R22 over entries gives a
        // SMALL shared σ system (nf·rblk+1 × nu) instead of the full nf·nrf+nu-wide one. The
        // data RHS is 0, so the projected RHS is 0 and Q is never needed.
        let ncol = nrf + nu;
        let rblk = (2 * n).min(ncol) - nrf; // R22 rows each entry contributes
        if nf * rblk + 1 < nu {
            return; // stacked σ system under-determined (rows < cols) — keep current poles
        }
        let (xf, xs, data) = (&self.xf, &self.xs, self.data);
        let mut rr = Mat::<f64>::zeros(nf * rblk + 1, nu);
        let wt = &self.wt;
        // The per-entry QRs are independent (disjoint R22 blocks) and dominate the
        // fit cost on multiport responses (nf = p² entries) — run them in the
        // ambient rayon pool. Assembly stays in entry order, so the stacked σ
        // system (and the whole fit) is bit-identical to the serial path.
        let r22: Vec<Vec<f64>> = (0..nf)
            .into_par_iter()
            .map(|e| {
                let ae = Mat::<f64>::from_fn(2 * n, ncol, |r, c| {
                    let imag = r >= n;
                    let k = if imag { r - n } else { r };
                    let v: C = if c < nrf {
                        xf[k * nrf + c]
                    } else if c < nrf + nrs {
                        -xs[k * nrs + (c - nrf)] * data[k][e]
                    } else {
                        -data[k][e]
                    };
                    // Relative weighting: scale the whole equation (both parts of the
                    // complex sample) by 1/|data[k][e]| so every entry is identified to
                    // its own accuracy.
                    let v = v * wt[k][e];
                    if imag {
                        v.im
                    } else {
                        v.re
                    }
                });
                let rmat = ae.qr().thin_R().to_owned();
                let mut blk = vec![0.0f64; rblk * nu];
                for i in 0..rblk {
                    for j in 0..nu {
                        blk[i * nu + j] = rmat[(nrf + i, nrf + j)];
                    }
                }
                blk
            })
            .collect();
        for (e, blk) in r22.iter().enumerate() {
            for i in 0..rblk {
                for j in 0..nu {
                    rr[(e * rblk + i, j)] = blk[i * nu + j];
                }
            }
        }
        // Relaxation row (Re{Σ_k σ basis}·c + N·d_relax = N) forbids the trivial σ≡0.
        for c in 0..nrs {
            rr[(nf * rblk, c)] = (0..n).map(|k| xs[k * nrs + c].re).sum();
        }
        rr[(nf * rblk, nrs)] = n as f64;
        let rhs = Mat::<f64>::from_fn(nf * rblk + 1, 1, |r, _| {
            if r == nf * rblk {
                n as f64
            } else {
                0.0
            }
        });
        let mut x = rr.col_piv_qr().solve_lstsq(&rhs);
        let mut d = x[(nrs, 0)];
        if d.abs() < self.relax_floor && nrs > 0 {
            d = if d < 0.0 {
                -self.relax_floor
            } else {
                self.relax_floor
            };
            let m = nf * rblk;
            let a = Mat::<f64>::from_fn(m, nrs, |r, c| rr[(r, c)]);
            let b = Mat::<f64>::from_fn(m, 1, |r, _| -rr[(r, nrs)] * d);
            x = a.col_piv_qr().solve_lstsq(&b);
        }
        let (npr, npc) = (self.pr.len(), self.pc.len());
        self.sr = (0..npr).map(|i| x[(i, 0)]).collect();
        self.sc = (0..npc)
            .map(|i| C::new(x[(npr + i, 0)], x[(npr + npc + i, 0)]))
            .collect();
        self.d_relax = d;
    }

    /// New poles = zeros of σ: eigenvalues of `A − b·rᵀ/d_relax` (real block form), split into
    /// real and c.c.-pair representatives.
    fn compute_poles(&mut self) {
        let (nr, nc) = (self.pr.len(), self.pc.len());
        let dim = nr + 2 * nc;
        if dim == 0 {
            return;
        }
        // If the σ residues are out of sync with the pole set (compute_sigma bails on an
        // under-determined σ system after a pole was seeded/skimmed), σ carries no relocation
        // information — keep the CURRENT poles instead of indexing stale residues (crash).
        if self.sr.len() != nr || self.sc.len() != nc {
            return;
        }
        let mut a = Mat::<f64>::zeros(dim, dim);
        let mut b = vec![0.0f64; dim];
        for i in 0..nr {
            a[(i, i)] = self.pr[i];
            b[i] = 1.0;
        }
        for i in 0..nc {
            let j = nr + 2 * i;
            let (re, im) = (self.pc[i].re, self.pc[i].im);
            a[(j, j)] = re;
            a[(j + 1, j + 1)] = re;
            a[(j, j + 1)] = im;
            a[(j + 1, j)] = -im;
            b[j] = 2.0;
        }
        let mut r = vec![0.0f64; dim];
        r[..nr].copy_from_slice(&self.sr[..nr]);
        for i in 0..nc {
            r[nr + 2 * i] = self.sc[i].re;
            r[nr + 2 * i + 1] = self.sc[i].im;
        }
        for i in 0..dim {
            for k in 0..dim {
                a[(i, k)] -= b[i] * r[k] / self.d_relax;
            }
        }
        // If the companion eigensolve fails (very ill-conditioned σ), keep the CURRENT poles
        // rather than collapsing the model to const+diff (empty pole set) — the next iteration
        // re-relocates from a valid set instead of silently degrading.
        let eigs = match a.eigenvalues() {
            Ok(e) if !e.is_empty() => e,
            _ => return,
        };
        let itol = self.s.iter().map(|z| z.im.abs()).fold(0.0, f64::max) * 1e-9;
        let (mut pr, mut pc) = (Vec::new(), Vec::new());
        for p in eigs {
            if p.im.abs() <= itol {
                pr.push(p.re);
            } else if p.im > 0.0 {
                pc.push(p);
            }
        }
        self.pr = pr;
        self.pc = pc;
    }

    fn enforce_stability(&mut self) {
        for p in self.pr.iter_mut() {
            *p = -p.abs();
        }
        for p in self.pc.iter_mut() {
            *p = C::new(-p.re.abs(), p.im);
        }
    }

    /// Fit the F residues (and const/diff) per entry, given the current poles. With absolute
    /// weights the basis matrix is the SAME for every entry: factored once, every entry a column
    /// of one right-hand side. With relative weights each entry has its own rows, factored per
    /// entry.
    fn compute_residues(&mut self) {
        let (nrf, ncd, n) = (self.nrf(), self.ncd(), self.s.len());
        let (nr, nc) = (self.pr.len(), self.pc.len());
        if 2 * n < nrf {
            return; // under-determined (order capped elsewhere to avoid this)
        }
        self.cache_x();
        let xf = &self.xf;
        let (data, wt) = (self.data, &self.wt);
        let (fit_const, fit_diff, fit_sqrt) = (self.fit_const, self.fit_diff, self.fit_sqrt);
        let basis = |e: usize| {
            Mat::<f64>::from_fn(2 * n, nrf, |r, c| {
                let imag = r >= n;
                let k = if imag { r - n } else { r };
                let v = xf[k * nrf + c];
                (if imag { v.im } else { v.re }) * wt[k][e]
            })
        };
        let rhs = |e0: usize, ne: usize| {
            Mat::<f64>::from_fn(2 * n, ne, |r, c| {
                let imag = r >= n;
                let k = if imag { r - n } else { r };
                let d = data[k][e0 + c];
                (if imag { d.im } else { d.re }) * wt[k][e0 + c]
            })
        };
        // the solution columns, one per entry, in entry order
        let x: Vec<Vec<f64>> = if self.uniform {
            let x = basis(0).col_piv_qr().solve_lstsq(&rhs(0, self.nf));
            (0..self.nf)
                .map(|e| (0..nrf).map(|i| x[(i, e)]).collect())
                .collect()
        } else {
            // Relative weighting makes the basis entry-dependent (each row scaled by
            // 1/|data[k][e]|), so factor per entry — independent solves, parallel over
            // the entries (nf = p² on a multiport), collected in entry order
            // (bit-identical to serial).
            (0..self.nf)
                .into_par_iter()
                .map(|e| {
                    let x = basis(e).col_piv_qr().solve_lstsq(&rhs(e, 1));
                    (0..nrf).map(|i| x[(i, 0)]).collect()
                })
                .collect()
        };
        self.rr = Vec::with_capacity(self.nf);
        self.rc = Vec::with_capacity(self.nf);
        self.cst = Vec::with_capacity(self.nf);
        self.dif = Vec::with_capacity(self.nf);
        self.sqt = Vec::with_capacity(self.nf);
        for x in x {
            let mut o = 0;
            let mut cst = C::new(0.0, 0.0);
            if fit_const {
                cst = C::new(x[o], 0.0);
                o += 1;
            }
            let mut dif = C::new(0.0, 0.0);
            if fit_diff {
                dif = C::new(x[o], 0.0);
                o += 1;
            }
            let mut sqt = C::new(0.0, 0.0);
            if fit_sqrt {
                sqt = C::new(x[o], 0.0); // last poly slot
            }
            self.cst.push(cst);
            self.dif.push(dif);
            self.sqt.push(sqt);
            self.rr
                .push((0..nr).map(|i| C::new(x[ncd + i], 0.0)).collect());
            self.rc.push(
                (0..nc)
                    .map(|i| C::new(x[ncd + nr + i], x[ncd + nr + nc + i]))
                    .collect(),
            );
        }
    }

    /// One relaxed-VF relocation sweep: identify σ, move the poles to its zeros, reflect unstable
    /// ones back into the left half-plane, refit the residues — repeated until `tol` or `max_steps`.
    fn relocate(&mut self, tol: f64, max_steps: usize) {
        for _ in 0..max_steps {
            self.compute_sigma();
            self.compute_poles();
            self.enforce_stability();
            self.compute_residues();
            if self.err_max() < tol {
                break;
            }
        }
    }

    /// Drop *spurious* poles — those whose isolated resonance carries negligible band-limited
    /// energy relative to the mean (Grivet-Talocia/Bandinu skimming; scikit-rf `get_spurious`).
    /// For pole `m` with per-entry residue `r`, `hₘ,ₑ(jω)=r/(jω−p)+r*/(jω−p*)` (a real pole omits
    /// the conjugate term); its energy `‖hₘ,ₑ‖₂` is measured over the samples. A pole is skimmed
    /// iff, for EVERY entry, its energy is below `gamma×mean` — i.e. it contributes to none of the
    /// responses and only fits noise. Returns whether anything was removed (caller refits residues).
    fn skim_spurious(&mut self, gamma: f64) -> bool {
        let (nr, nc) = (self.pr.len(), self.pc.len());
        if nr + nc == 0 {
            return false;
        }
        // Residues out of sync with the pole set (compute_residues bailed on an
        // under-determined system) — no energies to rank, keep every pole.
        if self.rr.iter().any(|r| r.len() != nr) || self.rc.iter().any(|r| r.len() != nc) {
            return false;
        }
        // Band energy e[pole][entry]; poles ordered [real…, cpx…].
        let npole = nr + nc;
        let mut energy = vec![vec![0.0f64; self.nf]; npole];
        for entry in 0..self.nf {
            for i in 0..nr {
                let (p, r) = (C::new(self.pr[i], 0.0), self.rr[entry][i]);
                energy[i][entry] = self
                    .s
                    .iter()
                    .map(|&z| (r / (z - p)).norm_sqr())
                    .sum::<f64>()
                    .sqrt();
            }
            for i in 0..nc {
                let (p, r) = (self.pc[i], self.rc[entry][i]);
                energy[nr + i][entry] = self
                    .s
                    .iter()
                    .map(|&z| (r / (z - p) + r.conj() / (z - p.conj())).norm_sqr())
                    .sum::<f64>()
                    .sqrt();
            }
        }
        // Per-entry mean over poles; a pole is spurious iff below gamma×mean for ALL entries.
        let mean: Vec<f64> = (0..self.nf)
            .map(|e| energy.iter().map(|row| row[e]).sum::<f64>() / npole as f64)
            .collect();
        let spurious: Vec<bool> = (0..npole)
            .map(|m| (0..self.nf).all(|e| mean[e] <= 0.0 || energy[m][e] < gamma * mean[e]))
            .collect();
        if !spurious.iter().any(|&x| x) {
            return false;
        }
        self.pr = (0..nr)
            .filter(|&i| !spurious[i])
            .map(|i| self.pr[i])
            .collect();
        self.pc = (0..nc)
            .filter(|&i| !spurious[nr + i])
            .map(|i| self.pc[i])
            .collect();
        true
    }

    /// Expand to all poles + per-entry residues (each c.c. pair → the pole and its conjugate).
    fn model(&self) -> VfModel {
        let mut poles = self.pr.iter().map(|&p| C::new(p, 0.0)).collect::<Vec<_>>();
        for &p in &self.pc {
            poles.push(p);
            poles.push(p.conj());
        }
        let res = (0..self.nf)
            .map(|e| {
                let mut r: Vec<C> = self.rr[e].clone();
                for &rc in &self.rc[e] {
                    r.push(rc);
                    r.push(rc.conj());
                }
                r
            })
            .collect();
        VfModel {
            poles,
            res,
            cst: self.cst.clone(),
            dif: self.dif.clone(),
            sqt: self.sqt.clone(),
            delay: vec![0.0; self.nf],
            d: self.nf,
        }
    }

    /// Maximum weighted fit error over all samples/entries — each residual weighted by the same
    /// `wt` the LS uses ([`weights`]), so the tolerance the greedy chases is a true relative
    /// error on every entry with relative weights (not an absolute one dominated by the
    /// largest-magnitude entry), an absolute one without.
    fn err_max(&self) -> f64 {
        let m = self.model();
        let mut e = 0.0f64;
        for (k, &s) in self.s.iter().enumerate() {
            let got = m.eval(s);
            for ent in 0..self.nf {
                e = e.max((got[ent] - self.data[k][ent]).norm() * self.wt[k][ent]);
            }
        }
        e
    }

    /// RMS weighted fit error over all samples/entries. Unlike the max, a single pole that nulls
    /// one anchor's residual barely moves it — so the RMS has a clean elbow at the PHYSICAL pole
    /// count, which is what stops the order growth before it fits noise. Relative weighting keeps
    /// the elbow honest when the entries span a large magnitude range. Averaged over the entries
    /// the response has (`norm_entries`): a compressed response is the same sum of squares over
    /// fewer columns.
    fn err_rms(&self) -> f64 {
        let m = self.model();
        let mut s2 = 0.0f64;
        for (k, &s) in self.s.iter().enumerate() {
            let got = m.eval(s);
            for ent in 0..self.nf {
                let d = (got[ent] - self.data[k][ent]).norm() * self.wt[k][ent];
                s2 += d * d;
            }
        }
        (s2 / (self.s.len() * self.norm_entries) as f64).sqrt()
    }

    /// Sample index of the largest (entry-summed) weighted residual — where a new pole should be
    /// seeded (the worst-fit band, matching what the greedy is minimising).
    /// `exclude` skips seed locations already tried this growth round, so a retry after a
    /// non-improving addition explores the NEXT-worst band instead of re-seeding the same spot.
    fn worst_sample(&self, exclude: &[usize]) -> usize {
        let m = self.model();
        let mut best = (0usize, -1.0f64);
        for (k, &s) in self.s.iter().enumerate() {
            if exclude.contains(&k) {
                continue;
            }
            let got = m.eval(s);
            let e: f64 = (0..self.nf)
                .map(|ent| (got[ent] - self.data[k][ent]).norm() * self.wt[k][ent])
                .sum();
            if e > best.1 {
                best = (k, e);
            }
        }
        best.0
    }
}

/// The damping of a starting pole as a share of its frequency: `-a + jb` with `a = b * this`
/// (Gustavsen and Semlyen 1999, `a = b / 100` for complex starting poles); the real starting
/// poles and the pairs seeded while the order grows take the same share.
const START_DAMPING: f64 = 0.01;

/// Even-spread, lightly-damped (left-half-plane) initial poles across the band.
fn spread_cpx(n: usize, wmax: f64) -> Vec<C> {
    (0..n)
        .map(|i| {
            let w = wmax * (i as f64 + 1.0) / (n as f64 + 1.0);
            C::new(-w * START_DAMPING, w)
        })
        .collect()
}
fn spread_real(n: usize, wmax: f64) -> Vec<f64> {
    (0..n)
        .map(|i| -wmax * (i as f64 + 1.0) / (n as f64 + 1.0) * START_DAMPING)
        .collect()
}

/// Singular values of the stacked real and imaginary parts of a response (samples x entries)
/// below this share of the largest are left out of its compression ([`compress`]): round-off,
/// a few hundred epsilons.
const COMPRESS_DROP: f64 = 1e-13;

/// The response `data` (samples x entries) as the real combinations of its entries that span
/// it: the thin SVD of its real and imaginary parts stacked (`[Re; Im] = U Σ Vᵀ`, real V, so
/// every combination is the response of a real system again), the columns `U Σ` above
/// [`COMPRESS_DROP`] as complex samples. A least-squares fit with given poles leaves the same
/// residual (the Frobenius norm over all samples and entries) on these as on `data`: its
/// residual is `(I − P)·[Re; Im]`, and `[Re; Im] = (U Σ) Vᵀ` with orthonormal `Vᵀ`. At most
/// `2·samples` columns, however many entries.
fn compress(data: &[Vec<C>]) -> Vec<Vec<C>> {
    let (ns, nf) = (data.len(), data[0].len());
    let a = Mat::<f64>::from_fn(2 * ns, nf, |r, e| {
        if r < ns {
            data[r][e].re
        } else {
            data[r - ns][e].im
        }
    });
    let Ok(svd) = a.thin_svd() else {
        return data.to_vec();
    };
    let sv: Vec<f64> = svd.S().column_vector().iter().copied().collect();
    let top = sv.first().copied().unwrap_or(0.0);
    let q = sv
        .iter()
        .filter(|&&x| x > COMPRESS_DROP * top)
        .count()
        .max(1);
    let u = svd.U();
    (0..ns)
        .map(|k| {
            (0..q)
                .map(|j| C::new(u[(k, j)] * sv[j], u[(ns + k, j)] * sv[j]))
                .collect()
        })
        .collect()
}

/// The delays of `opts` per entry (zeros where it has none), and the samples with them taken out
/// (`None` where every delay is zero): `exp(s·τ_e)·data_e`, the rational parts a fit models.
fn without_delays(s: &[C], data: &[Vec<C>], opts: &Options) -> (Vec<f64>, Option<Vec<Vec<C>>>) {
    let nf = data[0].len();
    let delays = if opts.delays.is_empty() {
        vec![0.0; nf]
    } else {
        assert_eq!(opts.delays.len(), nf, "one delay per entry");
        opts.delays.clone()
    };
    let shifted = delays.iter().any(|&t| t != 0.0).then(|| {
        s.iter()
            .zip(data)
            .map(|(&z, row)| {
                row.iter()
                    .zip(&delays)
                    .map(|(v, &t)| v * (z * t).exp())
                    .collect()
            })
            .collect()
    });
    (delays, shifted)
}

/// Poles as [`VfModel::poles`] lists them split into the c.c. pair representatives (`Im > 0`)
/// and the real poles, every one moved into the left half-plane.
fn split_poles(poles: &[C]) -> (Vec<C>, Vec<f64>) {
    (
        poles
            .iter()
            .filter(|p| p.im > 0.0)
            .map(|p| C::new(-p.re.abs(), p.im))
            .collect(),
        poles
            .iter()
            .filter(|p| p.im == 0.0)
            .map(|p| -p.re.abs())
            .collect(),
    )
}

/// The residues, constants and the optional terms of every entry for the given `poles` (as
/// [`VfModel::poles`] lists them), with the basis, weights and delays of `opts`: one
/// least-squares solve, no relocation (a refit of a model on other samples, a leave-one-out
/// model). `None` where the samples do not determine them: two equations a sample, fewer than
/// the poles, the constant and the optional terms.
pub fn fit_residues(s: &[C], data: &[Vec<C>], poles: &[C], opts: &Options) -> Option<VfModel> {
    super::pin_faer_sequential();
    let (delays, shifted) = without_delays(s, data, opts);
    let data = shifted.as_deref().unwrap_or(data);
    let (pc, pr) = split_poles(poles);
    let mut vf = Vf::new(s, data, pc, pr, opts.relative);
    vf.fit_diff = opts.proportional;
    vf.fit_sqrt = opts.sqrt;
    if 2 * s.len() < vf.nrf() {
        return None;
    }
    vf.compute_residues();
    let mut m = vf.model();
    m.delay = delays;
    Some(m)
}

/// Fixed-order set-valued fit by relaxed vector fitting. `n_cpx`/`n_real` are the initial pole
/// counts; `tol`/`max_steps` the relocation convergence target. The order is whatever the poles
/// relocate to (a c.c. pair may split into two reals or vice-versa), NOT trimmed here. The
/// fixed-order primitive `fit_auto` builds on — part of the public surface, exercised by the tests.
#[allow(dead_code)]
pub fn fit(
    s: &[C],
    data: &[Vec<C>],
    n_cpx: usize,
    n_real: usize,
    tol: f64,
    max_steps: usize,
) -> VfModel {
    super::pin_faer_sequential();
    let wmax = s.iter().map(|z| z.im.abs()).fold(0.0, f64::max).max(1e-12);
    let mut vf = Vf::new(
        s,
        data,
        spread_cpx(n_cpx, wmax),
        spread_real(n_real, wmax),
        true,
    );
    vf.compute_residues();
    vf.relocate(tol, max_steps);
    vf.model()
}

/// Automatic-order set-valued fit by *adding and skimming* (Grivet-Talocia/Bandinu; the scheme
/// behind scikit-rf's `auto_fit`). Start from one c.c. pair, relocate, and repeatedly: add a pair
/// seeded at the worst-error band, relocate, then skim poles whose band energy is spurious. Stop at
/// `tol`, when a new pair no longer cuts the RMS error (parsimony), or at the over-determination
/// cap. This decouples the model order from the anchor count and never fits solver noise into
/// spurious poles — the source of the between-anchor wobble/spikes. `max_poles` caps the order.
pub fn fit_auto(s: &[C], data: &[Vec<C>], tol: f64, max_poles: usize) -> VfModel {
    fit_auto_with(s, data, tol, max_poles, false)
}

/// [`fit_auto`] with an optional **√s skin-effect branch term** in the basis:
/// surface impedance goes as √f, which is not rational —
/// a common-pole fit floors at ~1e-3 on wide lossy bands with the residual
/// error piled at the top band edge. One real √s coefficient per entry (the
/// KK-consistent resistance+inductance pair) removes that floor. The LS drives
/// the coefficient to ~0 when the data carries no √f content, so enabling it
/// is safe whenever enough samples support the extra column.
pub fn fit_auto_with(
    s: &[C],
    data: &[Vec<C>],
    tol: f64,
    max_poles: usize,
    sqrt_term: bool,
) -> VfModel {
    fit_auto_opts(
        s,
        data,
        tol,
        max_poles,
        &Options {
            sqrt: sqrt_term,
            ..Options::default()
        },
    )
}

/// [`fit_auto`] with the basis, weights, delays, start and order search of `opts`. The order
/// never exceeds `max_poles` (counting both members of a c.c. pair), nor what the samples
/// determine: the spread start is one c.c. pair and one real pole where both fit, fewer where
/// they do not. With absolute weights and more entries than real equations a sample, the poles
/// are identified on the compressed response ([`compress`]) and the residues of every entry
/// fitted with them at the end.
pub fn fit_auto_opts(
    s: &[C],
    data: &[Vec<C>],
    tol: f64,
    max_poles: usize,
    opts: &Options,
) -> VfModel {
    super::pin_faer_sequential();
    let ns = s.len();
    let nf = data[0].len();
    let (delays, shifted) = without_delays(s, data, opts);
    let data = shifted.as_deref().unwrap_or(data);
    // the response the poles are identified on
    let compressed: Option<Vec<Vec<C>>> = (!opts.relative && nf > 2 * ns).then(|| compress(data));
    let ident: &[Vec<C>] = compressed.as_deref().unwrap_or(data);
    let nq = ident[0].len();
    let wmax = s.iter().map(|z| z.im.abs()).fold(0.0, f64::max).max(1e-12);
    // the lowest sample above DC: the start's real pole and the floor of a seeded pair, so a
    // pole near DC (a port grounded through a stub, `1/(R + sL)` in Y) is in reach
    let wlo = s
        .iter()
        .map(|z| z.im.abs())
        .filter(|&w| w > 0.0)
        .fold(wmax, f64::min);
    // Common-pole σ identification is over-determined only for nc ≤ (2·ns·nf − 3nf − 1)/(2(nf+1)).
    let nc_lim = (2 * ns * nq).saturating_sub(3 * nq + 1) / (2 * (nq + 1));
    let max_cpx = (max_poles / 2).min(nc_lim);
    // The starting order: the given poles where the order may take them, else one c.c. pair
    // spread over the band and one real pole at the lowest sample, fewer where `max_poles` or the
    // samples allow fewer (the residue LS of an entry needs as many equations, two a sample, as
    // unknowns: the poles, the constant and the optional terms).
    let ncd = 1 + opts.proportional as usize + opts.sqrt as usize;
    let room = max_poles.min((2 * ns).saturating_sub(ncd));
    let (pc0, pr0) = if !opts.start.is_empty() && opts.start.len() <= room {
        split_poles(&opts.start)
    } else {
        let (n_cpx0, n_real0) = match room {
            0 => (0, 0),
            1 => (0, 1),
            2 => (1, 0),
            _ => (1, 1),
        };
        (spread_cpx(n_cpx0, wmax), vec![-wlo; n_real0])
    };

    // the error the order is grown on: the rms, or the largest error over the response itself
    // (the residues of every entry fitted with the poles where they were identified on the
    // compressed response)
    let score = |vf: &Vf| -> f64 {
        if !opts.max_error {
            return vf.err_rms();
        }
        if compressed.is_none() {
            return vf.err_max();
        }
        let mut full = Vf::new(s, data, vf.pc.clone(), vf.pr.clone(), opts.relative);
        full.fit_diff = opts.proportional;
        full.fit_sqrt = opts.sqrt;
        full.compute_residues();
        full.err_max()
    };
    let mut vf = Vf::new(s, ident, pc0, pr0, opts.relative);
    vf.norm_entries = nf;
    vf.relax_floor = opts.relax_floor;
    vf.fit_diff = opts.proportional;
    vf.fit_sqrt = opts.sqrt;
    vf.compute_residues();
    vf.relocate(tol, opts.relocations);
    // Initial skim, guarded like the in-loop one (see below).
    let pre0 = score(&vf);
    let (pr0, pc0) = (vf.pr.clone(), vf.pc.clone());
    if vf.skim_spurious(opts.skim) {
        vf.compute_residues();
        if score(&vf) > pre0 * opts.skim_slack {
            vf.pr = pr0;
            vf.pc = pc0;
            vf.compute_residues();
        }
    }
    let mut best = vf.model();
    let mut best_poles = (vf.pr.clone(), vf.pc.clone());
    let mut best_err = score(&vf);
    crate::vflog!(
        "[vf] start: {ns} samples x {nf} entries ({nq} identified), cpx={} real={} err={:.2e} (tol={tol:.0e}, max_cpx={max_cpx})",
        vf.pc.len(),
        vf.pr.len(),
        best_err
    );

    // Growth patience: a single non-improving addition is NOT proof the order is
    // exhausted — multi-resonance data routinely plateaus while growing through an
    // intermediate order (the added pair lands on one resonance, the RMS is still
    // dominated by the next unseen one). Allow up to `patience` consecutive misses,
    // seeding each retry at the next-worst band, before declaring parsimony.
    let mut tried: Vec<usize> = Vec::new();
    let mut misses = 0usize;
    while vf.pc.len() < max_cpx
        && vf.pr.len() + 2 * (vf.pc.len() + 1) <= max_poles
        && best_err >= tol
        && misses < opts.patience
    {
        // A further c.c. pair must keep BOTH least-squares stages over-determined
        // (per-entry F residues: 2·ns > nrf; σ additionally needs its shared block) —
        // otherwise compute_residues/compute_sigma bail and the model cannot improve.
        // nr can grow during relocation (c.c. pairs splitting into real poles), so this
        // is checked per iteration, not folded into max_cpx up front.
        if 2 * ns <= vf.nrf() + 2 {
            break;
        }
        // Seed a new c.c. pair at the worst-error band (skipping spots already tried
        // this round) and re-relocate the whole set.
        let k = vf.worst_sample(&tried);
        tried.push(k);
        let w = s[k].im.abs().max(wlo);
        vf.pc.push(C::new(-w * START_DAMPING, w));
        vf.compute_residues();
        vf.relocate(tol, opts.relocations);
        // Guarded skim: the energy criterion compares against the MEAN pole energy,
        // which resonance-dominated responses inflate — a freshly seeded pair then
        // reads as "spurious" before it can specialise, and the order can never grow
        // (measured: every added pair was skimmed right back, three misses, stall at
        // 7 poles / rms 1.5e-2). A genuinely spurious pole contributes nothing, so
        // removing it leaves the RMS unchanged — accept a skim only on that evidence,
        // and roll it back when the RMS degrades (the pole was load-bearing).
        let n_before = vf.pr.len() + vf.pc.len();
        let pre_err = score(&vf);
        let (pr_bak, pc_bak) = (vf.pr.clone(), vf.pc.clone());
        if vf.skim_spurious(opts.skim) {
            vf.compute_residues();
            let post_err = score(&vf);
            let n_after = vf.pr.len() + vf.pc.len();
            if post_err > pre_err * opts.skim_slack {
                vf.pr = pr_bak;
                vf.pc = pc_bak;
                vf.compute_residues();
                crate::vflog!(
                    "[vf]   skim rejected ({n_before} -> {n_after} poles would cost err {pre_err:.2e} -> {post_err:.2e})"
                );
            } else {
                crate::vflog!(
                    "[vf]   skim: {} -> {} poles (err {pre_err:.2e} -> {post_err:.2e})",
                    n_before,
                    vf.pr.len() + vf.pc.len()
                );
            }
        }
        let e = score(&vf);
        if e < best_err {
            crate::vflog!(
                "[vf]   +pair@w={w:.3}: cpx={} real={} err={e:.2e} (improved)",
                vf.pc.len(),
                vf.pr.len()
            );
            best = vf.model();
            best_poles = (vf.pr.clone(), vf.pc.clone());
            best_err = e;
            misses = 0;
            tried.clear();
        } else {
            misses += 1;
            crate::vflog!(
                "[vf]   +pair@w={w:.3}: cpx={} real={} err={e:.2e} (miss {misses}/{}, best {best_err:.2e})",
                vf.pc.len(),
                vf.pr.len(),
                opts.patience
            );
        }
    }
    if compressed.is_some() {
        // the residues of every entry with the poles identified on the compressed response
        let (pr, pc) = best_poles;
        let mut full = Vf::new(s, data, pc, pr, opts.relative);
        full.fit_diff = opts.proportional;
        full.fit_sqrt = opts.sqrt;
        full.compute_residues();
        best = full.model();
    }
    crate::vflog!("[vf] done: {} poles, err={best_err:.2e}", best.poles.len());
    best.delay = delays;
    best
}

#[cfg(test)]
mod par_tests {
    use super::*;

    /// Synthetic multiport response: nf entries sharing npoles c.c. pairs.
    fn synth(ns: usize, nf: usize, npole: usize) -> (Vec<C>, Vec<Vec<C>>) {
        let s: Vec<C> = (0..ns)
            .map(|i| C::new(0.0, 0.02 + 3.0 * i as f64 / (ns - 1) as f64))
            .collect();
        let poles: Vec<C> = (0..npole)
            .map(|k| C::new(-0.02 - 0.01 * k as f64, 0.1 + 2.8 * k as f64 / npole as f64))
            .collect();
        let data: Vec<Vec<C>> = s
            .iter()
            .map(|&z| {
                (0..nf)
                    .map(|e| {
                        let mut v = C::new(0.05 * ((e % 7) as f64 - 3.0), 0.0);
                        for (k, p) in poles.iter().enumerate() {
                            let r = C::new(
                                0.3 * (((e + k) % 5) as f64 - 2.0),
                                0.2 * (((e * 3 + k) % 4) as f64 - 1.5),
                            );
                            v += r / (z - p) + r.conj() / (z - p.conj());
                        }
                        v
                    })
                    .collect()
            })
            .collect();
        (s, data)
    }

    /// The parallel per-entry LS stages must be bit-identical across pool sizes
    /// (disjoint outputs, fixed assembly order) — the crate-wide determinism
    /// contract extends to the vector fit.
    #[test]
    fn fit_is_bit_identical_across_thread_counts() {
        let (s, data) = synth(120, 16, 6);
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| fit_auto(&s, &data, 1e-8, 16))
        };
        let (a, b) = (run(1), run(8));
        assert_eq!(a.poles, b.poles);
        assert_eq!(a.res, b.res);
        assert_eq!(a.cst, b.cst);
        assert_eq!(a.dif, b.dif);
    }

    /// The many-port compression: a response of more entries than real equations a sample
    /// (absolute weights) is fitted on its compressed form to the accuracy of the full fit, and
    /// a warm start from the poles of that fit keeps that accuracy with no more poles.
    #[test]
    fn a_compressed_many_port_fit_matches_and_restarts() {
        let (s, data) = synth(30, 100, 4);
        let opts = Options {
            relative: false,
            ..Options::default()
        };
        let m = fit_auto_opts(&s, &data, 1e-9, 20, &opts);
        let err = s
            .iter()
            .zip(&data)
            .flat_map(|(&z, row)| {
                let got = m.eval(z);
                row.iter()
                    .zip(got)
                    .map(|(a, b)| (a - b).norm())
                    .collect::<Vec<_>>()
            })
            .fold(0.0, f64::max);
        assert!(err < 1e-6, "compressed fit error {err:e}");
        let again = fit_auto_opts(
            &s,
            &data,
            1e-9,
            20,
            &Options {
                start: m.poles().to_vec(),
                ..opts
            },
        );
        assert!(
            again.order() <= m.order(),
            "{} after {}",
            again.order(),
            m.order()
        );
        for (&z, row) in s.iter().zip(&data) {
            for (a, b) in row.iter().zip(again.eval(z)) {
                assert!(
                    (a - b).norm() < 1e-6,
                    "warm-started fit error {}",
                    (a - b).norm()
                );
            }
        }
    }

    /// Timing probe for the multiport fit (release gate; run explicitly):
    /// `cargo test --release fit_multiport_timing -- --ignored --nocapture`
    #[test]
    #[ignore = "timing probe, run in --release"]
    fn fit_multiport_timing() {
        let (s, data) = synth(300, 196, 10); // 14-port worth of entries
        for threads in [1usize, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let t = std::time::Instant::now();
            let m = pool.install(|| fit_auto(&s, &data, 1e-6, 24));
            eprintln!(
                "threads={threads}: {:?} ({} poles)",
                t.elapsed(),
                m.poles.len()
            );
        }
    }
}

#[cfg(test)]
mod sqrt_tests {
    use super::*;

    /// Rational part + a √s skin term, like a lossy-metal wide-band response.
    fn synth_skin(ns: usize, nf: usize, c_sqrt: f64) -> (Vec<C>, Vec<Vec<C>>) {
        let s: Vec<C> = (0..ns)
            .map(|i| C::new(0.0, 0.02 + 1.0 * i as f64 / (ns - 1) as f64))
            .collect();
        let p = C::new(-0.05, 0.4);
        let data: Vec<Vec<C>> = s
            .iter()
            .map(|&z| {
                (0..nf)
                    .map(|e| {
                        let r = C::new(0.3 + 0.1 * e as f64, 0.05);
                        C::new(0.2, 0.0)
                            + r / (z - p)
                            + r.conj() / (z - p.conj())
                            + z.sqrt() * (c_sqrt * (1.0 + e as f64))
                    })
                    .collect()
            })
            .collect();
        (s, data)
    }

    /// The rational-only fit floors on √f content; the √s basis term removes
    /// the floor and recovers the coefficient.
    #[test]
    fn sqrt_term_removes_the_rational_floor() {
        let (s, data) = synth_skin(160, 2, 0.05);
        let rational = fit_auto_with(&s, &data, 1e-9, 20, false);
        let with_sqrt = fit_auto_with(&s, &data, 1e-9, 20, true);
        let err = |m: &VfModel| -> f64 {
            let mut worst = 0.0f64;
            for (k, &z) in s.iter().enumerate() {
                let got = m.eval(z);
                for e in 0..2 {
                    worst = worst.max((got[e] - data[k][e]).norm() / data[k][e].norm().max(1e-12));
                }
            }
            worst
        };
        let (e_rat, e_sqrt) = (err(&rational), err(&with_sqrt));
        assert!(
            e_sqrt < 1e-7,
            "sqrt-basis fit should be near-exact, got {e_sqrt:.2e}"
        );
        assert!(
            e_sqrt < e_rat / 100.0,
            "sqrt basis must beat the rational floor by >100x: rational {e_rat:.2e}, sqrt {e_sqrt:.2e}"
        );
        // Recovered coefficients match the synthesised ones (real, per entry).
        for e in 0..2 {
            let want = 0.05 * (1.0 + e as f64);
            assert!(
                (with_sqrt.sqt[e].re - want).abs() < 1e-6,
                "entry {e}: sqt {} vs {want}",
                with_sqrt.sqt[e].re
            );
        }
    }

    /// Without √f content the extra column is driven to ~zero — enabling the
    /// term is harmless on clean rational data.
    #[test]
    fn sqrt_term_is_inert_on_rational_data() {
        let (s, data) = synth_skin(160, 2, 0.0);
        let m = fit_auto_with(&s, &data, 1e-9, 20, true);
        for e in 0..2 {
            assert!(
                m.sqt[e].norm() < 1e-6,
                "entry {e}: spurious sqt {}",
                m.sqt[e].norm()
            );
        }
    }
}
