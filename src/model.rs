//! The fitted macromodel produced by the vector fit: a common pole set with per-entry residues,
//! const/proportional terms and an optional per-entry delay. Evaluated at `s = jω` (normalised)
//! to reconstruct the response.

use super::C;

/// Fitted model: common `poles` (expanded: the real poles first, then each c.c. pair as the pole
/// and its conjugate), per-entry `res[e]` residues, per-entry constant `cst` / proportional `dif`
/// (s·h) terms, and per-entry delays: entry `e` is `exp(-s·delay[e])` times its rational part.
#[derive(Clone, Debug)]
pub struct VfModel {
    pub(super) poles: Vec<C>,
    pub(super) res: Vec<Vec<C>>,
    pub(super) cst: Vec<C>,
    pub(super) dif: Vec<C>,
    /// Per-entry √s (skin-effect branch) coefficient; all-zero unless the fit
    /// ran with the sqrt term enabled ([`crate::Options::sqrt`]).
    pub(super) sqt: Vec<C>,
    /// Per-entry delay on the normalised axis; all-zero unless the fit took delays
    /// ([`crate::Options::delays`]).
    pub(super) delay: Vec<f64>,
    pub(super) d: usize,
}

impl VfModel {
    pub fn n_support(&self) -> usize {
        self.poles.len() + 1
    }

    /// The number of poles (the real ones and both members of each c.c. pair): the model order.
    pub fn order(&self) -> usize {
        self.poles.len()
    }

    /// The number of entries (functions) the model fits.
    pub fn n_entries(&self) -> usize {
        self.d
    }

    /// The poles: the real ones first, then each c.c. pair as the pole (`Im > 0`) and its
    /// conjugate.
    pub fn poles(&self) -> &[C] {
        &self.poles
    }

    /// The residues of entry `e`, one per pole of [`Self::poles`].
    pub fn residues(&self, e: usize) -> &[C] {
        &self.res[e]
    }

    /// The constant term of entry `e`.
    pub fn constant(&self, e: usize) -> C {
        self.cst[e]
    }

    /// The proportional (`s·h`) coefficient of entry `e`.
    pub fn proportional(&self, e: usize) -> C {
        self.dif[e]
    }

    /// The delay of entry `e` (normalised axis).
    pub fn delay(&self, e: usize) -> f64 {
        self.delay[e]
    }

    /// Entry `e` at `s`.
    pub fn eval_entry(&self, e: usize, s: C) -> C {
        let mut v = self.cst[e] + s * self.dif[e] + s.sqrt() * self.sqt[e];
        for (p, r) in self.poles.iter().zip(&self.res[e]) {
            v += r / (s - p);
        }
        if self.delay[e] != 0.0 {
            v *= (-s * self.delay[e]).exp();
        }
        v
    }

    pub fn eval(&self, s: C) -> Vec<C> {
        (0..self.d).map(|e| self.eval_entry(e, s)).collect()
    }
}
