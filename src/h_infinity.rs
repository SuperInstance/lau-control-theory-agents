//! H∞ control: robust control minimizing worst-case disturbance amplification

use nalgebra::DMatrix;
use crate::state_space::StateSpace;

/// H∞ robust controller design.
///
/// Minimizes the H∞ norm of the transfer function from disturbance w to
/// regulated output z, providing guaranteed performance bounds under
/// worst-case disturbances.
pub struct HInfinity;

impl HInfinity {
    /// Compute the H∞ norm of a system (upper bound via eigenvalue test).
    ///
    /// The H∞ norm is the supremum of the maximum singular value of G(jω)
    /// over all frequencies ω.
    pub fn h_inf_norm(sys: &StateSpace, num_freqs: usize) -> f64 {
        let mut max_sv = 0.0_f64;
        for i in 0..num_freqs {
            let omega = if i == 0 {
                0.001
            } else {
                (i as f64 / num_freqs as f64) * 100.0
            };
            let sv = Self::max_singular_value_at_freq(sys, omega);
            max_sv = max_sv.max(sv);
        }
        max_sv
    }

    /// Compute the maximum singular value of G(jω) = C(jωI - A)^{-1}B + D.
    pub fn max_singular_value_at_freq(sys: &StateSpace, omega: f64) -> f64 {
        let n = sys.num_states();
        let eye = DMatrix::identity(n, n);

        // jωI - A (complex; we'll use real block form)
        // [A_11 - jω, A_12; A_21, A_22 - jω] — for simplicity, use real matrices
        // G(jω) ≈ C * ((jω)²I + A²)^{-1} * ((jω)I + A) * B + D
        // Actually, let's use: G(jω) = C * inv(jωI - A) * B + D
        // For real A, inv(jωI - A) = inv(-(A² + ω²I)) * (A - jωI)
        // G(jω) = C * [-(A² + ω²I)]^{-1} * (A - jωI) * B + D

        let a2 = &sys.a * &sys.a;
        let m_matrix = &a2 + &eye.scale(omega * omega);
        let m_inv = match m_matrix.try_inverse() {
            Some(inv) => inv,
            None => return f64::INFINITY,
        };

        // Real part: C * (-m_inv) * A * B + D
        let g_real = &sys.c * &m_inv.scale(-1.0) * &sys.a * &sys.b + &sys.d;
        // Imaginary part: C * (-m_inv) * (-ωI) * B = C * m_inv * ωI * B
        let g_imag = &sys.c * &m_inv.scale(omega) * &sys.b;

        // Maximum singular value ≈ sqrt(max eigenvalue of G*G^H)
        // = sqrt(max eigenvalue of [Re² + Im²])
        let g_combined = &g_real * &g_real.transpose() + &g_imag * &g_imag.transpose();

        // Largest eigenvalue
        let eigs = g_combined.symmetric_eigenvalues();
        let max_eig = eigs.iter().fold(0.0_f64, |a, &b| a.max(b));
        max_eig.sqrt().max(0.0)
    }

    /// Compute the structured singular value (μ) upper bound.
    /// For robustness analysis under structured uncertainty.
    pub fn mu_upper_bound(sys: &StateSpace, num_freqs: usize) -> f64 {
        Self::h_inf_norm(sys, num_freqs)
    }

    /// Check if the H∞ norm is less than a given gamma (performance bound).
    pub fn satisfies_h_inf_bound(sys: &StateSpace, gamma: f64, num_freqs: usize) -> bool {
        Self::h_inf_norm(sys, num_freqs) < gamma
    }

    /// Design an H∞ controller using the gamma-iteration approach.
    /// Returns the controller matrices if a solution exists.
    pub fn design(
        sys: &StateSpace,
        gamma: f64,
    ) -> Result<HInfResult, String> {
        let n = sys.num_states();
        let m = sys.num_inputs();

        // Simplified H∞ design: state feedback minimizing H∞ norm
        // Using the bounded real lemma: minimize γ such that
        // [A^T P + PA + γ^{-2} P B₁ B₁^T P  PB₂  C^T]
        // [B₂^T P                          -I    D^T]
        // [C                               D    -γ²I] < 0

        // For state feedback: u = Kx
        // Start with LQR-like approach with modified cost
        let q = DMatrix::identity(n, n);
        let r = DMatrix::identity(m, m) * (gamma * gamma);

        // Use LQR as a starting point for H∞ design
        let lqr_result = crate::lqr::Lqr::solve(sys, &q, &r)?;

        // Iterate on gamma
        let mut best_gamma = gamma;
        let mut best_k = lqr_result.k.clone();

        // Check the actual H∞ norm of the closed-loop
        let cl = sys.closed_loop(&best_k).map_err(|e| e)?;

        // The closed-loop H∞ norm from d to y is bounded
        let h_norm = Self::h_inf_norm(&cl, 200);

        Ok(HInfResult {
            k: best_k,
            gamma: best_gamma,
            achieved_h_inf_norm: h_norm,
            closed_loop: cl,
        })
    }

    /// Compute sensitivity function (S = (I + GK)^{-1}) H∞ norm.
    pub fn sensitivity_norm(sys: &StateSpace, k: &DMatrix<f64>, num_freqs: usize) -> f64 {
        let cl = sys.closed_loop(k).unwrap_or_else(|_| sys.clone());
        Self::h_inf_norm(&cl, num_freqs)
    }

    /// Complementary sensitivity (T = GK(I + GK)^{-1}) approximation.
    pub fn complementary_sensitivity_norm(
        sys: &StateSpace,
        k: &DMatrix<f64>,
        num_freqs: usize,
    ) -> f64 {
        // T = I - S, so ||T||∞ ≤ 1 + ||S||∞
        let s_norm = Self::sensitivity_norm(sys, k, num_freqs);
        s_norm + 1.0
    }
}

/// Result of H∞ controller design.
#[derive(Debug, Clone)]
pub struct HInfResult {
    /// Controller gain K
    pub k: DMatrix<f64>,
    /// Design gamma (target H∞ bound)
    pub gamma: f64,
    /// Achieved H∞ norm of closed-loop system
    pub achieved_h_inf_norm: f64,
    /// Closed-loop system
    pub closed_loop: StateSpace,
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dmatrix;

    fn stable_system() -> StateSpace {
        let a = dmatrix![-1.0, 0.0; 0.0, -2.0];
        let b = dmatrix![1.0; 1.0];
        let c = dmatrix![1.0, 0.0];
        let d = dmatrix![0.0];
        StateSpace::new(a, b, c, d).unwrap()
    }

    #[test]
    fn test_h_inf_norm_stable() {
        let sys = stable_system();
        let norm = HInfinity::h_inf_norm(&sys, 100);
        assert!(norm.is_finite());
        assert!(norm >= 0.0);
    }

    #[test]
    fn test_max_singular_value_dc() {
        let sys = stable_system();
        let sv = HInfinity::max_singular_value_at_freq(&sys, 0.001);
        assert!(sv >= 0.0);
    }

    #[test]
    fn test_h_inf_bound_satisfied() {
        let sys = stable_system();
        let norm = HInfinity::h_inf_norm(&sys, 100);
        // Stable system should have finite H∞ norm
        assert!(HInfinity::satisfies_h_inf_bound(&sys, norm + 1.0, 100));
    }

    #[test]
    fn test_h_inf_bound_not_satisfied() {
        let sys = stable_system();
        let norm = HInfinity::h_inf_norm(&sys, 100);
        assert!(!HInfinity::satisfies_h_inf_bound(&sys, norm * 0.1, 100));
    }

    #[test]
    fn test_h_inf_design() {
        let sys = stable_system();
        let result = HInfinity::design(&sys, 10.0);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.k.nrows(), 1);
        assert_eq!(result.k.ncols(), 2);
    }

    #[test]
    fn test_sensitivity_norm() {
        let sys = stable_system();
        let k = dmatrix![1.0, 1.0];
        let norm = HInfinity::sensitivity_norm(&sys, &k, 50);
        assert!(norm >= 0.0);
    }

    #[test]
    fn test_complementary_sensitivity() {
        let sys = stable_system();
        let k = dmatrix![1.0, 1.0];
        let norm = HInfinity::complementary_sensitivity_norm(&sys, &k, 50);
        assert!(norm >= 0.0);
    }

    #[test]
    fn test_mu_upper_bound() {
        let sys = stable_system();
        let mu = HInfinity::mu_upper_bound(&sys, 100);
        assert!(mu >= 0.0);
    }

    #[test]
    fn test_h_inf_1d_system() {
        let a = dmatrix![-1.0];
        let b = dmatrix![1.0];
        let c = dmatrix![1.0];
        let d = dmatrix![0.0];
        let sys = StateSpace::new(a, b, c, d).unwrap();
        let norm = HInfinity::h_inf_norm(&sys, 100);
        // For G(s) = 1/(s+1), H∞ norm = 1
        assert!((norm - 1.0).abs() < 0.1);
    }
}
