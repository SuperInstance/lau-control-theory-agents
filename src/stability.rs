//! Stability analysis: Lyapunov, asymptotic, exponential stability

use nalgebra::DMatrix;
use crate::state_space::StateSpace;
use crate::controllability::solve_lyapunov;

/// Stability analysis for LTI systems.
pub struct Stability;

impl Stability {
    /// Check if a matrix A has all eigenvalues with negative real parts (Hurwitz).
    pub fn is_hurwitz(a: &DMatrix<f64>) -> bool {
        let eigs = a.complex_eigenvalues();
        eigs.iter().all(|e| e.re < 0.0)
    }

    /// Check Lyapunov stability: all eigenvalues have non-positive real parts,
    /// and any on the imaginary axis have equal algebraic and geometric multiplicity.
    pub fn is_lyapunov_stable(a: &DMatrix<f64>) -> bool {
        let eigs = a.complex_eigenvalues();
        let all_nonpositive = eigs.iter().all(|e| e.re <= 1e-10);
        if !all_nonpositive {
            return false;
        }

        // Check that eigenvalues on the imaginary axis have geometric mult == algebraic mult
        let margin = 1e-8;
        let imaginary_eigs: Vec<_> = eigs.iter().filter(|e| e.re.abs() < margin).collect();
        if imaginary_eigs.is_empty() {
            return true;
        }

        // Simple check: no repeated eigenvalues on imaginary axis
        let _n = eigs.len();
        for i in 0..imaginary_eigs.len() {
            for j in (i + 1)..imaginary_eigs.len() {
                let diff = (imaginary_eigs[i] - imaginary_eigs[j]).norm();
                if diff < margin {
                    return false;
                }
            }
        }
        true
    }

    /// Check asymptotic stability: all eigenvalues have strictly negative real parts.
    pub fn is_asymptotically_stable(a: &DMatrix<f64>) -> bool {
        Self::is_hurwitz(a)
    }

    /// Check exponential stability: all eigenvalues have real parts < -α for some α > 0.
    /// Returns Some(α) if exponentially stable, None otherwise.
    pub fn exponential_stability_margin(a: &DMatrix<f64>) -> Option<f64> {
        let eigs = a.complex_eigenvalues();
        let max_real = eigs.iter().map(|e| e.re).fold(f64::NEG_INFINITY, f64::max);
        if max_real < 0.0 {
            Some(-max_real)
        } else {
            None
        }
    }

    /// Check exponential stability.
    pub fn is_exponentially_stable(a: &DMatrix<f64>) -> bool {
        Self::exponential_stability_margin(a).is_some()
    }

    /// Lyapunov equation test for stability.
    /// A is asymptotically stable iff for any Q > 0, the Lyapunov equation
    /// A^T P + P A + Q = 0 has a unique solution P > 0.
    pub fn lyapunov_test(a: &DMatrix<f64>) -> LyapunovResult {
        // Quick check: if A is not Hurwitz, it's not asymptotically stable
        if !Self::is_hurwitz(a) {
            return LyapunovResult {
                solution_exists: false,
                is_positive_definite: false,
                is_asymptotically_stable: false,
                p: DMatrix::zeros(a.nrows(), a.nrows()),
            };
        }

        let n = a.nrows();
        let q = DMatrix::identity(n, n);
        match solve_lyapunov(&a.transpose(), &q) {
            Some(p) => {
                let is_positive_definite = Self::is_positive_definite(&p);
                LyapunovResult {
                    solution_exists: true,
                    is_positive_definite,
                    is_asymptotically_stable: is_positive_definite,
                    p,
                }
            }
            None => LyapunovResult {
                solution_exists: false,
                is_positive_definite: false,
                is_asymptotically_stable: false,
                p: DMatrix::zeros(n, n),
            },
        }
    }

    /// Check if a symmetric matrix is positive definite using Cholesky.
    pub fn is_positive_definite(m: &DMatrix<f64>) -> bool {
        let sym = (m + m.transpose()).scale(0.5);
        sym.cholesky().is_some()
    }

    /// Compute the damping ratio for a given eigenvalue λ = -σ ± jω.
    /// ζ = -σ / √(σ² + ω²)
    pub fn damping_ratio(eigenvalue: num_complex::Complex64) -> f64 {
        let sigma = -eigenvalue.re;
        let omega = eigenvalue.im;
        let wn = (sigma * sigma + omega * omega).sqrt();
        if wn.abs() < 1e-15 {
            return 0.0;
        }
        sigma / wn
    }

    /// Compute natural frequency for a given eigenvalue.
    pub fn natural_frequency(eigenvalue: num_complex::Complex64) -> f64 {
        (eigenvalue.re * eigenvalue.re + eigenvalue.im * eigenvalue.im).sqrt()
    }

    /// Check stability of a system.
    pub fn system_stability(sys: &StateSpace) -> SystemStability {
        let a = &sys.a;
        SystemStability {
            is_lyapunov_stable: Self::is_lyapunov_stable(a),
            is_asymptotically_stable: Self::is_asymptotically_stable(a),
            is_exponentially_stable: Self::is_exponentially_stable(a),
            is_bibo_stable: Self::is_bibo_stable(sys),
            stability_margin: Self::exponential_stability_margin(a),
            eigenvalues: a.complex_eigenvalues().iter().cloned().collect(),
        }
    }

    /// BIBO (Bounded-Input Bounded-Output) stability check.
    /// For minimal systems, equivalent to asymptotic stability.
    pub fn is_bibo_stable(sys: &StateSpace) -> bool {
        Self::is_asymptotically_stable(&sys.a)
    }

    /// Gain margin and phase margin analysis for SISO systems.
    /// Returns (gain_margin_db, phase_margin_deg).
    pub fn stability_margins(_sys: &StateSpace) -> (f64, f64) {
        // Placeholder - would need frequency response for full implementation
        (f64::INFINITY, 180.0)
    }
}

/// Result of Lyapunov stability test.
#[derive(Debug, Clone)]
pub struct LyapunovResult {
    pub solution_exists: bool,
    pub is_positive_definite: bool,
    pub is_asymptotically_stable: bool,
    pub p: DMatrix<f64>,
}

/// Comprehensive stability report for a system.
#[derive(Debug, Clone)]
pub struct SystemStability {
    pub is_lyapunov_stable: bool,
    pub is_asymptotically_stable: bool,
    pub is_exponentially_stable: bool,
    pub is_bibo_stable: bool,
    pub stability_margin: Option<f64>,
    pub eigenvalues: Vec<num_complex::Complex64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dmatrix;
    use approx::assert_relative_eq;

    #[test]
    fn test_hurwitz_stable() {
        let a = dmatrix![-1.0, 0.0; 0.0, -2.0];
        assert!(Stability::is_hurwitz(&a));
    }

    #[test]
    fn test_hurwitz_unstable() {
        let a = dmatrix![1.0, 0.0; 0.0, -1.0];
        assert!(!Stability::is_hurwitz(&a));
    }

    #[test]
    fn test_lyapunov_stable_oscillator() {
        // Pure oscillator: eigenvalues on imaginary axis
        let a = dmatrix![0.0, 1.0; -1.0, 0.0];
        assert!(Stability::is_lyapunov_stable(&a));
    }

    #[test]
    fn test_not_lyapunov_stable() {
        let a = dmatrix![1.0, 0.0; 0.0, 2.0];
        assert!(!Stability::is_lyapunov_stable(&a));
    }

    #[test]
    fn test_asymptotically_stable() {
        let a = dmatrix![-1.0, 0.0; 0.0, -2.0];
        assert!(Stability::is_asymptotically_stable(&a));
    }

    #[test]
    fn test_not_asymptotically_stable() {
        let a = dmatrix![0.0, 1.0; -1.0, 0.0];
        assert!(!Stability::is_asymptotically_stable(&a));
    }

    #[test]
    fn test_exponential_stability_margin() {
        let a = dmatrix![-1.0, 0.0; 0.0, -3.0];
        let margin = Stability::exponential_stability_margin(&a);
        assert!(margin.is_some());
        assert_relative_eq!(margin.unwrap(), 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_not_exponentially_stable() {
        let a = dmatrix![1.0, 0.0; 0.0, -1.0];
        assert!(!Stability::is_exponentially_stable(&a));
    }

    #[test]
    fn test_lyapunov_test_stable() {
        let a = dmatrix![-2.0, 0.0; 0.0, -3.0];
        let result = Stability::lyapunov_test(&a);
        assert!(result.is_asymptotically_stable);
    }

    #[test]
    fn test_lyapunov_test_unstable() {
        let a = dmatrix![1.0, 0.0; 0.0, -1.0];
        let result = Stability::lyapunov_test(&a);
        // Unstable system - solution may not exist or not be PD
        assert!(!result.is_asymptotically_stable);
    }

    #[test]
    fn test_positive_definite() {
        let m = dmatrix![2.0, 0.0; 0.0, 3.0];
        assert!(Stability::is_positive_definite(&m));
    }

    #[test]
    fn test_not_positive_definite() {
        let m = dmatrix![-1.0, 0.0; 0.0, 2.0];
        assert!(!Stability::is_positive_definite(&m));
    }

    #[test]
    fn test_damping_ratio_overdamped() {
        let eig = num_complex::Complex64::new(-2.0, 0.0);
        let zeta = Stability::damping_ratio(eig);
        assert_relative_eq!(zeta, 1.0);
    }

    #[test]
    fn test_damping_ratio_underdamped() {
        let eig = num_complex::Complex64::new(-1.0, 1.0);
        let zeta = Stability::damping_ratio(eig);
        assert_relative_eq!(zeta, 1.0 / 2.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_natural_frequency() {
        let eig = num_complex::Complex64::new(-1.0, 1.0);
        let wn = Stability::natural_frequency(eig);
        assert_relative_eq!(wn, 2.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_system_stability() {
        let a = dmatrix![-1.0, 0.0; 0.0, -2.0];
        let b = dmatrix![0.0; 1.0];
        let c = dmatrix![1.0, 0.0];
        let d = dmatrix![0.0];
        let sys = StateSpace::new(a, b, c, d).unwrap();
        let stab = Stability::system_stability(&sys);
        assert!(stab.is_asymptotically_stable);
        assert!(stab.is_exponentially_stable);
        assert!(stab.is_bibo_stable);
    }

    #[test]
    fn test_bibo_stable() {
        let a = dmatrix![-5.0];
        let b = dmatrix![1.0];
        let c = dmatrix![1.0];
        let d = dmatrix![0.0];
        let sys = StateSpace::new(a, b, c, d).unwrap();
        assert!(Stability::is_bibo_stable(&sys));
    }
}
