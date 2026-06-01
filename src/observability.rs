//! Observability analysis: can we determine the state from outputs?

use nalgebra::DMatrix;
use crate::state_space::StateSpace;
use crate::controllability::Controllability;

/// Observability analysis for LTI systems.
pub struct Observability;

impl Observability {
    /// Build the observability matrix: O = [C; CA; CA²; ...; CA^(n-1)]
    pub fn observability_matrix(sys: &StateSpace) -> DMatrix<f64> {
        let n = sys.num_states();
        let p = sys.num_outputs();
        let mut mat = DMatrix::zeros(n * n, n);

        let mut cak = sys.c.clone();
        for i in 0..n {
            mat.view_mut((i * p, 0), (p, n)).copy_from(&cak);
            cak = &cak * &sys.a;
        }

        mat
    }

    /// Check if the system is observable (observability matrix has full column rank).
    pub fn is_observable(sys: &StateSpace) -> bool {
        let om = Self::observability_matrix(sys);
        Controllability::rank(&om) == sys.num_states()
    }

    /// Compute the observability Gramian (for stable systems):
    /// Wo = ∫₀^∞ e^(A^T t) C^T C e^(At) dt
    /// Solved via Lyapunov equation: A^T·Wo + Wo·A + C^T·C = 0
    pub fn observability_gramian(sys: &StateSpace) -> DMatrix<f64> {
        let n = sys.num_states();
        let q = &sys.c.transpose() * &sys.c;
        crate::controllability::solve_lyapunov(&sys.a.transpose(), &q)
            .unwrap_or_else(|| DMatrix::zeros(n, n))
    }

    /// Check observability using the dual system property.
    /// A system (A, B, C, D) is observable iff the dual (A^T, C^T, B^T, D^T) is controllable.
    pub fn is_observable_dual(sys: &StateSpace) -> bool {
        let dual = StateSpace::new(
            sys.a.clone().transpose(),
            sys.c.clone().transpose(),
            sys.b.clone().transpose(),
            sys.d.clone().transpose(),
        ).unwrap();
        Controllability::is_controllable(&dual)
    }

    /// Return the observability index.
    pub fn observability_index(sys: &StateSpace) -> usize {
        let n = sys.num_states();
        let _p = sys.num_outputs();
        let mut mat = DMatrix::zeros(0, n);
        let mut cak = sys.c.clone();
        for i in 0..n {
            let new_rows = DMatrix::from_rows(cak.row_iter().collect::<Vec<_>>().as_slice());
            mat = DMatrix::from_rows(
                mat.row_iter().chain(new_rows.row_iter()).collect::<Vec<_>>().as_slice()
            );
            if Controllability::rank(&mat) == n {
                return i + 1;
            }
            cak = &cak * &sys.a;
        }
        n
    }

    /// Detect unobservable modes via Popov-Belevitch-Hautus (PBH) test.
    /// Returns eigenvalues of A that are unobservable.
    pub fn unobservable_modes(sys: &StateSpace) -> Vec<num_complex::Complex64> {
        let eigs = sys.eigenvalues();
        let mut unobs = Vec::new();
        for &lambda in &eigs {
            // Check rank of [A - λI; C]
            let n = sys.num_states();
            let lambda_r = lambda.re;
            let eye = DMatrix::identity(n, n);
            let a_minus_lambda = &sys.a - &eye.scale(lambda_r);
            let combined = DMatrix::from_rows(
                a_minus_lambda.row_iter()
                    .chain(sys.c.row_iter())
                    .collect::<Vec<_>>()
                    .as_slice()
            );
            if Controllability::rank(&combined) < n {
                unobs.push(lambda);
            }
        }
        unobs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dmatrix;
    use approx::assert_relative_eq;

    fn observable_system() -> StateSpace {
        let a = dmatrix![0.0, 1.0; -2.0, -3.0];
        let b = dmatrix![0.0; 1.0];
        let c = dmatrix![1.0, 0.0];
        let d = dmatrix![0.0];
        StateSpace::new(a, b, c, d).unwrap()
    }

    fn unobservable_system() -> StateSpace {
        // Second state doesn't appear in output and doesn't couple to first
        let a = dmatrix![-1.0, 0.0; 0.0, -2.0];
        let b = dmatrix![0.0; 1.0];
        let c = dmatrix![1.0, 0.0]; // Only sees first state
        let d = dmatrix![0.0];
        StateSpace::new(a, b, c, d).unwrap()
    }

    #[test]
    fn test_observable_system() {
        let sys = observable_system();
        assert!(Observability::is_observable(&sys));
    }

    #[test]
    fn test_unobservable_system() {
        let sys = unobservable_system();
        assert!(!Observability::is_observable(&sys));
    }

    #[test]
    fn test_observability_matrix_size() {
        let sys = observable_system();
        let om = Observability::observability_matrix(&sys);
        assert_eq!(om.nrows(), 4);
        assert_eq!(om.ncols(), 2);
    }

    #[test]
    fn test_observability_matrix_values() {
        let sys = observable_system();
        let om = Observability::observability_matrix(&sys);
        // Row 0: C = [1, 0]
        assert_relative_eq!(om[(0, 0)], 1.0);
        assert_relative_eq!(om[(0, 1)], 0.0);
        // Row 1: CA = [0, 1]
        assert_relative_eq!(om[(1, 0)], 0.0);
        assert_relative_eq!(om[(1, 1)], 1.0);
    }

    #[test]
    fn test_observability_dual() {
        let sys = observable_system();
        assert!(Observability::is_observable_dual(&sys));
    }

    #[test]
    fn test_unobservable_dual() {
        let sys = unobservable_system();
        assert!(!Observability::is_observable_dual(&sys));
    }

    #[test]
    fn test_observability_index() {
        let sys = observable_system();
        let idx = Observability::observability_index(&sys);
        assert_eq!(idx, 2);
    }

    #[test]
    fn test_unobservable_modes_observable() {
        let sys = observable_system();
        let modes = Observability::unobservable_modes(&sys);
        assert!(modes.is_empty());
    }

    #[test]
    fn test_unobservable_modes_unobservable() {
        let sys = unobservable_system();
        let modes = Observability::unobservable_modes(&sys);
        assert!(!modes.is_empty());
    }

    #[test]
    fn test_observability_gramian() {
        let sys = observable_system();
        let wo = Observability::observability_gramian(&sys);
        assert_eq!(wo.nrows(), 2);
        assert_eq!(wo.ncols(), 2);
        // Should be positive definite for stable observable system
        assert!(wo[(0, 0)] > 0.0);
        assert!(wo[(1, 1)] > 0.0);
    }
}
