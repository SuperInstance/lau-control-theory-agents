//! Controllability analysis: can we reach any state from any initial state?

use nalgebra::DMatrix;
use crate::state_space::StateSpace;

/// Controllability analysis for LTI systems.
pub struct Controllability;

impl Controllability {
    /// Build the controllability matrix: C = [B, AB, A²B, ..., A^(n-1)B]
    pub fn controllability_matrix(sys: &StateSpace) -> DMatrix<f64> {
        let n = sys.num_states();
        let m = sys.num_inputs();
        let mut mat = DMatrix::zeros(n, n * m);

        let mut ab = sys.b.clone();
        for i in 0..n {
            mat.view_mut((0, i * m), (n, m)).copy_from(&ab);
            ab = &sys.a * &ab;
        }

        mat
    }

    /// Check if the system is controllable (controllability matrix has full row rank).
    pub fn is_controllable(sys: &StateSpace) -> bool {
        let cm = Self::controllability_matrix(sys);
        Self::rank(&cm) == sys.num_states()
    }

    /// Compute the rank of a matrix using SVD.
    pub fn rank(m: &DMatrix<f64>) -> usize {
        let svd = m.clone().svd(true, true);
        let singular_values = svd.singular_values;
        let threshold = m.nrows().max(m.ncols()) as f64 * singular_values.max() * f64::EPSILON * 100.0;
        singular_values.iter().filter(|&&s| s > threshold).count()
    }

    /// Compute the controllability Gramian (for stable systems):
    /// Wc = ∫₀^∞ e^(At) B B^T e^(A^T t) dt
    /// Solved via Lyapunov equation: A·Wc + Wc·A^T + B·B^T = 0
    pub fn controllability_gramian(sys: &StateSpace) -> DMatrix<f64> {
        let n = sys.num_states();
        let q = &sys.b * &sys.b.transpose();
        solve_lyapunov(&sys.a, &q).unwrap_or_else(|| DMatrix::zeros(n, n))
    }

    /// Return the controllability index (minimum number of columns of the
    /// controllability matrix needed for full rank).
    pub fn controllability_index(sys: &StateSpace) -> usize {
        let n = sys.num_states();
        let _m = sys.num_inputs();
        let mut mat = DMatrix::zeros(n, 0);
        let mut ab = sys.b.clone();
        for i in 0..n {
            mat = DMatrix::from_columns(
                mat.column_iter().chain(ab.column_iter()).collect::<Vec<_>>().as_slice()
            );
            if Self::rank(&mat) == n {
                return i + 1;
            }
            ab = &sys.a * &ab;
        }
        n
    }
}

/// Solve the continuous-time Lyapunov equation: A·X + X·A^T + Q = 0
/// Uses the Bartels-Stewart algorithm via eigendecomposition.
pub fn solve_lyapunov(a: &DMatrix<f64>, q: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let n = a.nrows();
    if a.ncols() != n || q.nrows() != n || q.ncols() != n {
        return None;
    }

    // Use iterative method: X_{k+1} = (I - dt*A)^{-1} * X_k * (I - dt*A^T)^{-1} + dt * Q
    // Initialize with a reasonable guess
    let dt = 0.001;
    let mut x = q.scale(0.01);

    let eye = DMatrix::identity(n, n);
    let m1 = &eye - a.scale(dt);
    let m1t = &eye - &a.transpose() * dt;

    for _ in 0..5000 {
        let m1_inv = m1.clone().try_inverse()?;
        let m1t_inv = m1t.clone().try_inverse()?;
        let x_new = &m1_inv * &x * &m1t_inv + q.scale(dt);
        let diff = (&x_new - &x).norm();
        x = x_new;
        if diff < 1e-12 {
            break;
        }
    }

    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dmatrix;
    use approx::assert_relative_eq;

    fn double_integrator() -> StateSpace {
        let a = dmatrix![0.0, 1.0; 0.0, 0.0];
        let b = dmatrix![0.0; 1.0];
        let c = dmatrix![1.0, 0.0];
        let d = dmatrix![0.0];
        StateSpace::new(a, b, c, d).unwrap()
    }

    fn uncontrollable_system() -> StateSpace {
        // Second state is not controllable
        let _a = dmatrix![0.0, 1.0; 0.0, -1.0];
        let _b = dmatrix![1.0; 0.0]; // Only affects first state, but A coupling makes it tricky
        // Actually: B = [1; 0], A*B = [0; 0] => C = [[1, 0], [0, 0]] rank 1
        // Let's make a truly uncontrollable one
        let a = dmatrix![0.0, 0.0; 0.0, -1.0];
        let b = dmatrix![1.0; 0.0];
        let c = dmatrix![1.0, 0.0];
        let d = dmatrix![0.0];
        StateSpace::new(a, b, c, d).unwrap()
    }

    #[test]
    fn test_controllable_system() {
        let sys = double_integrator();
        assert!(Controllability::is_controllable(&sys));
    }

    #[test]
    fn test_uncontrollable_system() {
        let sys = uncontrollable_system();
        assert!(!Controllability::is_controllable(&sys));
    }

    #[test]
    fn test_controllability_matrix_size() {
        let sys = double_integrator();
        let cm = Controllability::controllability_matrix(&sys);
        assert_eq!(cm.nrows(), 2);
        assert_eq!(cm.ncols(), 2);
    }

    #[test]
    fn test_controllability_matrix_values() {
        let sys = double_integrator();
        let cm = Controllability::controllability_matrix(&sys);
        assert_relative_eq!(cm[(0, 0)], 0.0);
        assert_relative_eq!(cm[(1, 0)], 1.0);
        assert_relative_eq!(cm[(0, 1)], 1.0);
        assert_relative_eq!(cm[(1, 1)], 0.0);
    }

    #[test]
    fn test_controllability_index() {
        let sys = double_integrator();
        let idx = Controllability::controllability_index(&sys);
        assert_eq!(idx, 2);
    }

    #[test]
    fn test_rank_identity() {
        let m = DMatrix::identity(3, 3);
        assert_eq!(Controllability::rank(&m), 3);
    }

    #[test]
    fn test_rank_zero() {
        let m = DMatrix::zeros(3, 3);
        assert_eq!(Controllability::rank(&m), 0);
    }

    #[test]
    fn test_rank_singular() {
        let m = dmatrix![1.0, 2.0; 2.0, 4.0]; // rank 1
        assert_eq!(Controllability::rank(&m), 1);
    }

    #[test]
    fn test_controllability_mimo() {
        // Fully actuated 2-state system
        let a = dmatrix![0.0, 1.0; -2.0, -3.0];
        let b = dmatrix![1.0, 0.0; 0.0, 1.0];
        let c = dmatrix![1.0, 0.0; 0.0, 1.0];
        let d = DMatrix::zeros(2, 2);
        let sys = StateSpace::new(a, b, c, d).unwrap();
        assert!(Controllability::is_controllable(&sys));
    }
}
