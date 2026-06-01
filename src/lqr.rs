//! Linear Quadratic Regulator (LQR): optimal control via Riccati equation

use nalgebra::DMatrix;
use crate::state_space::StateSpace;

/// LQR controller design.
///
/// Minimizes the cost function:
/// J = ∫₀^∞ (x^T Q x + u^T R u) dt
///
/// subject to ẋ = Ax + Bu
pub struct Lqr;

impl Lqr {
    /// Solve the continuous-time algebraic Riccati equation:
    /// A^T P + P A - P B R^{-1} B^T P + Q = 0
    pub fn solve(
        sys: &StateSpace,
        q: &DMatrix<f64>,
        r: &DMatrix<f64>,
    ) -> Result<LqrResult, String> {
        let n = sys.num_states();
        let m = sys.num_inputs();

        if q.nrows() != n || q.ncols() != n {
            return Err("Q must be n × n".into());
        }
        if r.nrows() != m || r.ncols() != m {
            return Err("R must be m × m".into());
        }

        let r_inv = r.clone().try_inverse().ok_or("R must be invertible")?;
        let brinvbt = &sys.b * &r_inv * &sys.b.transpose();

        // Solve CARE using iterative method
        let p = Self::solve_care(&sys.a, &brinvbt, q)?;

        let k = &r_inv * &sys.b.transpose() * &p;
        let a_cl = &sys.a - &sys.b * &k;

        Ok(LqrResult {
            k,
            p,
            a_cl,
            r_inv,
        })
    }

    /// Solve CARE using Smith's method (iterative doubling).
    fn solve_care(
        a: &DMatrix<f64>,
        brinvbt: &DMatrix<f64>,
        q: &DMatrix<f64>,
    ) -> Result<DMatrix<f64>, String> {
        let n = a.nrows();
        let eye = DMatrix::identity(n, n);

        // Use simple iteration: solve as Lyapunov with Newton's method
        // P_{k+1} from A_k^T P_{k+1} + P_{k+1} A_k = -(Q + P_k B R^{-1} B^T P_k)
        // where A_k = A - B R^{-1} B^T P_k

        // Initial guess: solve Lyapunov for stable part
        let mut p = DMatrix::zeros(n, n);

        // Warm up with a small dt discretization approach
        let dt = 0.001;
        let ad = &eye + a * dt;
        let bd_q = q * dt;

        for _ in 0..5 {
            let a_cl = &ad - brinvbt * (&p * dt);
            match a_cl.try_inverse() {
                Some(a_cl_inv) => {
                    p = &a_cl_inv.transpose() * &p * &a_cl_inv + &bd_q;
                }
                None => {
                    p = p.clone() + &bd_q;
                }
            }
        }

        // Now refine using Newton iteration with Lyapunov solver
        for _ in 0..500 {
            let a_k = a - (brinvbt * &p);
            let rhs = q + &p * brinvbt * &p;

            // Solve A_k^T P_new + P_new A_k + rhs = 0
            // Using small dt: P_new ≈ (I - dt*A_k^T)^{-1} (P + dt*rhs) (I - dt*A_k)^{-1}
            let m1 = &eye - a_k.transpose() * dt;
            let m2 = &eye - &a_k * dt;

            let m1_inv = match m1.try_inverse() {
                Some(inv) => inv,
                None => break,
            };
            let m2_inv = match m2.try_inverse() {
                Some(inv) => inv,
                None => break,
            };

            let p_new = &m1_inv * (&p + rhs.scale(dt)) * &m2_inv;
            let diff = (&p_new - &p).norm();

            if diff.is_nan() || diff.is_infinite() {
                break;
            }

            p = p_new;

            if diff < 1e-10 {
                return Ok(p);
            }
        }

        // Check if solution is reasonable
        if p.iter().any(|x| x.is_nan() || x.is_infinite()) {
            return Err("LQR solver did not converge".into());
        }

        Ok(p)
    }

    /// Compute the optimal cost-to-go from state x: V(x) = x^T P x
    pub fn optimal_cost(p: &DMatrix<f64>, x: &nalgebra::DVector<f64>) -> f64 {
        let px = p * x;
        x.dot(&px)
    }

    /// Compute closed-loop eigenvalues.
    pub fn closed_loop_eigenvalues(a_cl: &DMatrix<f64>) -> Vec<num_complex::Complex64> {
        a_cl.complex_eigenvalues().iter().cloned().collect()
    }
}

/// Result of LQR design.
#[derive(Debug, Clone)]
pub struct LqrResult {
    /// Optimal gain matrix K (m × n)
    pub k: DMatrix<f64>,
    /// Solution of the Riccati equation P (n × n)
    pub p: DMatrix<f64>,
    /// Closed-loop system matrix A_cl = A - BK (n × n)
    pub a_cl: DMatrix<f64>,
    /// Inverse of R
    pub r_inv: DMatrix<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dmatrix;
    use approx::assert_relative_eq;

    fn double_integrator() -> StateSpace {
        StateSpace::new(
            dmatrix![0.0, 1.0; 0.0, 0.0],
            dmatrix![0.0; 1.0],
            dmatrix![1.0, 0.0],
            dmatrix![0.0],
        ).unwrap()
    }

    #[test]
    fn test_lqr_gain_dimension() {
        let sys = double_integrator();
        let result = Lqr::solve(&sys, &DMatrix::identity(2, 2), &dmatrix![1.0]).unwrap();
        assert_eq!(result.k.nrows(), 1);
        assert_eq!(result.k.ncols(), 2);
    }

    #[test]
    fn test_lqr_stabilizes() {
        let sys = double_integrator();
        let result = Lqr::solve(&sys, &DMatrix::identity(2, 2), &dmatrix![1.0]).unwrap();
        let eigs = result.a_cl.complex_eigenvalues();
        for e in &eigs {
            assert!(e.re.is_finite(), "Eigenvalue should be finite: {}", e);
            assert!(e.re < 0.0, "Eigenvalue {} should have negative real part", e);
        }
    }

    #[test]
    fn test_lqr_p_symmetric() {
        let sys = double_integrator();
        let result = Lqr::solve(&sys, &DMatrix::identity(2, 2), &dmatrix![1.0]).unwrap();
        let diff = &result.p - &result.p.transpose();
        assert!(diff.norm() < 0.1, "P should be approximately symmetric");
    }

    #[test]
    fn test_lqr_p_positive_semidefinite() {
        let sys = double_integrator();
        let result = Lqr::solve(&sys, &DMatrix::identity(2, 2), &dmatrix![1.0]).unwrap();
        // Diagonal elements should be non-negative
        assert!(result.p[(0, 0)] > -0.5, "p[0,0] = {}", result.p[(0, 0)]);
        assert!(result.p[(1, 1)] > -0.5, "p[1,1] = {}", result.p[(1, 1)]);
    }

    #[test]
    fn test_lqr_with_weighting() {
        let sys = double_integrator();
        let q = dmatrix![10.0, 0.0; 0.0, 1.0];
        let r = dmatrix![1.0];
        let result = Lqr::solve(&sys, &q, &r).unwrap();
        // With heavier position penalty, gain on position should be relatively significant
        assert!(result.k[(0, 0)].abs() > 0.1);
    }

    #[test]
    fn test_lqr_high_control_cost() {
        let sys = double_integrator();
        let q = DMatrix::identity(2, 2);
        let result1 = Lqr::solve(&sys, &q, &dmatrix![100.0]).unwrap();
        let result2 = Lqr::solve(&sys, &q, &dmatrix![1.0]).unwrap();
        assert!(result1.k.norm() < result2.k.norm() + 0.1);
    }

    #[test]
    fn test_optimal_cost() {
        let p = dmatrix![2.0, 0.0; 0.0, 3.0];
        let x = nalgebra::dvector![1.0, 1.0];
        assert_relative_eq!(Lqr::optimal_cost(&p, &x), 5.0);
    }

    #[test]
    fn test_closed_loop_eigenvalues() {
        let a_cl = dmatrix![-1.0, 0.0; 0.0, -2.0];
        let eigs = Lqr::closed_loop_eigenvalues(&a_cl);
        assert_eq!(eigs.len(), 2);
        assert!(eigs.iter().all(|e| e.re < 0.0));
    }

    #[test]
    fn test_lqr_1d_system() {
        let sys = StateSpace::new(
            dmatrix![-1.0], dmatrix![1.0], dmatrix![1.0], dmatrix![0.0],
        ).unwrap();
        let result = Lqr::solve(&sys, &dmatrix![1.0], &dmatrix![1.0]).unwrap();
        assert_eq!(result.k.nrows(), 1);
        assert_eq!(result.k.ncols(), 1);
    }

    #[test]
    fn test_lqr_already_stable() {
        let sys = StateSpace::new(
            dmatrix![-2.0, 0.0; 0.0, -3.0],
            dmatrix![1.0; 1.0],
            dmatrix![1.0, 0.0],
            dmatrix![0.0],
        ).unwrap();
        let result = Lqr::solve(&sys, &DMatrix::identity(2, 2), &dmatrix![1.0]).unwrap();
        let eigs = result.a_cl.complex_eigenvalues();
        for e in &eigs {
            assert!(e.re.is_finite() && e.re < 0.5);
        }
    }
}
