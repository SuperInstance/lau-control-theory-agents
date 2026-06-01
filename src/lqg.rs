//! Linear Quadratic Gaussian (LQG): optimal control + Kalman filtering

use nalgebra::DMatrix;
use crate::state_space::StateSpace;
use crate::lqr::Lqr;

/// LQG controller combining LQR with a Kalman filter.
///
/// When the state is not directly measurable, LQG uses a Kalman filter
/// to estimate the state from noisy outputs, then applies LQR feedback.
pub struct Lqg;

impl Lqg {
    /// Design an LQG controller.
    ///
    /// - `q_lqr`: State cost matrix for LQR
    /// - `r_lqr`: Control cost matrix for LQR
    /// - `v`: Process noise covariance (n × n)
    /// - `w`: Measurement noise covariance (p × p)
    pub fn design(
        sys: &StateSpace,
        q_lqr: &DMatrix<f64>,
        r_lqr: &DMatrix<f64>,
        v: &DMatrix<f64>,
        w: &DMatrix<f64>,
    ) -> Result<LqgResult, String> {
        let n = sys.num_states();
        let m = sys.num_inputs();
        let p = sys.num_outputs();

        if v.nrows() != n || v.ncols() != n {
            return Err("V must be n × n".into());
        }
        if w.nrows() != p || w.ncols() != p {
            return Err("W must be p × p".into());
        }

        // Step 1: Design LQR controller
        let lqr_result = Lqr::solve(sys, q_lqr, r_lqr)?;

        // Step 2: Design Kalman filter (dual of LQR)
        let kalman_result = Self::design_kalman_filter(sys, v, w)?;

        // Combined controller: u = -K x_hat
        // Observer: ẋ_hat = A x_hat + B u + L(y - C x_hat - D u)
        let l = &kalman_result.l;
        let k = &lqr_result.k;

        // Observer state matrix
        let a_obs = &sys.a - l * &sys.c - (&sys.b - l * &sys.d) * k;

        Ok(LqgResult {
            k: k.clone(),
            l: l.clone(),
            p_riccati: lqr_result.p.clone(),
            s_riccati: kalman_result.s.clone(),
            a_obs,
            lqr_result,
            kalman_result,
        })
    }

    /// Design a continuous-time Kalman filter.
    ///
    /// Solves the Filter Algebraic Riccati Equation (FARE):
    /// A S + S A^T - S C^T W^{-1} C S + V = 0
    ///
    /// Returns the Kalman gain L = S C^T W^{-1}
    fn design_kalman_filter(
        sys: &StateSpace,
        v: &DMatrix<f64>,
        w: &DMatrix<f64>,
    ) -> Result<KalmanResult, String> {
        let w_inv = w.clone().try_inverse().ok_or("W must be invertible")?;

        // Solve the dual problem: use LQR solver with
        // A -> A^T, B -> C^T, Q -> V, R -> W
        let dual_sys = StateSpace::new(
            sys.a.clone().transpose(),
            sys.c.clone().transpose(),
            sys.b.clone().transpose(),
            sys.d.clone().transpose(),
        ).map_err(|e| e)?;

        let dual_lqr = Lqr::solve(&dual_sys, v, w)?;

        // Kalman gain: L = S C^T W^{-1} = (P_dual)^T C^T W^{-1}
        // Since P_dual is symmetric, L = P_dual C^T W^{-1}
        let s = dual_lqr.p;
        let l = &s * &sys.c.transpose() * &w_inv;

        Ok(KalmanResult {
            l,
            s,
            w_inv,
        })
    }

    /// Simulate the LQG-controlled system.
    pub fn simulate(
        sys: &StateSpace,
        lqg: &LqgResult,
        x0: &nalgebra::DVector<f64>,
        x_hat0: &nalgebra::DVector<f64>,
        dt: f64,
        steps: usize,
    ) -> LqgSimulation {
        let n = sys.num_states();
        let m = sys.num_inputs();
        let p = sys.num_outputs();

        let mut x = x0.clone();
        let mut x_hat = x_hat0.clone();
        let mut states = Vec::with_capacity(steps + 1);
        let mut estimates = Vec::with_capacity(steps + 1);
        let mut outputs = Vec::with_capacity(steps + 1);
        let mut controls = Vec::with_capacity(steps + 1);

        states.push(x.clone());
        estimates.push(x_hat.clone());
        outputs.push(sys.output(&x, &nalgebra::DVector::zeros(m)));
        controls.push(nalgebra::DVector::zeros(m));

        let l = &lqg.l;
        let k = &lqg.k;

        for _ in 0..steps {
            // Output
            let u_prev = &controls.last().unwrap();
            let y = sys.output(&x, u_prev);

            // Innovation
            let y_hat = sys.output(&x_hat, u_prev);
            let innovation = &y - &y_hat;

            // Control: u = -K x_hat
            let u = -k * &x_hat;

            // State update (Euler)
            let dx = sys.state_derivative(&x, &u);
            x = &x + dx.scale(dt);

            // Observer update
            let dx_hat = &sys.a * &x_hat + &sys.b * &u + l * innovation;
            x_hat = &x_hat + dx_hat.scale(dt);

            states.push(x.clone());
            estimates.push(x_hat.clone());
            outputs.push(y);
            controls.push(u);
        }

        LqgSimulation {
            states,
            estimates,
            outputs,
            controls,
        }
    }
}

/// Result of Kalman filter design.
#[derive(Debug, Clone)]
pub struct KalmanResult {
    /// Kalman gain L (n × p)
    pub l: DMatrix<f64>,
    /// Error covariance S (n × n)
    pub s: DMatrix<f64>,
    /// Inverse of W
    pub w_inv: DMatrix<f64>,
}

/// Result of LQG design.
#[derive(Debug, Clone)]
pub struct LqgResult {
    /// LQR gain K (m × n)
    pub k: DMatrix<f64>,
    /// Kalman gain L (n × p)
    pub l: DMatrix<f64>,
    /// LQR Riccati solution P
    pub p_riccati: DMatrix<f64>,
    /// Kalman Riccati solution S
    pub s_riccati: DMatrix<f64>,
    /// Observer state matrix
    pub a_obs: DMatrix<f64>,
    /// Full LQR result
    pub lqr_result: crate::lqr::LqrResult,
    /// Full Kalman result
    pub kalman_result: KalmanResult,
}

/// Simulation result for LQG system.
#[derive(Debug, Clone)]
pub struct LqgSimulation {
    pub states: Vec<nalgebra::DVector<f64>>,
    pub estimates: Vec<nalgebra::DVector<f64>>,
    pub outputs: Vec<nalgebra::DVector<f64>>,
    pub controls: Vec<nalgebra::DVector<f64>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dmatrix;

    fn simple_system() -> StateSpace {
        let a = dmatrix![-1.0, 0.0; 0.0, -2.0];
        let b = dmatrix![1.0; 1.0];
        let c = dmatrix![1.0, 0.0];
        let d = dmatrix![0.0];
        StateSpace::new(a, b, c, d).unwrap()
    }

    #[test]
    fn test_lqg_design() {
        let sys = simple_system();
        let q = DMatrix::identity(2, 2);
        let r = dmatrix![1.0];
        let v = DMatrix::identity(2, 2) * 0.1;
        let w = dmatrix![1.0];
        let result = Lqg::design(&sys, &q, &r, &v, &w);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.k.nrows(), 1);
        assert_eq!(result.k.ncols(), 2);
        assert_eq!(result.l.nrows(), 2);
        assert_eq!(result.l.ncols(), 1);
    }

    #[test]
    fn test_kalman_gain_reasonable() {
        let sys = simple_system();
        let q = DMatrix::identity(2, 2);
        let r = dmatrix![1.0];
        let v = DMatrix::identity(2, 2) * 0.01; // Low process noise
        let w = dmatrix![100.0]; // High measurement noise
        let result = Lqg::design(&sys, &q, &r, &v, &w).unwrap();
        // With high measurement noise, Kalman gain should be small
        assert!(result.l.norm() < 1.0);
    }

    #[test]
    fn test_lqg_simulation() {
        let sys = simple_system();
        let q = DMatrix::identity(2, 2);
        let r = dmatrix![1.0];
        let v = DMatrix::identity(2, 2) * 0.01;
        let w = dmatrix![1.0];
        let lqg = Lqg::design(&sys, &q, &r, &v, &w).unwrap();

        let x0 = nalgebra::dvector![1.0, 1.0];
        let x_hat0 = nalgebra::dvector![0.0, 0.0];

        let sim = Lqg::simulate(&sys, &lqg, &x0, &x_hat0, 0.01, 100);
        assert_eq!(sim.states.len(), 101);
        assert_eq!(sim.estimates.len(), 101);
        assert_eq!(sim.controls.len(), 101);
    }

    #[test]
    fn test_lqg_estimate_converges() {
        let sys = simple_system();
        let q = DMatrix::identity(2, 2);
        let r = dmatrix![1.0];
        let v = DMatrix::identity(2, 2) * 0.01;
        let w = dmatrix![0.01]; // Low measurement noise
        let lqg = Lqg::design(&sys, &q, &r, &v, &w).unwrap();

        let x0 = nalgebra::dvector![1.0, 0.0];
        let x_hat0 = nalgebra::dvector![0.0, 0.0];

        let sim = Lqg::simulate(&sys, &lqg, &x0, &x_hat0, 0.01, 200);

        // Error should decrease over time
        let initial_error = (&sim.states[0] - &sim.estimates[0]).norm();
        let final_error = (&sim.states[100] - &sim.estimates[100]).norm();
        assert!(final_error < initial_error);
    }

    #[test]
    fn test_lqg_stabilizes_system() {
        let sys = simple_system();
        let q = DMatrix::identity(2, 2);
        let r = dmatrix![1.0];
        let v = DMatrix::identity(2, 2) * 0.01;
        let w = dmatrix![1.0];
        let lqg = Lqg::design(&sys, &q, &r, &v, &w).unwrap();

        let x0 = nalgebra::dvector![10.0, 10.0];
        let x_hat0 = nalgebra::dvector![0.0, 0.0];

        let sim = Lqg::simulate(&sys, &lqg, &x0, &x_hat0, 0.01, 500);

        // State should converge toward zero
        let final_state_norm = sim.states.last().unwrap().norm();
        assert!(final_state_norm < x0.norm());
    }

    #[test]
    fn test_observer_matrix_shape() {
        let sys = simple_system();
        let q = DMatrix::identity(2, 2);
        let r = dmatrix![1.0];
        let v = DMatrix::identity(2, 2) * 0.1;
        let w = dmatrix![1.0];
        let lqg = Lqg::design(&sys, &q, &r, &v, &w).unwrap();
        assert_eq!(lqg.a_obs.nrows(), 2);
        assert_eq!(lqg.a_obs.ncols(), 2);
    }
}
