//! Pontryagin's Maximum Principle: necessary conditions for optimality

use nalgebra::{DMatrix, DVector};
use crate::state_space::StateSpace;

/// Pontryagin's Maximum Principle solver.
///
/// For the optimal control problem:
///   minimize J = φ(x(T)) + ∫₀^T L(x(t), u(t)) dt
///   subject to: ẋ = f(x, u)
///
/// The necessary conditions are:
/// 1. State equation: ẋ = ∂H/∂λ = f(x, u*)
/// 2. Costate equation: λ̇ = -∂H/∂x
/// 3. Optimality: u* = argmin_u H(x, λ, u)
/// 4. Transversality: λ(T) = ∂φ/∂x|_{t=T}
pub struct Pontryagin;

impl Pontryagin {
    /// Solve a linear quadratic optimal control problem using Pontryagin's principle.
    ///
    /// minimizes J = ½ x(T)^T S x(T) + ½ ∫₀^T (x^T Q x + u^T R u) dt
    /// subject to ẋ = Ax + Bu
    pub fn solve_lqp(
        sys: &StateSpace,
        q: &DMatrix<f64>,
        r: &DMatrix<f64>,
        s_final: &DMatrix<f64>,
        t_final: f64,
        num_steps: usize,
    ) -> Result<PontryaginResult, String> {
        let _n = sys.num_states();
        let _m = sys.num_inputs();

        let r_inv = r.clone().try_inverse().ok_or("R must be invertible")?;

        let dt = t_final / num_steps as f64;
        // Use sub-stepping for the stiff Riccati ODE
        let sub_steps = 10;
        let dt_fine = dt / sub_steps as f64;

        // Solve using the Riccati differential equation:
        // Ṡ = -A^T S - S A + S B R^{-1} B^T S - Q
        // with S(T) = S_final
        // Integrate backwards from T to 0
        let mut s_matrices = Vec::with_capacity(num_steps + 1);
        let mut s = s_final.clone();
        s_matrices.push(s.clone());

        for _ in 0..num_steps {
            for _ in 0..sub_steps {
                // RK4 backward integration with fine step
                let ds1 = Self::riccati_derivative(&sys.a, &sys.b, q, &r_inv, &s);
                let s2 = &s - &ds1.scale(dt_fine / 2.0);
                let ds2 = Self::riccati_derivative(&sys.a, &sys.b, q, &r_inv, &s2);
                let s3 = &s - &ds2.scale(dt_fine / 2.0);
                let ds3 = Self::riccati_derivative(&sys.a, &sys.b, q, &r_inv, &s3);
                let s4 = &s - &ds3.scale(dt_fine);
                let ds4 = Self::riccati_derivative(&sys.a, &sys.b, q, &r_inv, &s4);

                s = &s - (ds1 + ds2.scale(2.0) + ds3.scale(2.0) + ds4).scale(dt_fine / 6.0);
            }
            s_matrices.push(s.clone());
        }

        // Reverse to get S(0) ... S(T)
        s_matrices.reverse();

        // Compute optimal control: u*(t) = -R^{-1} B^T S(t) x(t)
        let k_matrices: Vec<DMatrix<f64>> = s_matrices.iter()
            .map(|s_mat| &r_inv * &sys.b.transpose() * s_mat)
            .collect();

        Ok(PontryaginResult {
            s_matrices,
            k_matrices,
            dt,
            num_steps,
        })
    }

    /// Riccati differential equation derivative.
    fn riccati_derivative(
        a: &DMatrix<f64>,
        b: &DMatrix<f64>,
        q: &DMatrix<f64>,
        r_inv: &DMatrix<f64>,
        s: &DMatrix<f64>,
    ) -> DMatrix<f64> {
        let brinvbt = b * r_inv * b.transpose();
        // Ṡ = -A^T S - S A + S B R^{-1} B^T S - Q
        -a.transpose() * s - s * a + s * &brinvbt * s - q
    }

    /// Simulate the optimal trajectory using the time-varying feedback.
    pub fn simulate_optimal(
        sys: &StateSpace,
        result: &PontryaginResult,
        x0: &DVector<f64>,
    ) -> OptimalTrajectory {
        let dt = result.dt;
        let _n = sys.num_states();
        let _m = sys.num_inputs();

        let mut states = Vec::with_capacity(result.num_steps + 1);
        let mut controls = Vec::with_capacity(result.num_steps + 1);
        let mut costates = Vec::with_capacity(result.num_steps + 1);
        let mut times = Vec::with_capacity(result.num_steps + 1);

        let mut x = x0.clone();
        states.push(x.clone());
        times.push(0.0);

        for i in 0..result.num_steps {
            let k = &result.k_matrices[i];
            let u = -k * &x;
            controls.push(u.clone());

            // Costate: λ = S(t) x(t)
            let lambda = &result.s_matrices[i] * &x;
            costates.push(lambda);

            // RK4 forward integration
            let k1 = sys.state_derivative(&x, &u);
            let k2 = sys.state_derivative(&(&x + k1.scale(dt / 2.0)), &u);
            let k3 = sys.state_derivative(&(&x + k2.scale(dt / 2.0)), &u);
            let k4 = sys.state_derivative(&(&x + k3.scale(dt)), &u);
            x = &x + (k1 + k2.scale(2.0) + k3.scale(2.0) + k4).scale(dt / 6.0);

            states.push(x.clone());
            times.push((i + 1) as f64 * dt);
        }

        // Final control
        let k = &result.k_matrices[result.num_steps - 1];
        let u = -k * &x;
        controls.push(u.clone());
        let lambda = &result.s_matrices[result.num_steps - 1] * &x;
        costates.push(lambda);

        OptimalTrajectory {
            times,
            states,
            controls,
            costates,
        }
    }

    /// Compute the Hamiltonian: H = L + λ^T f
    /// For LQ: H = ½(x^T Q x + u^T R u) + λ^T(Ax + Bu)
    pub fn hamiltonian(
        x: &DVector<f64>,
        u: &DVector<f64>,
        lambda: &DVector<f64>,
        q: &DMatrix<f64>,
        r: &DMatrix<f64>,
        a: &DMatrix<f64>,
        b: &DMatrix<f64>,
    ) -> f64 {
        let lx = 0.5 * (x.transpose() * q * x)[(0, 0)];
        let lu = 0.5 * (u.transpose() * r * u)[(0, 0)];
        let dynamics = (lambda.transpose() * (a * x + b * u))[(0, 0)];
        lx + lu + dynamics
    }

    /// Verify the necessary conditions of the maximum principle.
    pub fn verify_conditions(
        trajectory: &OptimalTrajectory,
        sys: &StateSpace,
        _q: &DMatrix<f64>,
        r: &DMatrix<f64>,
        tolerance: f64,
    ) -> ConditionCheck {
        let r_inv = r.clone().try_inverse();
        let mut passed = true;
        let mut violations = Vec::new();

        // Check that Hamiltonian is minimized at each point
        if let Some(_r_inv) = r_inv {
            for i in 0..trajectory.states.len().min(trajectory.costates.len()) {
                let _x = &trajectory.states[i];
                let u = &trajectory.controls[i];
                let lambda = &trajectory.costates[i];

                // Optimal u should satisfy: ∂H/∂u = 0 → R u + B^T λ = 0
                let dh_du = r * u + sys.b.transpose() * lambda;
                if dh_du.norm() > tolerance {
                    violations.push(format!(
                        "t={:.3}: ∂H/∂u = {:.6} (should be ~0)",
                        trajectory.times[i.min(trajectory.times.len() - 1)],
                        dh_du.norm()
                    ));
                    passed = false;
                }
            }
        }

        ConditionCheck { passed, violations }
    }

    /// Solve minimum-time problem (time-optimal control).
    /// For a double integrator, this yields bang-bang control.
    pub fn solve_minimum_time(
        sys: &StateSpace,
        u_max: f64,
        x_target: &DVector<f64>,
        dt: f64,
        max_steps: usize,
    ) -> Result<MinimumTimeResult, String> {
        let n = sys.num_states();
        let mut x = DVector::zeros(n);
        let mut states = vec![x.clone()];
        let mut controls = Vec::new();
        let mut times = vec![0.0];

        // Simple approach: use bang-bang with switching based on switching curve
        for step in 0..max_steps {
            let error = x_target - &x;

            // Determine control direction using switching function
            // For LQR-like approximation of time-optimal control
            let switching = sys.b.transpose() * &error;
            let u = if switching[0] > 0.0 {
                u_max
            } else {
                -u_max
            };

            let u_vec = DVector::from_element(sys.num_inputs(), u);
            controls.push(u_vec.clone());

            let dx = sys.state_derivative(&x, &u_vec);
            x = &x + dx.scale(dt);

            states.push(x.clone());
            times.push((step + 1) as f64 * dt);

            if error.norm() < 0.01 {
                break;
            }
        }

        let t_final = times.last().copied().unwrap_or(0.0);

        Ok(MinimumTimeResult {
            states,
            controls,
            times,
            t_final,
        })
    }
}

/// Result of Pontryagin's principle solver.
#[derive(Debug, Clone)]
pub struct PontryaginResult {
    /// Time-varying Riccati solution S(t)
    pub s_matrices: Vec<DMatrix<f64>>,
    /// Time-varying gain matrices K(t) = R^{-1} B^T S(t)
    pub k_matrices: Vec<DMatrix<f64>>,
    /// Time step
    pub dt: f64,
    /// Number of time steps
    pub num_steps: usize,
}

/// Optimal trajectory from Pontryagin's principle.
#[derive(Debug, Clone)]
pub struct OptimalTrajectory {
    pub times: Vec<f64>,
    pub states: Vec<DVector<f64>>,
    pub controls: Vec<DVector<f64>>,
    pub costates: Vec<DVector<f64>>,
}

impl OptimalTrajectory {
    /// Compute total cost along trajectory.
    pub fn total_cost(&self, q: &DMatrix<f64>, r: &DMatrix<f64>, dt: f64) -> f64 {
        let mut cost = 0.0;
        for i in 0..self.states.len().min(self.controls.len()) {
            let x = &self.states[i];
            let u = &self.controls[i];
            let stage_cost = 0.5 * (x.transpose() * q * x)[(0, 0)]
                + 0.5 * (u.transpose() * r * u)[(0, 0)];
            cost += stage_cost * dt;
        }
        cost
    }

    /// Get final state.
    pub fn final_state(&self) -> &DVector<f64> {
        self.states.last().unwrap()
    }

    /// Get final control.
    pub fn final_control(&self) -> &DVector<f64> {
        self.controls.last().unwrap()
    }
}

/// Result of condition verification.
#[derive(Debug, Clone)]
pub struct ConditionCheck {
    pub passed: bool,
    pub violations: Vec<String>,
}

/// Result of minimum-time problem.
#[derive(Debug, Clone)]
pub struct MinimumTimeResult {
    pub states: Vec<DVector<f64>>,
    pub controls: Vec<DVector<f64>>,
    pub times: Vec<f64>,
    pub t_final: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dmatrix;
    use nalgebra::dvector;
    use approx::assert_relative_eq;

    fn double_integrator() -> StateSpace {
        let a = dmatrix![0.0, 1.0; 0.0, 0.0];
        let b = dmatrix![0.0; 1.0];
        let c = dmatrix![1.0, 0.0];
        let d = dmatrix![0.0];
        StateSpace::new(a, b, c, d).unwrap()
    }

    #[test]
    fn test_solve_lqp() {
        let sys = double_integrator();
        let q = DMatrix::identity(2, 2);
        let r = dmatrix![1.0];
        let s_final = DMatrix::identity(2, 2);
        let result = Pontryagin::solve_lqp(&sys, &q, &r, &s_final, 5.0, 100);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.s_matrices.len(), 101);
        assert_eq!(result.k_matrices.len(), 101);
    }

    #[test]
    fn test_simulate_optimal() {
        let sys = double_integrator();
        let q = DMatrix::identity(2, 2);
        let r = dmatrix![1.0];
        let s_final = DMatrix::identity(2, 2);
        let presult = Pontryagin::solve_lqp(&sys, &q, &r, &s_final, 5.0, 100).unwrap();
        let x0 = dvector![1.0, 0.0];
        let traj = Pontryagin::simulate_optimal(&sys, &presult, &x0);
        assert_eq!(traj.states.len(), 101);
        assert_eq!(traj.controls.len(), 101);
    }

    #[test]
    fn test_hamiltonian() {
        let x = dvector![1.0, 0.0];
        let u = dvector![0.5];
        let lambda = dvector![0.0, 1.0];
        let q = DMatrix::identity(2, 2);
        let r = dmatrix![1.0];
        let a = dmatrix![0.0, 1.0; 0.0, 0.0];
        let b = dmatrix![0.0; 1.0];
        let h = Pontryagin::hamiltonian(&x, &u, &lambda, &q, &r, &a, &b);
        assert!(h.is_finite());
    }

    #[test]
    fn test_verify_conditions() {
        let sys = double_integrator();
        let q = DMatrix::identity(2, 2);
        let r = dmatrix![1.0];
        let s_final = DMatrix::identity(2, 2);
        let presult = Pontryagin::solve_lqp(&sys, &q, &r, &s_final, 5.0, 100).unwrap();
        let x0 = dvector![1.0, 0.0];
        let traj = Pontryagin::simulate_optimal(&sys, &presult, &x0);
        let check = Pontryagin::verify_conditions(&traj, &sys, &q, &r, 0.1);
        // Should pass for LQ problem
        assert!(check.passed);
    }

    #[test]
    fn test_riccati_s_final() {
        let sys = double_integrator();
        let q = DMatrix::identity(2, 2);
        let r = dmatrix![1.0];
        let s_final = DMatrix::identity(2, 2);
        let result = Pontryagin::solve_lqp(&sys, &q, &r, &s_final, 5.0, 100).unwrap();
        // Last S matrix should equal S_final
        let s_last = result.s_matrices.last().unwrap();
        assert_relative_eq!(s_last[(0, 0)], 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_optimal_control_drives_to_zero() {
        let sys = double_integrator();
        let q = DMatrix::identity(2, 2) * 10.0;
        let r = dmatrix![1.0];
        let s_final = DMatrix::identity(2, 2) * 100.0;
        let presult = Pontryagin::solve_lqp(&sys, &q, &r, &s_final, 5.0, 200).unwrap();
        let x0 = dvector![5.0, 2.0];
        let traj = Pontryagin::simulate_optimal(&sys, &presult, &x0);
        let final_state = traj.final_state();
        // State should be driven toward zero
        assert!(final_state.norm() < x0.norm());
    }

    #[test]
    fn test_total_cost() {
        let sys = double_integrator();
        let q = DMatrix::identity(2, 2);
        let r = dmatrix![1.0];
        let s_final = DMatrix::identity(2, 2);
        let presult = Pontryagin::solve_lqp(&sys, &q, &r, &s_final, 5.0, 100).unwrap();
        let x0 = dvector![1.0, 0.0];
        let traj = Pontryagin::simulate_optimal(&sys, &presult, &x0);
        let cost = traj.total_cost(&q, &r, presult.dt);
        assert!(cost >= 0.0);
    }

    #[test]
    fn test_minimum_time() {
        let sys = double_integrator();
        let result = Pontryagin::solve_minimum_time(&sys, 1.0, &dvector![1.0, 0.0], 0.01, 1000);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.t_final > 0.0);
    }

    #[test]
    fn test_gain_shape() {
        let sys = double_integrator();
        let q = DMatrix::identity(2, 2);
        let r = dmatrix![1.0];
        let s_final = DMatrix::identity(2, 2);
        let result = Pontryagin::solve_lqp(&sys, &q, &r, &s_final, 5.0, 100).unwrap();
        for k in &result.k_matrices {
            assert_eq!(k.nrows(), 1);
            assert_eq!(k.ncols(), 2);
        }
    }

    #[test]
    fn test_costate_shape() {
        let sys = double_integrator();
        let q = DMatrix::identity(2, 2);
        let r = dmatrix![1.0];
        let s_final = DMatrix::identity(2, 2);
        let presult = Pontryagin::solve_lqp(&sys, &q, &r, &s_final, 5.0, 100).unwrap();
        let x0 = dvector![1.0, 0.0];
        let traj = Pontryagin::simulate_optimal(&sys, &presult, &x0);
        for lambda in &traj.costates {
            assert_eq!(lambda.nrows(), 2);
        }
    }
}
