//! State-space representation of agent dynamics: ẋ = Ax + Bu, y = Cx + Du

use nalgebra::{DMatrix, DVector, ComplexField};
use serde::{Deserialize, Serialize};

/// State-space representation of a linear time-invariant (LTI) system.
///
/// Models agent dynamics as:
/// - State equation: ẋ(t) = A·x(t) + B·u(t)
/// - Output equation: y(t) = C·x(t) + D·u(t)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSpace {
    /// State matrix A (n × n)
    pub a: DMatrix<f64>,
    /// Input matrix B (n × m)
    pub b: DMatrix<f64>,
    /// Output matrix C (p × n)
    pub c: DMatrix<f64>,
    /// Feedthrough matrix D (p × m)
    pub d: DMatrix<f64>,
}

impl StateSpace {
    /// Create a new state-space model from matrices A, B, C, D.
    pub fn new(
        a: DMatrix<f64>,
        b: DMatrix<f64>,
        c: DMatrix<f64>,
        d: DMatrix<f64>,
    ) -> Result<Self, String> {
        let n = a.nrows();
        let m = b.ncols();
        let p = c.nrows();

        if a.ncols() != n {
            return Err("A must be square (n × n)".into());
        }
        if b.nrows() != n {
            return Err("B must have n rows".into());
        }
        if c.ncols() != n {
            return Err("C must have n columns".into());
        }
        if d.nrows() != p || d.ncols() != m {
            return Err("D must be p × m".into());
        }

        Ok(Self { a, b, c, d })
    }

    /// Number of states.
    pub fn num_states(&self) -> usize {
        self.a.nrows()
    }

    /// Number of inputs.
    pub fn num_inputs(&self) -> usize {
        self.b.ncols()
    }

    /// Number of outputs.
    pub fn num_outputs(&self) -> usize {
        self.c.nrows()
    }

    /// Compute the state derivative: ẋ = Ax + Bu
    pub fn state_derivative(&self, x: &DVector<f64>, u: &DVector<f64>) -> DVector<f64> {
        &self.a * x + &self.b * u
    }

    /// Compute the output: y = Cx + Du
    pub fn output(&self, x: &DVector<f64>, u: &DVector<f64>) -> DVector<f64> {
        &self.c * x + &self.d * u
    }

    /// Euler integration step.
    pub fn euler_step(&self, x: &DVector<f64>, u: &DVector<f64>, dt: f64) -> DVector<f64> {
        let dx = self.state_derivative(x, u);
        x + dx.scale(dt)
    }

    /// Runge-Kutta 4th order integration step.
    pub fn rk4_step(&self, x: &DVector<f64>, u: &DVector<f64>, dt: f64) -> DVector<f64> {
        let k1 = self.state_derivative(x, u);
        let k2 = self.state_derivative(&(x + k1.scale(dt / 2.0)), u);
        let k3 = self.state_derivative(&(x + k2.scale(dt / 2.0)), u);
        let k4 = self.state_derivative(&(x + k3.scale(dt)), u);
        x + (k1 + k2.scale(2.0) + k3.scale(2.0) + k4).scale(dt / 6.0)
    }

    /// Simulate the system for `steps` time steps using Euler integration.
    pub fn simulate_euler(
        &self,
        x0: &DVector<f64>,
        inputs: &[DVector<f64>],
        dt: f64,
    ) -> Vec<DVector<f64>> {
        let mut states = Vec::with_capacity(inputs.len() + 1);
        states.push(x0.clone());
        let mut x = x0.clone();
        for u in inputs {
            x = self.euler_step(&x, u, dt);
            states.push(x.clone());
        }
        states
    }

    /// Simulate the system for `steps` time steps using RK4 integration.
    pub fn simulate_rk4(
        &self,
        x0: &DVector<f64>,
        inputs: &[DVector<f64>],
        dt: f64,
    ) -> Vec<DVector<f64>> {
        let mut states = Vec::with_capacity(inputs.len() + 1);
        states.push(x0.clone());
        let mut x = x0.clone();
        for u in inputs {
            x = self.rk4_step(&x, u, dt);
            states.push(x.clone());
        }
        states
    }

    /// Compute eigenvalues of the A matrix.
    pub fn eigenvalues(&self) -> Vec<num_complex::Complex64> {
        self.a.complex_eigenvalues().iter().cloned().collect()
    }

    /// Matrix exponential e^(At) using Padé approximation (scaling and squaring).
    pub fn matrix_exp_at(&self, t: f64) -> DMatrix<f64> {
        let at = &self.a * t;
        matrix_exp(&at)
    }

    /// Compute the discrete-time state-space representation via zero-order hold.
    /// Ad = e^(A·Ts), Bd = ∫₀^Ts e^(A·τ) dτ · B
    pub fn discretize(&self, ts: f64) -> StateSpace {
        let n = self.num_states();
        let ad = self.matrix_exp_at(ts);

        // Approximate Bd using Taylor series: A^{-1}(Ad - I)B
        // Or via series: Bd = Σ (A^k * ts^(k+1) / (k+1)!) * B
        let mut bd_term = DMatrix::identity(n, n);
        let mut bd = DMatrix::zeros(n, n);
        for k in 0..20 {
            let coeff = ts.powi(k as i32 + 1) / Self::factorial(k + 1);
            bd += bd_term.clone() * coeff;
            bd_term = &bd_term * &self.a;
        }
        bd = bd * &self.b;

        StateSpace {
            a: ad,
            b: bd,
            c: self.c.clone(),
            d: self.d.clone(),
        }
    }

    fn factorial(n: usize) -> f64 {
        let mut f = 1.0_f64;
        for i in 2..=n {
            f *= i as f64;
        }
        f
    }

    /// Closed-loop system with state feedback u = -Kx.
    /// Returns new system: ẋ = (A - BK)x, y = Cx (D=0).
    pub fn closed_loop(&self, k: &DMatrix<f64>) -> Result<StateSpace, String> {
        if k.nrows() != self.num_inputs() || k.ncols() != self.num_states() {
            return Err("K must be m × n".into());
        }
        let a_cl = &self.a - &self.b * k;
        let d_cl = DMatrix::zeros(self.num_outputs(), self.num_inputs());
        Ok(StateSpace {
            a: a_cl,
            b: DMatrix::zeros(self.num_states(), self.num_inputs()),
            c: self.c.clone(),
            d: d_cl,
        })
    }
}

/// Compute matrix exponential using scaling and squaring with Padé approximation.
pub fn matrix_exp(a: &DMatrix<f64>) -> DMatrix<f64> {
    let n = a.nrows();
    if a.iter().all(|x| *x == 0.0) {
        return DMatrix::identity(n, n);
    }

    // Scale A so that ||A / 2^s|| < 0.5
    let mut s = 0usize;
    let norm = a.norm();
    while norm / 2.0_f64.powi(s as i32) > 0.5 {
        s += 1;
    }

    let scale = 2.0_f64.powi(-(s as i32));
    let mut ascaled = a.scale(scale);

    // [6/6] Padé approximant
    let i = DMatrix::identity(n, n);

    // Compute U and L for (I - A/2)^(-1) (I + A/2) style
    // Using Taylor series truncated at reasonable order
    let mut exp = i.clone();
    let mut term = i.clone();
    for k in 1..=30 {
        term = &term * &ascaled;
        let coeff = 1.0 / (1..=k).product::<u64>() as f64;
        exp += &term * coeff;
    }

    // Square back
    for _ in 0..s {
        exp = &exp * &exp;
    }

    exp
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::{dmatrix, dvector, matrix};

    #[test]
    fn test_new_valid_system() {
        let a = dmatrix![0.0, 1.0; -2.0, -3.0];
        let b = dmatrix![0.0; 1.0];
        let c = dmatrix![1.0, 0.0];
        let d = dmatrix![0.0];
        let sys = StateSpace::new(a, b, c, d);
        assert!(sys.is_ok());
        let sys = sys.unwrap();
        assert_eq!(sys.num_states(), 2);
        assert_eq!(sys.num_inputs(), 1);
        assert_eq!(sys.num_outputs(), 1);
    }

    #[test]
    fn test_new_invalid_a_not_square() {
        let a = DMatrix::zeros(2, 3);
        let b = DMatrix::zeros(2, 1);
        let c = DMatrix::zeros(1, 2);
        let d = DMatrix::zeros(1, 1);
        assert!(StateSpace::new(a, b, c, d).is_err());
    }

    #[test]
    fn test_new_invalid_b_rows() {
        let a = DMatrix::zeros(2, 2);
        let b = DMatrix::zeros(3, 1);
        let c = DMatrix::zeros(1, 2);
        let d = DMatrix::zeros(1, 1);
        assert!(StateSpace::new(a, b, c, d).is_err());
    }

    #[test]
    fn test_state_derivative() {
        let a = dmatrix![0.0, 1.0; -2.0, -3.0];
        let b = dmatrix![0.0; 1.0];
        let c = dmatrix![1.0, 0.0];
        let d = dmatrix![0.0];
        let sys = StateSpace::new(a, b, c, d).unwrap();
        let x = dvector![1.0, 0.0];
        let u = dvector![0.0];
        let dx = sys.state_derivative(&x, &u);
        assert_relative_eq!(dx[0], 0.0);
        assert_relative_eq!(dx[1], -2.0);
    }

    #[test]
    fn test_output() {
        let a = dmatrix![0.0, 1.0; -2.0, -3.0];
        let b = dmatrix![0.0; 1.0];
        let c = dmatrix![1.0, 0.0];
        let d = dmatrix![0.0];
        let sys = StateSpace::new(a, b, c, d).unwrap();
        let x = dvector![2.0, 3.0];
        let u = dvector![1.0];
        let y = sys.output(&x, &u);
        assert_relative_eq!(y[0], 2.0);
    }

    #[test]
    fn test_euler_step() {
        let a = dmatrix![0.0, 1.0; -2.0, -3.0];
        let b = dmatrix![0.0; 1.0];
        let c = dmatrix![1.0, 0.0];
        let d = dmatrix![0.0];
        let sys = StateSpace::new(a, b, c, d).unwrap();
        let x = dvector![1.0, 0.0];
        let u = dvector![0.0];
        let x_next = sys.euler_step(&x, &u, 0.01);
        assert_relative_eq!(x_next[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(x_next[1], -0.02, epsilon = 1e-6);
    }

    #[test]
    fn test_rk4_step() {
        let a = dmatrix![0.0, 1.0; -2.0, -3.0];
        let b = dmatrix![0.0; 1.0];
        let c = dmatrix![1.0, 0.0];
        let d = dmatrix![0.0];
        let sys = StateSpace::new(a, b, c, d).unwrap();
        let x = dvector![1.0, 0.0];
        let u = dvector![0.0];
        let x_next = sys.rk4_step(&x, &u, 0.01);
        // Should be close to Euler but slightly different
        assert_relative_eq!(x_next[0], 1.0, epsilon = 1e-3);
    }

    #[test]
    fn test_simulate_euler() {
        let a = dmatrix![0.0];
        let b = dmatrix![1.0];
        let c = dmatrix![1.0];
        let d = dmatrix![0.0];
        let sys = StateSpace::new(a, b, c, d).unwrap();
        let x0 = dvector![0.0];
        let inputs: Vec<_> = (0..10).map(|_| dvector![1.0]).collect();
        let states = sys.simulate_euler(&x0, &inputs, 0.1);
        assert_eq!(states.len(), 11);
        assert_relative_eq!(states[10][0], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_simulate_rk4() {
        let a = dmatrix![0.0];
        let b = dmatrix![1.0];
        let c = dmatrix![1.0];
        let d = dmatrix![0.0];
        let sys = StateSpace::new(a, b, c, d).unwrap();
        let x0 = dvector![0.0];
        let inputs: Vec<_> = (0..10).map(|_| dvector![1.0]).collect();
        let states = sys.simulate_rk4(&x0, &inputs, 0.1);
        assert_eq!(states.len(), 11);
        assert_relative_eq!(states[10][0], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_matrix_exp_zero() {
        let a = DMatrix::zeros(2, 2);
        let exp = matrix_exp(&a);
        assert_relative_eq!(exp[(0, 0)], 1.0);
        assert_relative_eq!(exp[(1, 1)], 1.0);
        assert_relative_eq!(exp[(0, 1)], 0.0);
    }

    #[test]
    fn test_matrix_exp_identity() {
        let a = dmatrix![1.0, 0.0; 0.0, 1.0];
        let exp = matrix_exp(&a);
        assert_relative_eq!(exp[(0, 0)], std::f64::consts::E, epsilon = 1e-6);
        assert_relative_eq!(exp[(1, 1)], std::f64::consts::E, epsilon = 1e-6);
    }

    #[test]
    fn test_discretize() {
        let a = dmatrix![0.0, 1.0; -2.0, -3.0];
        let b = dmatrix![0.0; 1.0];
        let c = dmatrix![1.0, 0.0];
        let d = dmatrix![0.0];
        let sys = StateSpace::new(a, b, c, d).unwrap();
        let dsys = sys.discretize(0.1);
        assert_eq!(dsys.num_states(), 2);
        assert_eq!(dsys.num_inputs(), 1);
    }

    #[test]
    fn test_closed_loop() {
        let a = dmatrix![0.0, 1.0; -2.0, -3.0];
        let b = dmatrix![0.0; 1.0];
        let c = dmatrix![1.0, 0.0];
        let d = dmatrix![0.0];
        let sys = StateSpace::new(a, b, c, d).unwrap();
        let k = dmatrix![1.0, 1.0];
        let cl = sys.closed_loop(&k).unwrap();
        // A_cl = A - BK
        assert_relative_eq!(cl.a[(0, 0)], 0.0);
        assert_relative_eq!(cl.a[(0, 1)], 1.0);
        assert_relative_eq!(cl.a[(1, 0)], -3.0);
        assert_relative_eq!(cl.a[(1, 1)], -4.0);
    }

    #[test]
    fn test_eigenvalues() {
        let a = dmatrix![-1.0, 0.0; 0.0, -2.0];
        let b = dmatrix![0.0; 1.0];
        let c = dmatrix![1.0, 0.0];
        let d = dmatrix![0.0];
        let sys = StateSpace::new(a, b, c, d).unwrap();
        let eigs = sys.eigenvalues();
        assert_eq!(eigs.len(), 2);
    }

    #[test]
    fn test_feedthrough() {
        let a = DMatrix::zeros(1, 1);
        let b = dmatrix![0.0];
        let c = dmatrix![0.0];
        let d = dmatrix![2.0];
        let sys = StateSpace::new(a, b, c, d).unwrap();
        let x = dvector![0.0];
        let u = dvector![3.0];
        let y = sys.output(&x, &u);
        assert_relative_eq!(y[0], 6.0);
    }
}
