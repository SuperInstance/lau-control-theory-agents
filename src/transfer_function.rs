//! Transfer functions: frequency-domain analysis (Laplace transform)

use nalgebra::DMatrix;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::fmt;

/// A transfer function represented as a ratio of polynomials in s.
///
/// H(s) = b_m s^m + ... + b_1 s + b_0
///        ----------------------------
///        a_n s^n + ... + a_1 s + a_0
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferFunction {
    /// Numerator coefficients [b_0, b_1, ..., b_m]
    pub numerator: Vec<f64>,
    /// Denominator coefficients [a_0, a_1, ..., a_n]
    pub denominator: Vec<f64>,
}

impl TransferFunction {
    /// Create a new transfer function.
    pub fn new(numerator: Vec<f64>, denominator: Vec<f64>) -> Result<Self, String> {
        if denominator.is_empty() {
            return Err("Denominator cannot be empty".into());
        }
        if denominator.iter().all(|&c| c == 0.0) {
            return Err("Denominator cannot be all zeros".into());
        }
        Ok(Self { numerator, denominator })
    }

    /// Create a gain transfer function: H(s) = K
    pub fn gain(k: f64) -> Self {
        Self {
            numerator: vec![k],
            denominator: vec![1.0],
        }
    }

    /// Create a first-order transfer function: H(s) = K / (τs + 1)
    pub fn first_order(k: f64, tau: f64) -> Result<Self, String> {
        if tau <= 0.0 {
            return Err("Time constant must be positive".into());
        }
        Ok(Self {
            numerator: vec![k],
            denominator: vec![1.0, tau],
        })
    }

    /// Create a second-order transfer function: H(s) = ω_n² / (s² + 2ζω_n s + ω_n²)
    pub fn second_order(wn: f64, zeta: f64) -> Result<Self, String> {
        if wn <= 0.0 {
            return Err("Natural frequency must be positive".into());
        }
        if zeta < 0.0 {
            return Err("Damping ratio must be non-negative".into());
        }
        Ok(Self {
            numerator: vec![wn * wn],
            denominator: vec![wn * wn, 2.0 * zeta * wn, 1.0],
        })
    }

    /// Create an integrator: H(s) = K/s
    pub fn integrator(k: f64) -> Self {
        Self {
            numerator: vec![k],
            denominator: vec![0.0, 1.0],
        }
    }

    /// Create a differentiator: H(s) = Ks
    pub fn differentiator(k: f64) -> Self {
        Self {
            numerator: vec![0.0, k],
            denominator: vec![1.0],
        }
    }

    /// Create a pure delay: H(s) = e^(-sT) approximated by Padé.
    pub fn delay(t: f64, order: usize) -> Self {
        if t <= 0.0 || order == 0 {
            return Self::gain(1.0);
        }
        // First-order Padé: (1 - sT/2) / (1 + sT/2)
        let ht = t / 2.0;
        Self {
            numerator: vec![1.0, -ht],
            denominator: vec![1.0, ht],
        }
    }

    /// Evaluate the transfer function at a complex frequency s.
    pub fn evaluate(&self, s: Complex64) -> Complex64 {
        let num = Self::eval_poly(&self.numerator, s);
        let den = Self::eval_poly(&self.denominator, s);
        if den.norm() < 1e-15 {
            Complex64::new(f64::INFINITY, 0.0)
        } else {
            num / den
        }
    }

    /// Evaluate polynomial at complex point.
    fn eval_poly(coeffs: &[f64], s: Complex64) -> Complex64 {
        let mut result = Complex64::new(0.0, 0.0);
        let mut s_power = Complex64::new(1.0, 0.0);
        for &coeff in coeffs {
            result = result + coeff * s_power;
            s_power = s_power * s;
        }
        result
    }

    /// Compute the frequency response for a range of frequencies.
    /// Returns (magnitude, phase) pairs.
    pub fn frequency_response(&self, frequencies: &[f64]) -> Vec<(f64, f64)> {
        frequencies
            .iter()
            .map(|&omega| {
                let s = Complex64::new(0.0, omega);
                let h = self.evaluate(s);
                (h.norm(), h.arg().to_degrees())
            })
            .collect()
    }

    /// Compute the Bode plot data.
    /// Returns (frequencies, magnitudes_db, phases_deg).
    pub fn bode(&self, freq_range: (f64, f64), num_points: usize) -> BodeData {
        let (f_min, f_max) = freq_range;
        let frequencies: Vec<f64> = (0..num_points)
            .map(|i| {
                let t = i as f64 / (num_points - 1) as f64;
                f_min * (f_max / f_min).powf(t)
            })
            .collect();

        let mut magnitudes_db = Vec::with_capacity(num_points);
        let mut phases_deg = Vec::with_capacity(num_points);

        for &omega in &frequencies {
            let s = Complex64::new(0.0, omega);
            let h = self.evaluate(s);
            magnitudes_db.push(20.0 * h.norm().log10());
            phases_deg.push(h.arg().to_degrees());
        }

        BodeData {
            frequencies,
            magnitudes_db,
            phases_deg,
        }
    }

    /// Compute poles (roots of denominator).
    pub fn poles(&self) -> Vec<Complex64> {
        Self::find_roots(&self.denominator)
    }

    /// Compute zeros (roots of numerator).
    pub fn zeros(&self) -> Vec<Complex64> {
        Self::find_roots(&self.numerator)
    }

    /// Find roots of a polynomial using companion matrix eigenvalues.
    fn find_roots(coeffs: &[f64]) -> Vec<Complex64> {
        // Remove trailing zeros
        let trimmed: Vec<f64> = coeffs
            .iter()
            .rev()
            .skip_while(|&&c| c == 0.0)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        if trimmed.len() <= 1 {
            return vec![];
        }

        let n = trimmed.len() - 1;
        let leading = trimmed[n];

        // Build companion matrix
        let mut companion = DMatrix::zeros(n, n);
        for i in 1..n {
            companion[(i, i - 1)] = 1.0;
        }
        for i in 0..n {
            companion[(0, i)] = -trimmed[i] / leading;
        }

        companion.complex_eigenvalues().iter().cloned().collect()
    }

    /// Check if the transfer function is stable (all poles in left half-plane).
    pub fn is_stable(&self) -> bool {
        self.poles().iter().all(|p| p.re < 0.0)
    }

    /// Check if the transfer function is minimum phase (all zeros in LHP).
    pub fn is_minimum_phase(&self) -> bool {
        self.zeros().iter().all(|p| p.re < 0.0)
    }

    /// Get the system order (degree of denominator).
    pub fn order(&self) -> usize {
        let trimmed: Vec<f64> = self.denominator.iter().rev().skip_while(|&&c| c == 0.0).cloned().collect::<Vec<_>>().into_iter().rev().collect();
        trimmed.len().saturating_sub(1)
    }

    /// Get the DC gain (H(0)).
    pub fn dc_gain(&self) -> f64 {
        self.evaluate(Complex64::new(0.0, 0.0)).re
    }

    /// Series connection: H(s) = H1(s) * H2(s)
    pub fn series(&self, other: &TransferFunction) -> TransferFunction {
        let num = Self::convolve(&self.numerator, &other.numerator);
        let den = Self::convolve(&self.denominator, &other.denominator);
        TransferFunction { numerator: num, denominator: den }
    }

    /// Parallel connection: H(s) = H1(s) + H2(s)
    pub fn parallel(&self, other: &TransferFunction) -> TransferFunction {
        let num1 = Self::convolve(&self.numerator, &other.denominator);
        let num2 = Self::convolve(&other.numerator, &self.denominator);
        let den = Self::convolve(&self.denominator, &other.denominator);
        let num: Vec<f64> = num1.iter().zip(num2.iter()).map(|(a, b)| a + b).collect();
        TransferFunction { numerator: num, denominator: den }
    }

    /// Feedback connection: H_cl(s) = H(s) / (1 + H(s))
    pub fn unity_feedback(&self) -> TransferFunction {
        // Closed-loop: num / (den + num)
        let den_closed: Vec<f64> = self.denominator.iter()
            .zip(self.numerator.iter())
            .map(|(d, n)| d + n)
            .collect();
        TransferFunction {
            numerator: self.numerator.clone(),
            denominator: Self::pad_to_length(den_closed, self.denominator.len()),
        }
    }

    /// Feedback with controller: H_cl = G*C / (1 + G*C)
    pub fn feedback_with(&self, controller: &TransferFunction) -> TransferFunction {
        let open_loop = self.series(controller);
        open_loop.unity_feedback()
    }

    /// Convolve two polynomials.
    fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; a.len() + b.len() - 1];
        for i in 0..a.len() {
            for j in 0..b.len() {
                result[i + j] += a[i] * b[j];
            }
        }
        result
    }

    /// Pad vector to given length with zeros.
    fn pad_to_length(v: Vec<f64>, len: usize) -> Vec<f64> {
        let mut result = v;
        while result.len() < len {
            result.push(0.0);
        }
        result
    }

    /// Convert to state-space using controllable canonical form.
    pub fn to_state_space(&self) -> crate::state_space::StateSpace {
        let n = self.order();
        if n == 0 {
            let k = self.dc_gain();
            return crate::state_space::StateSpace::new(
                DMatrix::zeros(1, 1),
                DMatrix::zeros(1, 1),
                DMatrix::from_element(1, 1, k),
                DMatrix::zeros(1, 1),
            ).unwrap();
        }

        let leading = self.denominator[n];
        let mut a = DMatrix::zeros(n, n);
        let mut b = DMatrix::zeros(n, 1);

        for i in 0..(n - 1) {
            a[(i + 1, i)] = 1.0;
        }
        for i in 0..n {
            a[(0, i)] = -self.denominator[i] / leading;
        }
        b[(0, 0)] = 1.0 / leading;

        // C and D from numerator
        let num_padded = Self::pad_to_length(self.numerator.clone(), n + 1);
        let mut c = DMatrix::zeros(1, n);
        for i in 0..n {
            c[(0, i)] = (num_padded.get(i).copied().unwrap_or(0.0)
                - self.denominator.get(i).copied().unwrap_or(0.0) * num_padded.get(n).copied().unwrap_or(0.0) / leading);
        }
        let d_val = num_padded.get(n).copied().unwrap_or(0.0) / leading;
        let d = DMatrix::from_element(1, 1, d_val);

        crate::state_space::StateSpace::new(a, b, c, d).unwrap()
    }

    /// Compute the step response using inverse Laplace approximation.
    /// Returns time-domain response samples.
    pub fn step_response(&self, t_end: f64, dt: f64) -> Vec<(f64, f64)> {
        let ss = self.to_state_space();
        let n_steps = (t_end / dt) as usize;
        let x0 = nalgebra::DVector::zeros(ss.num_states());
        let u = nalgebra::dvector![1.0];
        let inputs: Vec<_> = (0..n_steps).map(|_| u.clone()).collect();
        let states = ss.simulate_rk4(&x0, &inputs, dt);

        let mut result = Vec::with_capacity(n_steps + 1);
        for (i, state) in states.iter().enumerate() {
            let y = ss.output(state, &u);
            result.push((i as f64 * dt, y[0]));
        }
        result
    }

    /// Compute the impulse response approximation.
    pub fn impulse_response(&self, t_end: f64, dt: f64) -> Vec<(f64, f64)> {
        // Impulse ≈ step response derivative
        let step = self.step_response(t_end, dt);
        let mut result = Vec::with_capacity(step.len());
        result.push((0.0, 0.0));
        for i in 1..step.len() {
            let derivative = (step[i].1 - step[i - 1].1) / dt;
            result.push((step[i].0, derivative));
        }
        result
    }
}

impl fmt::Display for TransferFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "H(s) = [")?;
        for (i, &c) in self.numerator.iter().enumerate() {
            if i > 0 { write!(f, " + ")?; }
            if i == 0 { write!(f, "{:.4}", c)?; }
            else if i == 1 { write!(f, "{:.4}s", c)?; }
            else { write!(f, "{:.4}s^{}", c, i)?; }
        }
        write!(f, "] / [")?;
        for (i, &c) in self.denominator.iter().enumerate() {
            if i > 0 { write!(f, " + ")?; }
            if i == 0 { write!(f, "{:.4}", c)?; }
            else if i == 1 { write!(f, "{:.4}s", c)?; }
            else { write!(f, "{:.4}s^{}", c, i)?; }
        }
        write!(f, "]")
    }
}

/// Bode plot data.
#[derive(Debug, Clone)]
pub struct BodeData {
    pub frequencies: Vec<f64>,
    pub magnitudes_db: Vec<f64>,
    pub phases_deg: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_gain_tf() {
        let tf = TransferFunction::gain(5.0);
        assert_relative_eq!(tf.dc_gain(), 5.0);
        assert_eq!(tf.order(), 0);
    }

    #[test]
    fn test_first_order() {
        let tf = TransferFunction::first_order(1.0, 1.0).unwrap();
        assert_relative_eq!(tf.dc_gain(), 1.0);
        assert_eq!(tf.order(), 1);
    }

    #[test]
    fn test_second_order() {
        let tf = TransferFunction::second_order(1.0, 0.5).unwrap();
        assert_relative_eq!(tf.dc_gain(), 1.0);
        assert_eq!(tf.order(), 2);
    }

    #[test]
    fn test_evaluate_dc() {
        let tf = TransferFunction::first_order(2.0, 1.0).unwrap();
        let h = tf.evaluate(Complex64::new(0.0, 0.0));
        assert_relative_eq!(h.re, 2.0);
    }

    #[test]
    fn test_evaluate_high_freq() {
        let tf = TransferFunction::first_order(1.0, 1.0).unwrap();
        let h = tf.evaluate(Complex64::new(0.0, 1000.0));
        assert!(h.norm() < 0.01);
    }

    #[test]
    fn test_poles() {
        let tf = TransferFunction::first_order(1.0, 1.0).unwrap();
        let poles = tf.poles();
        assert_eq!(poles.len(), 1);
        assert_relative_eq!(poles[0].re, -1.0, epsilon = 0.01);
    }

    #[test]
    fn test_zeros() {
        let tf = TransferFunction::new(vec![1.0, 1.0], vec![1.0, 2.0]).unwrap(); // (s+1)/(s+2)
        let zeros = tf.zeros();
        assert_eq!(zeros.len(), 1);
        assert_relative_eq!(zeros[0].re, -1.0, epsilon = 0.01);
    }

    #[test]
    fn test_is_stable() {
        let tf = TransferFunction::first_order(1.0, 1.0).unwrap();
        assert!(tf.is_stable());
    }

    #[test]
    fn test_is_not_stable() {
        let tf = TransferFunction::new(vec![1.0], vec![1.0, -1.0]).unwrap(); // 1/(s-1)
        assert!(!tf.is_stable());
    }

    #[test]
    fn test_minimum_phase() {
        let tf = TransferFunction::new(vec![1.0, 1.0], vec![1.0, 2.0, 1.0]).unwrap();
        assert!(tf.is_minimum_phase());
    }

    #[test]
    fn test_series() {
        let h1 = TransferFunction::gain(2.0);
        let h2 = TransferFunction::gain(3.0);
        let h = h1.series(&h2);
        assert_relative_eq!(h.dc_gain(), 6.0);
    }

    #[test]
    fn test_parallel() {
        let h1 = TransferFunction::gain(2.0);
        let h2 = TransferFunction::gain(3.0);
        let h = h1.parallel(&h2);
        assert_relative_eq!(h.dc_gain(), 5.0);
    }

    #[test]
    fn test_unity_feedback() {
        let tf = TransferFunction::gain(9.0);
        let cl = tf.unity_feedback();
        assert_relative_eq!(cl.dc_gain(), 0.9, epsilon = 0.01);
    }

    #[test]
    fn test_bode() {
        let tf = TransferFunction::first_order(1.0, 1.0).unwrap();
        let bode = tf.bode((0.01, 100.0), 50);
        assert_eq!(bode.frequencies.len(), 50);
        assert_eq!(bode.magnitudes_db.len(), 50);
        assert_eq!(bode.phases_deg.len(), 50);
    }

    #[test]
    fn test_step_response() {
        let tf = TransferFunction::first_order(1.0, 0.1).unwrap();
        let resp = tf.step_response(1.0, 0.01);
        assert!(!resp.is_empty());
        // Final value should approach DC gain = 1.0
        assert_relative_eq!(resp.last().unwrap().1, 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_impulse_response() {
        let tf = TransferFunction::first_order(1.0, 0.5).unwrap();
        let resp = tf.impulse_response(2.0, 0.01);
        assert!(!resp.is_empty());
    }

    #[test]
    fn test_integrator() {
        let tf = TransferFunction::integrator(1.0);
        assert_eq!(tf.order(), 1);
        assert!(!tf.is_stable()); // pole at s=0
    }

    #[test]
    fn test_differentiator() {
        let tf = TransferFunction::differentiator(1.0);
        let h = tf.evaluate(Complex64::new(0.0, 1.0));
        assert_relative_eq!(h.im, 1.0);
    }

    #[test]
    fn test_delay() {
        let tf = TransferFunction::delay(1.0, 1);
        assert_eq!(tf.order(), 1);
    }

    #[test]
    fn test_to_state_space() {
        let tf = TransferFunction::second_order(1.0, 0.7).unwrap();
        let ss = tf.to_state_space();
        assert_eq!(ss.num_states(), 2);
    }

    #[test]
    fn test_display() {
        let tf = TransferFunction::first_order(1.0, 1.0).unwrap();
        let s = format!("{}", tf);
        assert!(s.contains("H(s)"));
    }

    #[test]
    fn test_second_order_poles() {
        let tf = TransferFunction::second_order(1.0, 0.5).unwrap();
        let poles = tf.poles();
        assert_eq!(poles.len(), 2);
        // All poles should have negative real parts
        for p in &poles {
            assert!(p.re < 0.0);
        }
    }

    #[test]
    fn test_frequency_response() {
        let tf = TransferFunction::first_order(1.0, 1.0).unwrap();
        let resp = tf.frequency_response(&[0.1, 1.0, 10.0]);
        assert_eq!(resp.len(), 3);
        // Magnitude should decrease with frequency for first-order
        assert!(resp[0].0 > resp[1].0);
        assert!(resp[1].0 > resp[2].0);
    }
}
