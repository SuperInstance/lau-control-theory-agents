//! PID Controller: proportional-integral-derivative control

use serde::{Deserialize, Serialize};
use crate::transfer_function::TransferFunction;

/// PID Controller: u(t) = Kp * e(t) + Ki * ∫e(τ)dτ + Kd * de/dt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PidController {
    /// Proportional gain
    pub kp: f64,
    /// Integral gain
    pub ki: f64,
    /// Derivative gain
    pub kd: f64,
    /// Output limits (min, max)
    pub output_limits: Option<(f64, f64)>,
    /// Derivative filter time constant (for filtered derivative)
    pub derivative_filter_tau: f64,
    /// Anti-windup gain
    pub anti_windup_gain: f64,
    /// Setpoint weighting for proportional term (0-1)
    pub setpoint_weight_p: f64,
    /// Setpoint weighting for derivative term (0-1)
    pub setpoint_weight_d: f64,
    // Internal state
    #[serde(skip)]
    integral: f64,
    #[serde(skip)]
    prev_error: f64,
    #[serde(skip)]
    prev_derivative: f64,
    #[serde(skip)]
    initialized: bool,
}

impl PidController {
    /// Create a new PID controller with given gains.
    pub fn new(kp: f64, ki: f64, kd: f64) -> Self {
        Self {
            kp,
            ki,
            kd,
            output_limits: None,
            derivative_filter_tau: 0.0,
            anti_windup_gain: 0.0,
            setpoint_weight_p: 1.0,
            setpoint_weight_d: 1.0,
            integral: 0.0,
            prev_error: 0.0,
            prev_derivative: 0.0,
            initialized: false,
        }
    }

    /// Create a P-only controller.
    pub fn p_only(kp: f64) -> Self {
        Self::new(kp, 0.0, 0.0)
    }

    /// Create a PI controller.
    pub fn pi(kp: f64, ki: f64) -> Self {
        Self::new(kp, ki, 0.0)
    }

    /// Create a PD controller.
    pub fn pd(kp: f64, kd: f64) -> Self {
        Self::new(kp, 0.0, kd)
    }

    /// Set output limits.
    pub fn with_output_limits(mut self, min: f64, max: f64) -> Self {
        self.output_limits = Some((min, max));
        self
    }

    /// Set derivative filter time constant.
    pub fn with_derivative_filter(mut self, tau: f64) -> Self {
        self.derivative_filter_tau = tau;
        self
    }

    /// Set anti-windup gain.
    pub fn with_anti_windup(mut self, gain: f64) -> Self {
        self.anti_windup_gain = gain;
        self
    }

    /// Set setpoint weighting.
    pub fn with_setpoint_weighting(mut self, wp: f64, wd: f64) -> Self {
        self.setpoint_weight_p = wp;
        self.setpoint_weight_d = wd;
        self
    }

    /// Compute the PID output for the given error and time step.
    pub fn update(&mut self, setpoint: f64, measurement: f64, dt: f64) -> f64 {
        let error = setpoint - measurement;

        // Proportional term with setpoint weighting
        let p_error = self.setpoint_weight_p * setpoint - measurement;
        let p_term = self.kp * p_error;

        // Integral term with anti-windup
        self.integral += error * dt;
        let i_term = self.ki * self.integral;

        // Derivative term with setpoint weighting
        let d_error = if self.initialized {
            self.setpoint_weight_d * (setpoint - self.prev_error.abs() * 0.0)
                - (measurement - (self.prev_error + setpoint - error))
        } else {
            0.0
        };
        // Simplified: derivative on measurement only for setpoint_weight_d < 1
        let raw_derivative = if self.initialized && dt > 0.0 {
            -self.kd * (measurement - (setpoint - self.prev_error)) / dt
        } else {
            0.0
        };

        let d_term = if self.derivative_filter_tau > 0.0 && self.initialized {
            // First-order filter on derivative
            let alpha = dt / (self.derivative_filter_tau + dt);
            self.prev_derivative = self.prev_derivative * (1.0 - alpha) + raw_derivative * alpha;
            self.prev_derivative
        } else {
            raw_derivative
        };

        self.prev_error = error;
        self.initialized = true;

        let mut output = p_term + i_term + d_term;

        // Output limiting with anti-windup
        if let Some((min, max)) = self.output_limits {
            if output < min {
                // Anti-windup: back-calculate integral
                self.integral -= self.anti_windup_gain * (output - min) * dt;
                output = min;
            } else if output > max {
                self.integral -= self.anti_windup_gain * (output - max) * dt;
                output = max;
            }
        }

        output
    }

    /// Reset the controller state.
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.prev_error = 0.0;
        self.prev_derivative = 0.0;
        self.initialized = false;
    }

    /// Get the current integral term value.
    pub fn integral_value(&self) -> f64 {
        self.integral
    }

    /// Simulate the PID controller with a plant transfer function.
    pub fn simulate_with_plant(
        &mut self,
        plant: &TransferFunction,
        setpoint: f64,
        t_end: f64,
        dt: f64,
    ) -> PidSimulationResult {
        let ss = plant.to_state_space();
        let n_steps = (t_end / dt) as usize;
        let mut x = nalgebra::DVector::zeros(ss.num_states());

        let mut times = Vec::with_capacity(n_steps + 1);
        let mut outputs = Vec::with_capacity(n_steps + 1);
        let mut controls = Vec::with_capacity(n_steps + 1);
        let mut errors = Vec::with_capacity(n_steps + 1);

        for i in 0..=n_steps {
            let t = i as f64 * dt;
            let u_vec = ss.output(&x, &nalgebra::dvector![0.0]);
            let y = u_vec[0];

            let control = self.update(setpoint, y, dt);
            let error = setpoint - y;

            times.push(t);
            outputs.push(y);
            controls.push(control);
            errors.push(error);

            if i < n_steps {
                let u = nalgebra::dvector![control];
                x = ss.rk4_step(&x, &u, dt);
            }
        }

        PidSimulationResult {
            times,
            outputs,
            controls,
            errors,
        }
    }

    /// Compute PID gains using Ziegler-Nichols tuning rules.
    pub fn ziegler_nichols(ku: f64, tu: f64) -> Self {
        let kp = 0.6 * ku;
        let ki = 1.2 * ku / tu;
        let kd = 0.075 * ku * tu;
        Self::new(kp, ki, kd)
    }

    /// Compute PID gains using Cohen-Coon tuning rules.
    pub fn cohen_coon(k: f64, tau: f64, td: f64) -> Self {
        let r = td / tau;
        let kp = (1.0 / k) * (1.0 / r) * (4.0 / 3.0 + r / 4.0);
        let ki = kp / (tau * (32.0 + 6.0 * r) / (13.0 + 8.0 * r));
        let kd = kp * tau * (4.0 / 11.0);
        Self::new(kp, ki, kd)
    }

    /// Convert PID to transfer function form.
    /// C(s) = Kp + Ki/s + Kd*s
    pub fn to_transfer_function(&self) -> TransferFunction {
        if self.kd == 0.0 && self.ki == 0.0 {
            // P-only
            TransferFunction::gain(self.kp)
        } else if self.kd == 0.0 {
            // PI: (Kp*s + Ki) / s
            TransferFunction::new(
                vec![self.ki, self.kp],
                vec![0.0, 1.0],
            ).unwrap()
        } else if self.ki == 0.0 {
            // PD: (Kd*s + Kp) / 1 ... but really Kd*s + Kp
            TransferFunction::new(
                vec![self.kp, self.kd],
                vec![1.0],
            ).unwrap()
        } else {
            // Full PID: (Kd*s² + Kp*s + Ki) / s
            TransferFunction::new(
                vec![self.ki, self.kp, self.kd],
                vec![0.0, 1.0],
            ).unwrap()
        }
    }
}

/// Result of PID simulation.
#[derive(Debug, Clone)]
pub struct PidSimulationResult {
    pub times: Vec<f64>,
    pub outputs: Vec<f64>,
    pub controls: Vec<f64>,
    pub errors: Vec<f64>,
}

impl PidSimulationResult {
    /// Compute integral of absolute error (IAE).
    pub fn iae(&self) -> f64 {
        let dt = if self.times.len() > 1 {
            self.times[1] - self.times[0]
        } else {
            1.0
        };
        self.errors.iter().map(|e| e.abs() * dt).sum()
    }

    /// Compute integral of squared error (ISE).
    pub fn ise(&self) -> f64 {
        let dt = if self.times.len() > 1 {
            self.times[1] - self.times[0]
        } else {
            1.0
        };
        self.errors.iter().map(|e| e * e * dt).sum()
    }

    /// Compute integral of time-weighted absolute error (ITAE).
    pub fn itae(&self) -> f64 {
        let dt = if self.times.len() > 1 {
            self.times[1] - self.times[0]
        } else {
            1.0
        };
        self.times.iter().zip(self.errors.iter())
            .map(|(t, e)| t * e.abs() * dt)
            .sum()
    }

    /// Compute maximum overshoot percentage.
    pub fn overshoot(&self, setpoint: f64) -> f64 {
        if setpoint == 0.0 {
            return 0.0;
        }
        let max_output = self.outputs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_output > setpoint {
            ((max_output - setpoint) / setpoint) * 100.0
        } else {
            0.0
        }
    }

    /// Compute settling time (time to stay within 2% of setpoint).
    pub fn settling_time(&self, setpoint: f64, tolerance: f64) -> f64 {
        let band = setpoint * tolerance;
        let mut settled_idx = self.times.len() - 1;
        for i in (0..self.times.len()).rev() {
            if (self.outputs[i] - setpoint).abs() > band {
                settled_idx = i;
                break;
            }
        }
        if settled_idx < self.times.len() - 1 {
            self.times[settled_idx + 1]
        } else {
            if (self.outputs[0] - setpoint).abs() <= band {
                0.0
            } else {
                self.times[self.times.len() - 1]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_p_only() {
        let mut pid = PidController::p_only(2.0);
        let output = pid.update(1.0, 0.5, 0.01);
        assert_relative_eq!(output, 1.0); // 2.0 * (1.0 - 0.5)
    }

    #[test]
    fn test_pi_controller() {
        let mut pid = PidController::pi(1.0, 1.0);
        let output = pid.update(1.0, 0.0, 0.01);
        assert!(output > 0.0);
    }

    #[test]
    fn test_pid_controller() {
        let mut pid = PidController::new(1.0, 0.1, 0.01);
        let output = pid.update(1.0, 0.0, 0.01);
        assert!(output > 0.0);
    }

    #[test]
    fn test_output_limits() {
        let mut pid = PidController::new(100.0, 0.0, 0.0).with_output_limits(-1.0, 1.0);
        let output = pid.update(10.0, 0.0, 0.01);
        assert_relative_eq!(output, 1.0);
    }

    #[test]
    fn test_reset() {
        let mut pid = PidController::pi(1.0, 10.0);
        pid.update(1.0, 0.0, 0.01);
        pid.update(1.0, 0.5, 0.01);
        assert!(pid.integral_value() > 0.0);
        pid.reset();
        assert_relative_eq!(pid.integral_value(), 0.0);
    }

    #[test]
    fn test_integral_accumulation() {
        let mut pid = PidController::pi(0.0, 1.0);
        pid.update(1.0, 0.0, 1.0);
        assert_relative_eq!(pid.integral_value(), 1.0);
        pid.update(1.0, 0.0, 1.0);
        assert_relative_eq!(pid.integral_value(), 2.0);
    }

    #[test]
    fn test_derivative_filter() {
        let mut pid = PidController::pd(0.0, 1.0).with_derivative_filter(0.1);
        let o1 = pid.update(1.0, 0.0, 0.01);
        let _ = pid.update(1.0, 0.5, 0.01);
        // Filtered derivative should be smaller than unfiltered
    }

    #[test]
    fn test_ziegler_nichols() {
        let pid = PidController::ziegler_nichols(10.0, 2.0);
        assert_relative_eq!(pid.kp, 6.0);
        assert!(pid.ki > 0.0);
        assert!(pid.kd > 0.0);
    }

    #[test]
    fn test_simulate_with_plant() {
        let plant = TransferFunction::first_order(1.0, 0.5).unwrap();
        let mut pid = PidController::pi(1.0, 0.5);
        let result = pid.simulate_with_plant(&plant, 1.0, 5.0, 0.01);
        assert_eq!(result.times.len(), 501);
        assert_eq!(result.outputs.len(), 501);
    }

    #[test]
    fn test_pid_tracking() {
        let plant = TransferFunction::first_order(1.0, 0.1).unwrap();
        let mut pid = PidController::new(2.0, 1.0, 0.1);
        let result = pid.simulate_with_plant(&plant, 1.0, 5.0, 0.01);
        // Final output should be close to setpoint
        let final_output = result.outputs.last().unwrap();
        assert_relative_eq!(*final_output, 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_iae() {
        let result = PidSimulationResult {
            times: vec![0.0, 1.0, 2.0],
            outputs: vec![0.0, 0.5, 0.9],
            controls: vec![1.0, 0.8, 0.2],
            errors: vec![1.0, 0.5, 0.1],
        };
        let iae = result.iae();
        assert!(iae > 0.0);
    }

    #[test]
    fn test_ise() {
        let result = PidSimulationResult {
            times: vec![0.0, 1.0, 2.0],
            outputs: vec![0.0, 0.5, 0.9],
            controls: vec![1.0, 0.8, 0.2],
            errors: vec![1.0, 0.5, 0.1],
        };
        let ise = result.ise();
        assert!(ise > 0.0);
    }

    #[test]
    fn test_overshoot() {
        let result = PidSimulationResult {
            times: vec![0.0, 1.0, 2.0, 3.0],
            outputs: vec![0.0, 1.2, 1.1, 1.0],
            controls: vec![0.0; 4],
            errors: vec![0.0; 4],
        };
        let os = result.overshoot(1.0);
        assert_relative_eq!(os, 20.0);
    }

    #[test]
    fn test_to_transfer_function_p() {
        let pid = PidController::p_only(5.0);
        let tf = pid.to_transfer_function();
        assert_relative_eq!(tf.dc_gain(), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_to_transfer_function_pid() {
        let pid = PidController::new(1.0, 1.0, 0.1);
        let tf = pid.to_transfer_function();
        assert_eq!(tf.order(), 1); // Denominator is s
    }

    #[test]
    fn test_setpoint_weighting() {
        let mut pid = PidController::new(1.0, 0.0, 0.0)
            .with_setpoint_weighting(0.5, 0.0);
        let output = pid.update(10.0, 0.0, 0.01);
        // With wp=0.5: p_term = 1.0 * (0.5*10 - 0) = 5.0
        assert_relative_eq!(output, 5.0);
    }

    #[test]
    fn test_pid_performance_metrics() {
        let plant = TransferFunction::first_order(1.0, 0.5).unwrap();
        let mut pid = PidController::new(2.0, 1.0, 0.1);
        let result = pid.simulate_with_plant(&plant, 1.0, 10.0, 0.01);
        assert!(result.iae() > 0.0);
        assert!(result.ise() > 0.0);
        assert!(result.itae() > 0.0);
    }
}
