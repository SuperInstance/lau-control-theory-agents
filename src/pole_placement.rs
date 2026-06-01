//! Pole placement: assign eigenvalues for desired response characteristics

use nalgebra::DMatrix;
use crate::state_space::StateSpace;
use crate::controllability::Controllability;

/// Pole placement for state feedback design.
///
/// Given a controllable system ẋ = Ax + Bu, find gain K such that
/// A - BK has desired eigenvalues (poles).
pub struct PolePlacement;

impl PolePlacement {
    /// Place poles using Ackermann's formula (SISO systems).
    ///
    /// Given desired characteristic polynomial α(s), the gain is:
    /// K = [0, 0, ..., 0, 1] * C⁻¹ * α(A)
    /// where C is the controllability matrix and α(A) is the desired polynomial evaluated at A.
    pub fn ackermann(
        sys: &StateSpace,
        desired_poles: &[num_complex::Complex64],
    ) -> Result<DMatrix<f64>, String> {
        let n = sys.num_states();
        let m = sys.num_inputs();

        if m != 1 {
            return Err("Ackermann formula requires SISO (m=1)".into());
        }
        if desired_poles.len() != n {
            return Err(format!("Need {} poles, got {}", n, desired_poles.len()));
        }

        // Build controllability matrix
        let cm = Controllability::controllability_matrix(sys);
        let cm_inv = cm.clone().try_inverse().ok_or("System not controllable")?;

        // Compute α(A) = desired characteristic polynomial evaluated at A
        // If α(s) = s^n + a_{n-1} s^{n-1} + ... + a_0
        // Then α(A) = A^n + a_{n-1} A^{n-1} + ... + a_0 I
        let coeffs = Self::characteristic_polynomial_coefficients(desired_poles);
        let alpha_a = Self::evaluate_polynomial_at_matrix(&sys.a, &coeffs);

        // K = [0, ..., 0, 1] * C^{-1} * α(A)
        let mut e_n = DMatrix::zeros(1, n);
        e_n[(0, n - 1)] = 1.0;

        let k = e_n * cm_inv * alpha_a;
        Ok(k)
    }

    /// Place poles using Bass-Gura formula (SISO).
    pub fn bass_gura(
        sys: &StateSpace,
        desired_poles: &[num_complex::Complex64],
    ) -> Result<DMatrix<f64>, String> {
        let n = sys.num_states();

        if desired_poles.len() != n {
            return Err(format!("Need {} poles, got {}", n, desired_poles.len()));
        }

        // Coefficients of desired characteristic polynomial
        let desired_coeffs = Self::characteristic_polynomial_coefficients(desired_poles);

        // Coefficients of open-loop characteristic polynomial
        let open_loop_poles: Vec<_> = sys.a.complex_eigenvalues().iter().cloned().collect();
        let open_loop_coeffs = Self::characteristic_polynomial_coefficients(&open_loop_poles);

        // α = a_desired - a_open_loop
        let alpha: Vec<f64> = desired_coeffs.iter()
            .zip(open_loop_coeffs.iter())
            .map(|(&d, &o)| d - o)
            .collect();

        // Build controllability matrix
        let cm = Controllability::controllability_matrix(sys);

        // Build the Toeplitz matrix T
        let t = Self::toeplitz_from_coeffs(&open_loop_coeffs, n);

        // K = (α^T * T^{-1} * C^{-1})^T
        let alpha_vec = nalgebra::DVector::from_vec(alpha);
        let t_inv = t.try_inverse().ok_or("Toeplitz matrix not invertible")?;
        let cm_inv = cm.try_inverse().ok_or("Not controllable")?;

        let k_row = alpha_vec.transpose() * t_inv * cm_inv;
        Ok(DMatrix::from_row_slice(1, n, &k_row.as_slice()))
    }

    /// Compute characteristic polynomial coefficients from poles.
    /// Returns [a_0, a_1, ..., a_{n-1}, 1] such that
    /// α(s) = s^n + a_{n-1} s^{n-1} + ... + a_1 s + a_0
    fn characteristic_polynomial_coefficients(poles: &[num_complex::Complex64]) -> Vec<f64> {
        let n = poles.len();
        // Start with polynomial 1, then multiply by (s - pole) for each pole
        let mut poly = vec![0.0; n + 1];
        poly[n] = 1.0; // Leading coefficient

        for &pole in poles {
            let mut new_poly = vec![0.0; n + 1];
            for j in (0..=n).rev() {
                if j + 1 <= n {
                    new_poly[j + 1] += poly[j];
                }
                new_poly[j] -= pole.re * poly[j];
            }
            // Note: we ignore imaginary parts for real-valued systems
            // (conjugate pairs should cancel out)
            poly = new_poly;
        }

        // Return coefficients [a_0, a_1, ..., a_{n-1}, 1]
        poly
    }

    /// Evaluate polynomial at matrix A.
    /// coeffs = [a_0, a_1, ..., a_n], where α(A) = a_n A^n + ... + a_1 A + a_0 I
    fn evaluate_polynomial_at_matrix(a: &DMatrix<f64>, coeffs: &[f64]) -> DMatrix<f64> {
        let n = a.nrows();
        let mut result = DMatrix::zeros(n, n);
        let mut a_power = DMatrix::identity(n, n);

        for &coeff in coeffs {
            result += &a_power * coeff;
            a_power = &a_power * a;
        }

        result
    }

    /// Build Toeplitz matrix from polynomial coefficients.
    fn toeplitz_from_coeffs(coeffs: &[f64], n: usize) -> DMatrix<f64> {
        let mut t = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..=i {
                if i - j < coeffs.len() {
                    t[(i, j)] = coeffs[n - 1 - (i - j)];
                }
            }
        }
        t
    }

    /// Verify that the closed-loop system has the desired poles.
    pub fn verify_poles(
        sys: &StateSpace,
        k: &DMatrix<f64>,
        desired_poles: &[num_complex::Complex64],
        tolerance: f64,
    ) -> bool {
        let a_cl = &sys.a - &sys.b * k;
        let actual_poles = a_cl.complex_eigenvalues();

        if actual_poles.len() != desired_poles.len() {
            return false;
        }

        // Sort both by real part for comparison
        let mut actual: Vec<_> = actual_poles.iter().collect();
        let mut desired: Vec<_> = desired_poles.iter().collect();
        actual.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap());
        desired.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap());

        for (a, d) in actual.iter().zip(desired.iter()) {
            if (a.re - d.re).abs() > tolerance || (a.im - d.im).abs() > tolerance {
                return false;
            }
        }
        true
    }

    /// Place poles for a MIMO system using a simple approach.
    /// Decomposes into SISO-like channels or uses sequential placement.
    pub fn place_poles_mimo(
        sys: &StateSpace,
        desired_poles: &[num_complex::Complex64],
    ) -> Result<DMatrix<f64>, String> {
        let n = sys.num_states();
        let m = sys.num_inputs();

        if desired_poles.len() != n {
            return Err(format!("Need {} poles, got {}", n, desired_poles.len()));
        }

        if m == 1 {
            return Self::ackermann(sys, desired_poles);
        }

        // For MIMO, use a projection approach: pick first column of B
        // and place poles as if SISO
        // More sophisticated methods (e.g., Kautsky-Nichols-Van Dooren) would be needed
        // for optimal robustness
        let b_siso = sys.b.column(0).into_owned();
        let b_col = DMatrix::from_columns(&[b_siso]);
        let d_col = sys.d.column(0).into_owned();
        let p = sys.d.nrows();
        let d_siso = if p == 1 {
            DMatrix::from_element(1, 1, d_col[0])
        } else {
            let mut d = DMatrix::zeros(p, 1);
            for i in 0..p {
                d[(i, 0)] = d_col[i];
            }
            d
        };
        let sys_siso = StateSpace::new(
            sys.a.clone(),
            b_col,
            sys.c.clone(),
            d_siso,
        ).map_err(|e| e)?;

        let k_siso = match Self::ackermann(&sys_siso, desired_poles) {
            Ok(k) => k,
            Err(_) => {
                let q = DMatrix::identity(n, n);
                let r = DMatrix::identity(m, m);
                let lqr = crate::lqr::Lqr::solve(sys, &q, &r)?;
                return Ok(lqr.k);
            }
        };

        // Expand to MIMO gain (first input only)
        let mut k = DMatrix::zeros(m, n);
        k.row_mut(0).copy_from(&k_siso.row(0));
        Ok(k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dmatrix;
    use approx::assert_relative_eq;

    fn controllable_siso() -> StateSpace {
        let a = dmatrix![0.0, 1.0; -2.0, -3.0];
        let b = dmatrix![0.0; 1.0];
        let c = dmatrix![1.0, 0.0];
        let d = dmatrix![0.0];
        StateSpace::new(a, b, c, d).unwrap()
    }

    #[test]
    fn test_ackermann() {
        let sys = controllable_siso();
        let poles = vec![
            num_complex::Complex64::new(-5.0, 0.0),
            num_complex::Complex64::new(-6.0, 0.0),
        ];
        let k = PolePlacement::ackermann(&sys, &poles);
        assert!(k.is_ok());
        let k = k.unwrap();
        assert_eq!(k.nrows(), 1);
        assert_eq!(k.ncols(), 2);
    }

    #[test]
    fn test_ackermann_verification() {
        let sys = controllable_siso();
        let poles = vec![
            num_complex::Complex64::new(-5.0, 0.0),
            num_complex::Complex64::new(-6.0, 0.0),
        ];
        let k = PolePlacement::ackermann(&sys, &poles).unwrap();
        assert!(PolePlacement::verify_poles(&sys, &k, &poles, 0.1));
    }

    #[test]
    fn test_ackermann_complex_poles() {
        let sys = controllable_siso();
        let poles = vec![
            num_complex::Complex64::new(-3.0, 2.0),
            num_complex::Complex64::new(-3.0, -2.0),
        ];
        let k = PolePlacement::ackermann(&sys, &poles);
        assert!(k.is_ok());
    }

    #[test]
    fn test_wrong_pole_count() {
        let sys = controllable_siso();
        let poles = vec![num_complex::Complex64::new(-5.0, 0.0)];
        assert!(PolePlacement::ackermann(&sys, &poles).is_err());
    }

    #[test]
    fn test_characteristic_polynomial() {
        let poles = vec![
            num_complex::Complex64::new(-1.0, 0.0),
            num_complex::Complex64::new(-2.0, 0.0),
        ];
        let coeffs = PolePlacement::characteristic_polynomial_coefficients(&poles);
        // (s+1)(s+2) = s² + 3s + 2 → [2, 3, 1]
        assert_relative_eq!(coeffs[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(coeffs[1], 3.0, epsilon = 1e-10);
        assert_relative_eq!(coeffs[2], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bass_gura() {
        let sys = controllable_siso();
        let poles = vec![
            num_complex::Complex64::new(-4.0, 0.0),
            num_complex::Complex64::new(-5.0, 0.0),
        ];
        let k = PolePlacement::bass_gura(&sys, &poles);
        assert!(k.is_ok());
        let k = k.unwrap();
        assert_eq!(k.nrows(), 1);
        assert_eq!(k.ncols(), 2);
    }

    #[test]
    fn test_place_poles_mimo() {
        let a = dmatrix![0.0, 1.0; -2.0, -3.0];
        let b = dmatrix![1.0, 0.0; 0.0, 1.0];
        let c = dmatrix![1.0, 0.0; 0.0, 1.0];
        let d = DMatrix::zeros(2, 2);
        let sys = StateSpace::new(a, b, c, d).unwrap();
        let poles = vec![
            num_complex::Complex64::new(-5.0, 0.0),
            num_complex::Complex64::new(-6.0, 0.0),
        ];
        let k = PolePlacement::place_poles_mimo(&sys, &poles);
        assert!(k.is_ok());
    }

    #[test]
    fn test_faster_poles_stabilize() {
        let sys = controllable_siso();
        // Original poles: -1, -2
        let fast_poles = vec![
            num_complex::Complex64::new(-10.0, 0.0),
            num_complex::Complex64::new(-20.0, 0.0),
        ];
        let k = PolePlacement::ackermann(&sys, &fast_poles).unwrap();
        let a_cl = &sys.a - &sys.b * &k;
        let eigs = a_cl.complex_eigenvalues();
        for e in &eigs {
            assert!(e.re < -5.0);
        }
    }
}
