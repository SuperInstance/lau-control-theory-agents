# lau-control-theory-agents

**Classical and modern control theory for agents — stability, controllability, observability, optimal control, robustness.**

A Rust library implementing state-space control design for Linear Time-Invariant (LTI) systems: PID controllers, LQR/LQG optimal control, H∞ robust control, pole placement, Pontryagin's maximum principle, stability analysis (Lyapunov, Hurwitz), controllability/observability Gramians, and transfer function analysis — all built on the standard ẋ = Ax + Bu framework.

[![134 tests passing](https://img.shields.io/badge/tests-134%20passing-brightgreen)]()

---

## What This Does

Control theory provides principled methods for designing agents that behave predictably and optimally. This library covers the full pipeline:

1. **Model** agent dynamics as a state-space system ẋ = Ax + Bu, y = Cx + Du
2. **Analyze** stability (eigenvalues, Lyapunov), controllability (can we steer anywhere?), observability (can we infer the state?)
3. **Design** controllers: PID, LQR, LQG, pole placement, H∞, Pontryagin
4. **Validate** with frequency-domain analysis via transfer functions

## Key Idea

The state-space representation ẋ = Ax + Bu captures agent dynamics in matrix form. Everything else follows from linear algebra: stability = eigenvalues of A have negative real parts; controllability = [B, AB, A²B, ...] has full rank; optimal control = solve the Riccati equation; robustness = minimize the H∞ norm.

## Install

```toml
[dependencies]
lau-control-theory-agents = "0.1.0"
```

Requires Rust 2021 edition. Dependencies: `nalgebra` (with serde), `num-complex`, `num-traits`, `serde`.

## Quick Start

```rust
use lau_control_theory_agents::{StateSpace, Lqr, Controllability, Stability, PidController};
use nalgebra::{DMatrix, DVector};

fn main() {
    // Double integrator: ẍ = u → state [x, ẋ]
    let a = DMatrix::from_row_slice(2, 2, &[
        0.0, 1.0,
        0.0, 0.0,
    ]);
    let b = DMatrix::from_row_slice(2, 1, &[0.0, 1.0]);
    let c = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
    let d = DMatrix::zeros(1, 1);

    let sys = StateSpace::new(a, b, c, d).unwrap();

    // Check system properties
    println!("Controllable: {}", Controllability::is_controllable(&sys));
    println!("Stable: {}", Stability::is_hurwitz(&sys.a));

    // Design LQR controller
    let q = DMatrix::identity(2, 2);  // state cost
    let r = DMatrix::identity(1, 1);  // control cost
    let lqr = Lqr::solve(&sys, &q, &r).unwrap();
    println!("LQR gain K:\n{}", lqr.k);

    // Simulate closed-loop response
    let x0 = DVector::from_vec(vec![1.0, 0.0]);
    let response = sys.simulate(&x0, &lqr.k, 0.01, 200);
    println!("Final state: {}", response.last().unwrap());
}
```

## API Reference

### Module: `state_space` — LTI System Representation

| Type / Method | Description |
|---|---|
| `StateSpace` | LTI system: ẋ = Ax + Bu, y = Cx + Du |
| `StateSpace::new(A, B, C, D)` | Create with dimension validation |
| `.num_states() → usize` | State dimension n |
| `.num_inputs() → usize` | Input dimension m |
| `.num_outputs() → usize` | Output dimension p |
| `.simulate(x0, K, dt, steps) → Vec<DVector>` | Closed-loop: x' = (A−BK)x with initial state x0 |
| `.simulate_open_loop(x0, u_func, dt, steps) → Vec<DVector>` | Open-loop with time-varying input |
| `.step(x, u) → DVector` | Single Euler step: x + dt(Ax + Bu) |
| `.output(x, u) → DVector` | y = Cx + Du |
| `.eigenvalues() → Vec<Complex64>` | Eigenvalues of A |
| `.is_stable() → bool` | All eigenvalues have Re < 0 |
| `.transfer_matrix(s) → DMatrix<Complex64>` | G(s) = C(sI−A)⁻¹B + D |
| `.controllable_canonical_form() → StateSpace` | Transform to CCF |
| `.observable_canonical_form() → StateSpace` | Transform to OCF |
| `.series(other) → StateSpace` | Series interconnection |
| `.parallel(other) → StateSpace` | Parallel interconnection |
| `.feedback(K) → StateSpace` | Closed-loop with gain K |
| `.augment(disturbance, noise) → StateSpace` | Augment with noise inputs |

### Module: `controllability` — Can We Reach Any State?

| Type / Method | Description |
|---|---|
| `Controllability` | Static methods for controllability analysis |
| `.controllability_matrix(sys) → DMatrix` | C = [B, AB, A²B, ..., Aⁿ⁻¹B] |
| `.is_controllable(sys) → bool` | rank(C) = n? |
| `.rank(m) → usize` | SVD-based rank computation |
| `.controllability_gramian(sys) → DMatrix` | Wc via Lyapunov: A·Wc + Wc·A^T + BB^T = 0 |
| `.controllability_index(sys) → usize` | Min columns of C for full rank |
| `.minimum_energy(sys, x0, xf, T) → f64` | Min control energy to steer x0 → xf |
| `.uncontrollable_modes(sys) → Vec<Complex64>` | Eigenvalues of A not reachable from B |

### Module: `observability` — Can We Determine State from Output?

| Type / Method | Description |
|---|---|
| `Observability` | Static methods for observability analysis |
| `.observability_matrix(sys) → DMatrix` | O = [C; CA; CA²; ...; CAⁿ⁻¹] |
| `.is_observable(sys) → bool` | rank(O) = n? |
| `.observability_gramian(sys) → DMatrix` | Wo via Lyapunov: A^T·Wo + Wo·A + C^TC = 0 |
| `.is_observable_dual(sys) → bool` | Via duality: (A^T, C^T) controllable? |
| `.unobservable_modes(sys) → Vec<Complex64>` | Eigenvalues not visible in output |
| `.observability_index(sys) → usize` | Min rows of O for full rank |

### Module: `stability` — Lyapunov and Eigenvalue Analysis

| Type / Method | Description |
|---|---|
| `Stability` | Static methods for stability analysis |
| `.is_hurwitz(A) → bool` | All eigenvalues have Re < 0 |
| `.is_lyapunov_stable(A) → bool` | All Re ≤ 0, imaginary axis eigenvalues non-defective |
| `.is_asymptotically_stable(A) → bool` | Same as Hurwitz |
| `.stability_margin(A) → f64` | Distance of closest eigenvalue to imaginary axis |
| `.phase_margin(sys, num_freqs) → f64` | Phase margin in degrees |
| `.gain_margin(sys, num_freqs) → f64` | Gain margin (multiplicative) |
| `.lyapunov_exponents(A, x0, dt, steps) → Vec<f64>` | Estimate Lyapunov exponents |
| `.find_lyapunov_function(A, Q) → DMatrix` | Solve A^TP + PA = −Q for P |
| `.is_passive(sys) → bool` | Check dissipativity condition |

### Module: `lqr` — Linear Quadratic Regulator

| Type / Method | Description |
|---|---|
| `Lqr` | Static methods for LQR design |
| `Lqr::solve(sys, Q, R) → LqrResult` | Minimize ∫(x^TQx + u^TRu) dt |
| `Lqr::solve_care(A, BR⁻¹B^T, Q) → DMatrix` | Solve CARE via Smith's iterative doubling |
| `Lqr::infinite_horizon(sys, Q, R) → LqrResult` | Same as solve (steady-state) |
| `Lqr::finite_horizon(sys, Q, R, Sf, T, steps) → Vec<DMatrix>` | Time-varying Riccati solution |
| `LqrResult` | k (gain), p (Riccati solution), a_cl (closed-loop A), r_inv |

### Module: `lqg` — Linear Quadratic Gaussian (LQR + Kalman Filter)

| Type / Method | Description |
|---|---|
| `Lqg` | Static methods for LQG design |
| `Lqg::design(sys, Q, R, V, W) → LqgResult` | LQR + Kalman filter combined |
| `Lqg::design_kalman_filter(sys, V, W) → KalmanResult` | Design Kalman observer |
| `LqgResult` | k (LQR gain), l (Kalman gain), a_obs, b_obs, observer matrices |
| `KalmanResult` | l (gain), p (error covariance), steady-state filter |

### Module: `h_infinity` — Robust Control

| Type / Method | Description |
|---|---|
| `HInfinity` | H∞ robust control methods |
| `.h_inf_norm(sys, num_freqs) → f64` | Upper bound on ‖G‖∞ |
| `.max_singular_value_at_freq(sys, ω) → f64` | σ_max(G(jω)) |
| `.sensitivity_function(sys, K) → StateSpace` | S = (I + GK)⁻¹ |
| `.complementary_sensitivity(sys, K) → StateSpace` | T = GK(I + GK)⁻¹ |
| `.mixed_sensitivity_design(sys, W1, W2, W3) → DMatrix` | Mixed-sensitivity H∞ synthesis |
| `.gamma_iteration(sys, γ_init) → f64` | Bisection to find optimal γ |

### Module: `pole_placement` — Eigenvalue Assignment

| Type / Method | Description |
|---|---|
| `PolePlacement` | Assign closed-loop eigenvalues |
| `PolePlacement::ackermann(sys, poles) → DMatrix` | Ackermann's formula (SISO) |
| `PolePlacement::characteristic_polynomial_coefficients(poles) → Vec<f64>` | From roots to polynomial |
| `PolePlacement::evaluate_polynomial_at_matrix(A, coeffs) → DMatrix` | α(A) = Aⁿ + aₙ₋₁Aⁿ⁻¹ + ... + a₀I |
| `PolePlacement::bass_gura(sys, poles) → DMatrix` | Bass-Gura formula (SISO alternative) |

### Module: `transfer_function` — Frequency-Domain Analysis

| Type / Method | Description |
|---|---|
| `TransferFunction` | H(s) = num(s)/den(s) as polynomial ratio |
| `TransferFunction::new(num, den)` | Create from coefficients [b₀, b₁, ...] |
| `TransferFunction::gain(k)` | H(s) = K |
| `TransferFunction::first_order(k, τ)` | H(s) = K/(τs + 1) |
| `TransferFunction::second_order(k, ωn, ζ)` | H(s) = Kωn²/(s² + 2ζωns + ωn²) |
| `.evaluate(s) → Complex64` | Evaluate H(s) at complex frequency |
| `.frequency_response(ω) → Complex64` | H(jω) |
| `.magnitude(ω) → f64` | |H(jω)| |
| `.phase(ω) → f64` | ∠H(jω) in radians |
| `.poles() → Vec<Complex64>` | Roots of denominator |
| `.zeros() → Vec<Complex64>` | Roots of numerator |
| `.is_stable() → bool` | All poles have Re < 0 |
| `.dc_gain() → f64` | H(0) |
| `.bandwidth(γ) → f64` | Frequency where |H| drops to γ·|H(0)| |
| `.bode(num_freqs) → Vec<(f64, f64, f64)>` | (ω, magnitude_dB, phase_deg) |
| `.step_response(dt, steps) → Vec<f64>` | Unit step response |
| `.impulse_response(dt, steps) → Vec<f64>` | Impulse response |
| `.series(other) → TransferFunction` | Series connection |
| `.parallel(other) → TransferFunction` | Parallel connection |
| `.feedback() → TransferFunction` | Unity negative feedback |
| `.pid_compensator(kp, ki, kd) → TransferFunction` | C(s) = kp + ki/s + kd·s |
| `.to_state_space() → StateSpace` | Convert to observable canonical form |

### Module: `pid` — Proportional-Integral-Derivative Controller

| Type / Method | Description |
|---|---|
| `PidController` | PID with anti-windup, derivative filter, setpoint weighting |
| `PidController::new(kp, ki, kd)` | Create with gains |
| `.update(setpoint, measurement, dt) → f64` | Compute control output |
| `.reset()` | Reset integral and derivative state |
| `.set_output_limits(min, max)` | Clamp output |
| `.set_derivative_filter(tau)` | Low-pass filter on derivative |
| `.set_anti_windup(gain)` | Back-calculation anti-windup |
| `.set_setpoint_weighting(b_p, b_d)` | Setpoint weighting for P and D terms |
| `.to_transfer_function() → TransferFunction` | C(s) = kp + ki/s + kd·s |
| `.tune_ziegler_nichols(ku, tu) → PidController` | Ziegler-Nichols tuning |
| `.tune_cohen_coon(k, τ, θ) → PidController` | Cohen-Coon tuning for FOPDT |
| `.integral_term() → f64` | Current integral accumulator |
| `.kp`, `.ki`, `.kd` | Gains |
| `.output_limits` | Optional (min, max) |

### Module: `pontryagin` — Pontryagin's Maximum Principle

| Type / Method | Description |
|---|---|
| `Pontryagin` | Necessary conditions for optimal control |
| `Pontryagin::solve_lqp(sys, Q, R, Sf, T, steps) → PontryaginResult` | Solve finite-horizon LQP |
| `Pontryagin::costate_equation(A, B, R⁻¹, Q) → DMatrix` | λ̇ = −A^Tλ + ... |
| `Pontryagin::optimal_control(R⁻¹, B^T, λ) → DVector` | u* = −R⁻¹B^Tλ |
| `PontryaginResult` | s_trajectory (Riccati), k_trajectory (gains), x_trajectory, u_trajectory, cost |

## How It Works

The library is structured around the standard control theory pipeline:

1. **`state_space`** — Foundation. The state-space model ẋ = Ax + Bu, y = Cx + Du captures LTI dynamics in matrix form. All dimensions are validated on construction. Simulation uses forward Euler integration.

2. **`controllability`** — The controllability matrix C = [B, AB, A²B, ...] must have full rank n for the system to be reachable from any initial state to any target. The controllability Gramian Wc (solved via the Lyapunov equation AW + WA^T = −BB^T) quantifies how "easy" it is to control each state direction.

3. **`observability`** — The dual concept: the observability matrix O = [C; CA; CA²; ...] must have full rank. The system is observable iff the dual system (A^T, C^T) is controllable. The observability Gramian quantifies output sensitivity to each state.

4. **`stability`** — Eigenvalue analysis of A: Hurwitz (all Re < 0), Lyapunov (all Re ≤ 0, no defective imaginary eigenvalues), stability margin (distance to imaginary axis). Phase and gain margins from frequency response. Lyapunov functions found by solving A^TP + PA = −Q.

5. **`lqr`** — The Linear Quadratic Regulator minimizes J = ∫(x^TQx + u^TRu) dt by solving the Continuous-time Algebraic Riccati Equation (CARE): A^TP + PA − PBR⁻¹B^TP + Q = 0. The optimal gain is K = R⁻¹B^TP. Smith's iterative doubling method solves the CARE.

6. **`lqg`** — When states aren't directly measurable, LQG combines LQR with a Kalman filter. The Kalman filter gain L is found by solving the dual Riccati equation (using V as process noise and W as measurement noise). The combined controller uses estimated states: u = −Kx̂.

7. **`h_infinity`** — Robust control minimizes the worst-case disturbance amplification ‖G‖∞ = sup_ω σ_max(G(jω)). Mixed-sensitivity design shapes S (sensitivity), T (complementary sensitivity), and KS (control effort) using weighting functions W₁, W₂, W₃. γ-iteration finds the optimal H∞ norm.

8. **`pole_placement`** — Directly assigns closed-loop eigenvalues via state feedback K. Ackermann's formula: K = e_n^T · C⁻¹ · α(A), where α(A) is the desired characteristic polynomial evaluated at A. Bass-Gura provides an alternative formula.

9. **`transfer_function`** — Frequency-domain representation H(s) = num(s)/den(s). Supports Bode plots, step/impulse response, pole/zero computation, stability checking, series/parallel/feedback interconnections, and conversion to state-space.

10. **`pid`** — The workhorse of industrial control. Implements u(t) = Kp·e(t) + Ki·∫e·dt + Kd·de/dt with practical features: output clamping, anti-windup (back-calculation), derivative filter (first-order low-pass), and setpoint weighting. Auto-tuning via Ziegler-Nichols and Cohen-Coon methods.

11. **`pontryagin`** — Pontryagin's Maximum Principle provides necessary conditions for optimality: the Hamiltonian H = x^TQx + u^TRu + λ^T(Ax+Bu) must be minimized pointwise. For LQP, this reduces to the Riccati differential equation integrated backward from S(T) = S_final.

## The Math

### State-Space
ẋ(t) = Ax(t) + Bu(t), y(t) = Cx(t) + Du(t). The transfer matrix is G(s) = C(sI − A)⁻¹B + D.

### Controllability
The system is controllable iff rank[B, AB, ..., Aⁿ⁻¹B] = n. The controllability Gramian Wc = ∫₀^∞ e^{At}BB^Te^{A^Tt}dt satisfies the Lyapunov equation AWc + WcA^T = −BB^T.

### LQR
Minimize J = ∫₀^∞ (x^TQx + u^TRu) dt. Solution: solve CARE A^TP + PA − PBR⁻¹B^TP + Q = 0, then K = R⁻¹B^TP. Closed-loop: ẋ = (A − BK)x.

### LQG
LQR + Kalman filter. Observer: ẋ̂ = Ax̂ + Bu + L(y − ŷ). Kalman gain L = PC^TW⁻¹ where P solves the filter Riccati AP + PA^T − PC^TW⁻¹CP + V = 0. Separation principle guarantees stability.

### H∞
‖G‖∞ = sup_ω σ_max(G(jω)). Mixed-sensitivity minimizes ‖[W₁S, W₂KS, W₃T]‖∞. The γ-iteration bisects on the achievable H∞ norm.

### Pole Placement
Given desired eigenvalues {λ₁, ..., λₙ}, Ackermann's formula gives K = e_n^T · C⁻¹ · α(A) where α(s) = Π(s − λᵢ).

### PID
u(t) = Kp·e(t) + Ki·∫₀^t e(τ)dτ + Kd·de/dt. Transfer function: C(s) = Kp + Ki/s + Kd·s. Ziegler-Nichols tuning: set Ki = 0, Kd = 0, increase Kp until sustained oscillation at Ku, period Tu, then Kp = 0.6Ku, Ki = 2Kp/Tu, Kd = KpTu/8.

### Pontryagin
For minimize J = φ(x(T)) + ∫₀^T L(x,u)dt subject to ẋ = f(x,u): define H = L + λ^Tf. Necessary conditions: ẋ = ∂H/∂λ, λ̇ = −∂H/∂x, u* = argmin H, λ(T) = ∂φ/∂x. For LQP: reduces to the Riccati differential equation Ṡ = −A^TS − SA + SBR⁻¹B^TS − Q.

## License

MIT
