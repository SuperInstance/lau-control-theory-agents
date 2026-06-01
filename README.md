# lau-control-theory-agents

> Classical and modern control theory for agents — stability, controllability, observability, optimal control

## What This Does

Classical and modern control theory for agents — stability, controllability, observability, optimal control. Part of the PLATO/LAU ecosystem — a mathematically rigorous framework for building educational agents that learn, teach, and evolve.

## The Key Idea

This crate implements the core abstractions needed for its domain, with a focus on correctness, composability, and conservation guarantees. Every public type is serializable (serde), every algorithm is tested, and every invariant is verified.

## Install

```bash
cargo add lau-control-theory-agents
```

## Quick Start

See the API Reference below for complete usage. Key entry points:

```rust
use lau_control_theory_agents::*;
// See types and methods below for complete usage
```

## API Reference

```rust
pub struct Stability;
    pub fn is_hurwitz(a: &DMatrix<f64>) -> bool 
    pub fn is_lyapunov_stable(a: &DMatrix<f64>) -> bool 
    pub fn is_asymptotically_stable(a: &DMatrix<f64>) -> bool 
    pub fn exponential_stability_margin(a: &DMatrix<f64>) -> Option<f64> 
    pub fn is_exponentially_stable(a: &DMatrix<f64>) -> bool 
    pub fn lyapunov_test(a: &DMatrix<f64>) -> LyapunovResult 
    pub fn is_positive_definite(m: &DMatrix<f64>) -> bool 
    pub fn damping_ratio(eigenvalue: num_complex::Complex64) -> f64 
    pub fn natural_frequency(eigenvalue: num_complex::Complex64) -> f64 
    pub fn system_stability(sys: &StateSpace) -> SystemStability 
    pub fn is_bibo_stable(sys: &StateSpace) -> bool 
    pub fn stability_margins(_sys: &StateSpace) -> (f64, f64) 
pub struct LyapunovResult 
pub struct SystemStability 
pub struct Pontryagin;
    pub fn solve_lqp(
    pub fn simulate_optimal(
    pub fn hamiltonian(
    pub fn verify_conditions(
    pub fn solve_minimum_time(
pub struct PontryaginResult 
pub struct OptimalTrajectory 
    pub fn total_cost(&self, q: &DMatrix<f64>, r: &DMatrix<f64>, dt: f64) -> f64 
    pub fn final_state(&self) -> &DVector<f64> 
    pub fn final_control(&self) -> &DVector<f64> 
pub struct ConditionCheck 
pub struct MinimumTimeResult 
pub struct Observability;
    pub fn observability_matrix(sys: &StateSpace) -> DMatrix<f64> 
    pub fn is_observable(sys: &StateSpace) -> bool 
    pub fn observability_gramian(sys: &StateSpace) -> DMatrix<f64> 
    pub fn is_observable_dual(sys: &StateSpace) -> bool 
    pub fn observability_index(sys: &StateSpace) -> usize 
    pub fn unobservable_modes(sys: &StateSpace) -> Vec<num_complex::Complex64> 
pub struct Controllability;
    pub fn controllability_matrix(sys: &StateSpace) -> DMatrix<f64> 
    pub fn is_controllable(sys: &StateSpace) -> bool 
    pub fn rank(m: &DMatrix<f64>) -> usize 
    pub fn controllability_gramian(sys: &StateSpace) -> DMatrix<f64> 
    pub fn controllability_index(sys: &StateSpace) -> usize 
pub fn solve_lyapunov(a: &DMatrix<f64>, q: &DMatrix<f64>) -> Option<DMatrix<f64>> 
pub struct Lqg;
    pub fn design(
    pub fn simulate(
pub struct KalmanResult 
pub struct LqgResult 
pub struct LqgSimulation 
pub struct TransferFunction 
    pub fn new(numerator: Vec<f64>, denominator: Vec<f64>) -> Result<Self, String> 
    pub fn gain(k: f64) -> Self 
    pub fn first_order(k: f64, tau: f64) -> Result<Self, String> 
    pub fn second_order(wn: f64, zeta: f64) -> Result<Self, String> 
    pub fn integrator(k: f64) -> Self 
    pub fn differentiator(k: f64) -> Self 
    pub fn delay(t: f64, order: usize) -> Self 
    pub fn evaluate(&self, s: Complex64) -> Complex64 
    pub fn frequency_response(&self, frequencies: &[f64]) -> Vec<(f64, f64)> 
    pub fn bode(&self, freq_range: (f64, f64), num_points: usize) -> BodeData 
    pub fn poles(&self) -> Vec<Complex64> 
```

## How It Works

Read the source in `src/` for full implementation details. All algorithms are documented with inline comments explaining the mathematical foundations.

## The Math

This crate implements formal mathematical constructs. See the source documentation for theorem statements and proofs of correctness.

## Testing

**134 tests** covering construction, serialization, correctness properties, edge cases, and composability with other lau-* crates.

## License

MIT
