//! # lau-control-theory-agents
//!
//! Classical and modern control theory for agents — stability, controllability,
//! observability, optimal control.
//!
//! This crate provides tools for designing agent controllers that are stable,
//! robust, and optimal using well-established control-theoretic principles.

pub mod state_space;
pub mod controllability;
pub mod observability;
pub mod stability;
pub mod lqr;
pub mod lqg;
pub mod h_infinity;
pub mod pole_placement;
pub mod transfer_function;
pub mod pid;
pub mod pontryagin;

pub use state_space::StateSpace;
pub use controllability::Controllability;
pub use observability::Observability;
pub use stability::Stability;
pub use lqr::Lqr;
pub use lqg::Lqg;
pub use h_infinity::HInfinity;
pub use pole_placement::PolePlacement;
pub use transfer_function::TransferFunction;
pub use pid::PidController;
pub use pontryagin::Pontryagin;
