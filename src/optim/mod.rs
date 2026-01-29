//! 옵티마이저 (Optimizer) 모듈
//!
//! 기울기를 사용하여 파라미터를 업데이트합니다.
//!
//! # 경사하강법 기본 원리
//! ```text
//! θ_new = θ_old - η · ∇L(θ)
//!
//! θ: 파라미터
//! η: 학습률 (learning rate)
//! ∇L: 손실의 기울기
//! ```
//!
//! # 주요 옵티마이저
//! - SGD: 기본 경사하강법
//! - Adam: 적응적 학습률
//! - AdamW: weight decay 개선

pub mod sgd;
pub mod adam;
pub mod scheduler;

pub use sgd::SGD;
pub use adam::{Adam, AdamW};
pub use scheduler::*;
