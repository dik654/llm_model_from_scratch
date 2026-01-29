//! 자동 미분 (Automatic Differentiation) 모듈
//!
//! 역전파(Backpropagation)를 위한 계산 그래프를 구현합니다.
//!
//! # 핵심 개념
//!
//! ## 순전파 (Forward Pass)
//! 입력에서 출력으로 값을 계산합니다.
//! ```text
//! x → f → g → loss
//! ```
//!
//! ## 역전파 (Backward Pass)
//! 출력에서 입력으로 기울기를 전파합니다.
//! ```text
//! ∂loss/∂x ← ∂loss/∂f ← ∂loss/∂g ← 1
//! ```
//!
//! ## Chain Rule (연쇄 법칙)
//! ```text
//! ∂loss/∂x = ∂loss/∂f · ∂f/∂x
//! ```
//!
//! # 구현 전략
//! 교육 목적으로 수동 역전파(manual backprop)를 구현합니다.
//! 실제 프로덕션에서는 자동 미분 라이브러리를 사용합니다.

pub mod backward;
pub mod loss;

pub use backward::*;
pub use loss::*;
