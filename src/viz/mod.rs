//! 시각화 모듈
//!
//! Transformer의 동작을 시각화하여 이해를 돕습니다.
//!
//! # 지원 시각화
//! - Attention 패턴
//! - 텐서 형태 흐름
//! - 학습 진행 상황
//! - 토큰화 과정

pub mod terminal;
pub mod attention;
pub mod training;

pub use terminal::TerminalViz;
pub use attention::{AttentionViz, AttentionData, HeadAttention};
pub use training::{TrainingViz, TrainingStats, TrainingEvent};
