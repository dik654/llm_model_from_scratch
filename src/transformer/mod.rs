//! Transformer 아키텍처 모듈
//!
//! "Attention Is All You Need" (Vaswani et al., 2017)를 기반으로 한
//! Transformer 구현입니다.
//!
//! # 핵심 구성 요소
//! - `ScaledDotProductAttention`: 기본 어텐션 메커니즘
//! - `MultiHeadAttention`: 여러 관점에서의 어텐션
//! - `TransformerBlock`: 어텐션 + FFN + 정규화
//! - `GPT`: 디코더 전용 언어 모델

pub mod attention;
pub mod block;
pub mod model;
pub mod feed_forward;

pub use attention::{ScaledDotProductAttention, MultiHeadAttention, CausalSelfAttention};
pub use block::{TransformerBlock, TransformerBlocks};
pub use model::{GPT, GPTConfig};
pub use feed_forward::FeedForward;
