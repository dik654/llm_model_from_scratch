//! Transformer Block
//!
//! Transformer의 기본 구성 단위입니다.
//! 여러 블록을 쌓아서 깊은 모델을 만듭니다.
//!
//! # 구조 (Pre-LN, GPT-2 스타일)
//! ```text
//!          입력 x
//!             ↓
//!     ┌───────┴───────┐
//!     │               │
//!     ↓               │
//!  LayerNorm          │
//!     ↓               │
//!  Self-Attention     │
//!     ↓               │
//!     └───────+───────┘  ← Add (Residual)
//!             ↓
//!     ┌───────┴───────┐
//!     │               │
//!     ↓               │
//!  LayerNorm          │
//!     ↓               │
//!  Feed-Forward       │
//!     ↓               │
//!     └───────+───────┘  ← Add (Residual)
//!             ↓
//!           출력
//! ```
//!
//! # Pre-LN vs Post-LN
//! - **Post-LN** (원본 Transformer): Attention → Add → LN
//! - **Pre-LN** (GPT-2+): LN → Attention → Add
//!
//! Pre-LN의 장점:
//! - 더 안정적인 학습
//! - Warmup이 덜 필요
//! - 더 깊은 모델 가능
//!
//! # Residual Connection
//! y = x + F(x)
//!
//! 왜 중요한가?
//! - Gradient가 직접 흐르는 경로 제공
//! - ∂y/∂x = ∂F(x)/∂x + 1 ← 최소 1의 기울기 보장
//! - 깊은 네트워크 학습 가능

use crate::tensor::{Tensor3D, add};
use crate::nn::{LayerNorm, Dropout, Module};
use super::{CausalSelfAttention, FeedForward};

/// Transformer Block (Pre-LN)
///
/// # 예시
/// ```
/// let block = TransformerBlock::new(512, 8, 2048, 256, 0.1);
/// let x = randn_3d(32, 10, 512);  // (batch, seq, d_model)
/// let y = block.forward(&x);      // (batch, seq, d_model)
/// ```
#[derive(Clone)]
pub struct TransformerBlock {
    /// 첫 번째 Layer Normalization (Attention 전)
    pub ln1: LayerNorm,
    /// Causal Self-Attention
    pub attention: CausalSelfAttention,
    /// 두 번째 Layer Normalization (FFN 전)
    pub ln2: LayerNorm,
    /// Feed-Forward Network
    pub ffn: FeedForward,
    /// Residual Dropout
    pub dropout: Dropout,
    /// 모델 차원
    pub d_model: usize,
}

impl TransformerBlock {
    /// 새로운 Transformer Block 생성
    ///
    /// # 인자
    /// - `d_model`: 모델 차원 (예: 512)
    /// - `num_heads`: Attention Head 개수 (예: 8)
    /// - `d_ff`: FFN 내부 차원 (예: 2048)
    /// - `max_seq_len`: 최대 시퀀스 길이
    /// - `dropout_p`: 드롭아웃 확률
    pub fn new(
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        max_seq_len: usize,
        dropout_p: f32,
    ) -> Self {
        Self {
            ln1: LayerNorm::new(d_model),
            attention: CausalSelfAttention::new(d_model, num_heads, max_seq_len, dropout_p),
            ln2: LayerNorm::new(d_model),
            ffn: FeedForward::new(d_model, d_ff, dropout_p),
            dropout: Dropout::new(dropout_p),
            d_model,
        }
    }

    /// 학습/추론 모드 설정
    pub fn set_training(&mut self, training: bool) {
        self.dropout.set_training(training);
        self.ffn.set_training(training);
    }

    /// 파라미터 개수 반환
    pub fn num_parameters(&self) -> usize {
        self.ln1.num_parameters()
            + self.attention.num_parameters()
            + self.ln2.num_parameters()
            + self.ffn.num_parameters()
    }
}

impl Module for TransformerBlock {
    /// Transformer Block 순전파 (Pre-LN)
    ///
    /// # 동작 과정
    /// ```text
    /// 1. x_norm = LayerNorm(x)
    /// 2. attn_out = SelfAttention(x_norm)
    /// 3. x = x + Dropout(attn_out)     ← Residual 1
    /// 4. x_norm = LayerNorm(x)
    /// 5. ffn_out = FFN(x_norm)
    /// 6. x = x + Dropout(ffn_out)      ← Residual 2
    /// ```
    fn forward(&self, x: &Tensor3D) -> Tensor3D {
        // ===== Attention 서브레이어 =====
        // 1. Layer Normalization
        let x_norm = self.ln1.forward(x);

        // 2. Self-Attention
        let attn_out = self.attention.forward(&x_norm);

        // 3. Residual Connection + Dropout
        let attn_out = self.dropout.forward(&attn_out);
        let x = add(x, &attn_out);

        // ===== FFN 서브레이어 =====
        // 4. Layer Normalization
        let x_norm = self.ln2.forward(&x);

        // 5. Feed-Forward Network
        let ffn_out = self.ffn.forward(&x_norm);

        // 6. Residual Connection + Dropout
        let ffn_out = self.dropout.forward(&ffn_out);
        add(&x, &ffn_out)
    }

    fn num_parameters(&self) -> usize {
        self.ln1.num_parameters()
            + self.attention.num_parameters()
            + self.ln2.num_parameters()
            + self.ffn.num_parameters()
    }
}

/// Transformer Block 스택
///
/// 여러 Transformer Block을 순차적으로 쌓습니다.
#[derive(Clone)]
pub struct TransformerBlocks {
    /// 블록 리스트
    pub blocks: Vec<TransformerBlock>,
}

impl TransformerBlocks {
    /// 새로운 Transformer 스택 생성
    pub fn new(
        num_layers: usize,
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        max_seq_len: usize,
        dropout_p: f32,
    ) -> Self {
        let blocks = (0..num_layers)
            .map(|_| TransformerBlock::new(d_model, num_heads, d_ff, max_seq_len, dropout_p))
            .collect();

        Self { blocks }
    }

    /// 학습/추론 모드 설정
    pub fn set_training(&mut self, training: bool) {
        for block in &mut self.blocks {
            block.set_training(training);
        }
    }

    /// 파라미터 개수 반환
    pub fn num_parameters(&self) -> usize {
        self.blocks.iter().map(|b| b.num_parameters()).sum()
    }
}

impl Module for TransformerBlocks {
    /// 모든 블록을 순차적으로 통과
    fn forward(&self, x: &Tensor3D) -> Tensor3D {
        let mut out = x.clone();
        for block in &self.blocks {
            out = block.forward(&out);
        }
        out
    }

    fn num_parameters(&self) -> usize {
        self.blocks.iter().map(|b| b.num_parameters()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::randn_3d;

    #[test]
    fn test_transformer_block_shape() {
        let block = TransformerBlock::new(64, 8, 256, 100, 0.0);
        let x = randn_3d(2, 10, 64);
        let y = block.forward(&x);

        assert_eq!(y.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_transformer_blocks_shape() {
        let blocks = TransformerBlocks::new(6, 64, 8, 256, 100, 0.0);
        let x = randn_3d(2, 10, 64);
        let y = blocks.forward(&x);

        assert_eq!(y.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_residual_connection() {
        // 입력이 그대로 출력에 영향을 미치는지 확인
        let block = TransformerBlock::new(64, 8, 256, 100, 0.0);

        let x1 = randn_3d(1, 5, 64);
        let x2 = randn_3d(1, 5, 64);

        let y1 = block.forward(&x1);
        let y2 = block.forward(&x2);

        // 다른 입력은 다른 출력을 내야 함
        assert_ne!(y1, y2);
    }
}
