//! # LLM from Scratch (Rust)
//!
//! 교육 목적의 Transformer/LLM 구현입니다.
//!
//! 이 라이브러리는 LLM의 내부 동작을 이해하기 위해
//! 최적화 없이 순수 로직만으로 구현되었습니다.
//!
//! ## 주요 모듈
//!
//! - [`tensor`]: 텐서 연산 (ndarray 기반)
//! - [`nn`]: 신경망 레이어 (Linear, Embedding, LayerNorm 등)
//! - [`transformer`]: Transformer 아키텍처 (Attention, FFN, GPT)
//! - [`autograd`]: 역전파와 손실 함수
//! - [`optim`]: 옵티마이저 (SGD, Adam, AdamW)
//! - [`tokenizer`]: 토크나이저 (Char, BPE)
//! - [`viz`]: 시각화 유틸리티
//!
//! ## 사용 예시
//!
//! ```rust,ignore
//! use llm_from_scratch::prelude::*;
//!
//! // 모델 생성
//! let config = GPTConfig::mini();
//! let model = GPT::new(config);
//!
//! // 토큰화
//! let tokenizer = CharTokenizer::new();
//! let tokens = tokenizer.encode("Hello, world!");
//!
//! // 순전파
//! let logits = model.forward(&[tokens]);
//!
//! // 텍스트 생성
//! let generated = model.generate(&[1, 2, 3], 20, 1.0);
//! ```
//!
//! ## 아키텍처
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │              GPT Model                   │
//! │  ┌─────────────────────────────────────┐│
//! │  │         Token Embedding             ││
//! │  └─────────────────────────────────────┘│
//! │                  +                       │
//! │  ┌─────────────────────────────────────┐│
//! │  │       Position Encoding             ││
//! │  └─────────────────────────────────────┘│
//! │                  ↓                       │
//! │  ┌─────────────────────────────────────┐│
//! │  │      Transformer Block × N          ││
//! │  │  ┌───────────┐  ┌───────────────┐  ││
//! │  │  │ Attention │  │ Feed-Forward  │  ││
//! │  │  └───────────┘  └───────────────┘  ││
//! │  └─────────────────────────────────────┘│
//! │                  ↓                       │
//! │  ┌─────────────────────────────────────┐│
//! │  │           LM Head                   ││
//! │  └─────────────────────────────────────┘│
//! └─────────────────────────────────────────┘
//! ```

pub mod tensor;
pub mod nn;
pub mod transformer;
pub mod autograd;
pub mod optim;
pub mod tokenizer;
pub mod viz;

/// 자주 사용되는 타입들을 모아놓은 prelude
pub mod prelude {
    // Tensor types
    pub use crate::tensor::{Tensor1D, Tensor2D, Tensor3D, Tensor4D};

    // Tensor operations
    pub use crate::tensor::{
        zeros_1d, zeros_2d, zeros_3d,
        ones_1d, ones_2d,
        randn_1d, randn_2d, randn_3d,
        xavier_uniform, he_normal,
        matmul_2d, batched_matmul,
        softmax_1d, softmax_2d, softmax_3d_last,
        relu, gelu, sigmoid, tanh,
        layer_norm,
        create_causal_mask,
        sinusoidal_positional_encoding,
    };

    // Neural network layers
    pub use crate::nn::{Module, Linear, Embedding, LayerNorm, Dropout};

    // Transformer components
    pub use crate::transformer::{
        ScaledDotProductAttention, MultiHeadAttention, CausalSelfAttention,
        TransformerBlock, FeedForward, GPT, GPTConfig,
    };

    // Autograd and loss
    pub use crate::autograd::{
        cross_entropy_loss, perplexity, language_model_loss,
        linear_backward, relu_backward, gelu_backward,
    };

    // Optimizers
    pub use crate::optim::{SGD, Adam, AdamW};
    pub use crate::optim::{
        LRScheduler, ConstantLR, LinearWarmup, CosineAnnealing, WarmupCosineDecay,
    };

    // Tokenizers
    pub use crate::tokenizer::{Tokenizer, SpecialTokens, CharTokenizer, SimpleTokenizer, BPETokenizer};

    // Visualization
    pub use crate::viz::{AttentionViz, TrainingViz};
}

#[cfg(test)]
mod tests {
    use super::prelude::*;

    #[test]
    fn test_simple_forward() {
        let config = GPTConfig::mini();
        let model = GPT::new(config.clone());

        let tokens = vec![vec![1, 2, 3, 4, 5]];
        let logits = model.forward(&tokens);

        assert_eq!(logits.shape(), &[1, 5, config.vocab_size]);
    }

    #[test]
    fn test_text_generation() {
        let config = GPTConfig::mini();
        let model = GPT::new(config);

        let start_tokens = vec![1, 2, 3];
        let generated = model.generate(&start_tokens, 5, 1.0);

        assert_eq!(generated.len(), 8); // 3 + 5
    }

    #[test]
    fn test_full_pipeline() {
        // 1. 토크나이저
        let tokenizer = CharTokenizer::new();
        let text = "Hello";
        let tokens = tokenizer.encode(text);

        // 2. 모델
        let config = GPTConfig::mini();
        let model = GPT::new(config);

        // 3. 순전파
        let logits = model.forward(&[tokens.clone()]);

        // 4. 손실 계산
        let target = vec![tokens[1..].to_vec()];
        // (첫 토큰으로 두 번째 예측, ...)

        println!("Input tokens: {:?}", tokens);
        println!("Logits shape: {:?}", logits.shape());
    }
}
