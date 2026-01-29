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
    use approx::assert_relative_eq;

    // ==================== 기본 순전파 테스트 ====================

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
        let _target = vec![tokens[1..].to_vec()];
        // (첫 토큰으로 두 번째 예측, ...)

        println!("Input tokens: {:?}", tokens);
        println!("Logits shape: {:?}", logits.shape());
    }

    // ==================== 텐서 연산 테스트 ====================

    #[test]
    fn test_tensor_matmul_correctness() {
        // 2x3 @ 3x2 = 2x2
        let a = ndarray::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = ndarray::array![[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]];
        let c = matmul_2d(&a, &b);

        // 수동 계산: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        //          [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        assert_relative_eq!(c[[0, 0]], 58.0, epsilon = 1e-5);
        assert_relative_eq!(c[[0, 1]], 64.0, epsilon = 1e-5);
        assert_relative_eq!(c[[1, 0]], 139.0, epsilon = 1e-5);
        assert_relative_eq!(c[[1, 1]], 154.0, epsilon = 1e-5);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = ndarray::array![1.0, 2.0, 3.0, 4.0];
        let probs = softmax_1d(&logits);
        let sum: f32 = probs.sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_softmax_preserves_order() {
        let logits = ndarray::array![1.0, 3.0, 2.0];
        let probs = softmax_1d(&logits);

        // 가장 큰 로짓 → 가장 큰 확률
        assert!(probs[1] > probs[2]);
        assert!(probs[2] > probs[0]);
    }

    #[test]
    fn test_gelu_activation() {
        let x = randn_3d(2, 3, 4);
        let y = gelu(&x);

        // GELU(0) ≈ 0
        assert_relative_eq!(gelu(&ndarray::array![[[0.0]]])[[0, 0, 0]], 0.0, epsilon = 1e-3);

        // Shape 보존
        assert_eq!(y.shape(), x.shape());
    }

    #[test]
    fn test_layer_norm_statistics() {
        let x = randn_3d(2, 3, 64);
        let gamma = ones_1d(64);
        let beta = zeros_1d(64);
        let y = layer_norm(&x, &gamma, &beta, 1e-5);

        // 정규화 후 각 시퀀스 위치의 평균은 0에 가깝고 분산은 1에 가까워야 함
        for b in 0..2 {
            for s in 0..3 {
                let slice: Vec<f32> = (0..64).map(|d| y[[b, s, d]]).collect();
                let mean: f32 = slice.iter().sum::<f32>() / 64.0;
                let var: f32 = slice.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / 64.0;

                assert_relative_eq!(mean, 0.0, epsilon = 1e-4);
                assert_relative_eq!(var, 1.0, epsilon = 1e-4);
            }
        }
    }

    // ==================== 멀티배치 테스트 ====================

    #[test]
    fn test_multi_batch_forward() {
        let config = GPTConfig::mini();
        let model = GPT::new(config.clone());

        // 배치 크기 4
        let tokens = vec![
            vec![1, 2, 3, 4, 5],
            vec![6, 7, 8, 9, 10],
            vec![11, 12, 13, 14, 15],
            vec![16, 17, 18, 19, 20],
        ];
        let logits = model.forward(&tokens);

        assert_eq!(logits.shape(), &[4, 5, config.vocab_size]);
    }

    #[test]
    fn test_different_sequence_lengths() {
        let config = GPTConfig::mini();
        let model = GPT::new(config.clone());

        // 시퀀스 길이 3
        let tokens_short = vec![vec![1, 2, 3]];
        let logits_short = model.forward(&tokens_short);
        assert_eq!(logits_short.shape(), &[1, 3, config.vocab_size]);

        // 시퀀스 길이 10
        let tokens_long = vec![vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]];
        let logits_long = model.forward(&tokens_long);
        assert_eq!(logits_long.shape(), &[1, 10, config.vocab_size]);
    }

    // ==================== 토크나이저 테스트 ====================

    #[test]
    fn test_char_tokenizer_roundtrip() {
        let tokenizer = CharTokenizer::new();
        let text = "Hello, World! 123";
        let tokens = tokenizer.encode(text);
        let decoded = tokenizer.decode(&tokens);
        assert_eq!(text, decoded);
    }

    #[test]
    fn test_simple_tokenizer_roundtrip() {
        let text = "the quick brown fox jumps";
        let tokenizer = SimpleTokenizer::from_text(text, 1);
        let tokens = tokenizer.encode(text);
        let decoded = tokenizer.decode(&tokens);
        assert_eq!(text, decoded);
    }

    #[test]
    fn test_bpe_tokenizer_roundtrip() {
        let corpus = "hello world hello there world peace";
        let tokenizer = BPETokenizer::train(corpus, 300);

        let text = "hello";
        let tokens = tokenizer.encode(text);
        let decoded = tokenizer.decode(&tokens);
        assert_eq!(text, decoded);
    }

    // ==================== 손실 함수 테스트 ====================

    #[test]
    fn test_cross_entropy_perfect_match() {
        // 완벽하게 예측하면 손실이 0에 가까워야 함
        let mut logits = zeros_3d(1, 1, 10);
        logits[[0, 0, 5]] = 100.0;  // 타겟 클래스에 매우 높은 값

        let targets = vec![vec![5usize]];
        let loss = cross_entropy_loss(&logits, &targets);

        assert!(loss < 1e-3, "Loss should be near 0 for perfect prediction, got {}", loss);
    }

    #[test]
    fn test_cross_entropy_uniform() {
        // 균등 분포면 손실이 ln(vocab_size)에 가까워야 함
        let vocab_size = 10;
        let logits = zeros_3d(1, 1, vocab_size);  // 모두 0 → 균등 분포

        let targets = vec![vec![5usize]];
        let loss = cross_entropy_loss(&logits, &targets);

        let expected_loss = (vocab_size as f32).ln();
        assert_relative_eq!(loss, expected_loss, epsilon = 1e-4);
    }

    #[test]
    fn test_perplexity_calculation() {
        // perplexity = exp(loss)
        let loss = 2.0;
        let ppl = perplexity(loss);
        assert_relative_eq!(ppl, loss.exp(), epsilon = 1e-5);
    }

    // ==================== 옵티마이저 테스트 ====================

    #[test]
    fn test_sgd_parameter_update() {
        let mut param = ndarray::Array2::ones((3, 3));
        let grad = ndarray::Array2::from_elem((3, 3), 0.1);

        let mut sgd = SGD::new(0.1, 0.0);
        sgd.step_2d(&mut param, &grad);

        // param = 1.0 - 0.1 * 0.1 = 0.99
        assert_relative_eq!(param[[0, 0]], 0.99, epsilon = 1e-6);
    }

    #[test]
    fn test_adam_parameter_update() {
        let mut param = ndarray::Array2::ones((3, 3));
        let grad = ndarray::Array2::from_elem((3, 3), 0.1);

        let mut adam = Adam::new(0.001);
        adam.step_2d(&mut param, &grad, 0);

        // 첫 스텝 후 파라미터가 감소해야 함
        assert!(param[[0, 0]] < 1.0);
    }

    // ==================== 스케줄러 테스트 ====================

    #[test]
    fn test_warmup_cosine_scheduler() {
        let scheduler = WarmupCosineDecay::new(0.001, 10, 100);

        // Warmup 시작
        assert_relative_eq!(scheduler.get_lr(0), 0.0, epsilon = 1e-6);

        // Warmup 중간
        let lr_5 = scheduler.get_lr(5);
        assert!(lr_5 > 0.0 && lr_5 < 0.001);

        // Warmup 끝
        assert_relative_eq!(scheduler.get_lr(10), 0.001, epsilon = 1e-6);

        // Decay 진행
        let lr_50 = scheduler.get_lr(50);
        assert!(lr_50 < 0.001 && lr_50 > 0.0);

        // 거의 끝
        let lr_99 = scheduler.get_lr(99);
        assert!(lr_99 < lr_50);
    }

    // ==================== Attention 마스크 테스트 ====================

    #[test]
    fn test_causal_mask_shape() {
        let mask = create_causal_mask(5);
        assert_eq!(mask.shape(), &[5, 5]);
    }

    #[test]
    fn test_causal_mask_values() {
        let mask = create_causal_mask(4);

        // 하삼각 행렬이어야 함 (대각선 포함)
        // 첫 행: [1, 0, 0, 0]
        // 둘째 행: [1, 1, 0, 0]
        // ...
        for i in 0..4 {
            for j in 0..4 {
                if j <= i {
                    assert_eq!(mask[[i, j]], 1.0, "mask[{},{}] should be 1", i, j);
                } else {
                    assert_eq!(mask[[i, j]], 0.0, "mask[{},{}] should be 0", i, j);
                }
            }
        }
    }

    // ==================== Positional Encoding 테스트 ====================

    #[test]
    fn test_positional_encoding_shape() {
        let pos_enc = sinusoidal_positional_encoding(100, 64);
        assert_eq!(pos_enc.shape(), &[100, 64]);
    }

    #[test]
    fn test_positional_encoding_uniqueness() {
        let pos_enc = sinusoidal_positional_encoding(10, 64);

        // 각 위치의 인코딩이 유일해야 함
        for i in 0..10 {
            for j in (i+1)..10 {
                let pos_i: Vec<f32> = (0..64).map(|d| pos_enc[[i, d]]).collect();
                let pos_j: Vec<f32> = (0..64).map(|d| pos_enc[[j, d]]).collect();

                // 두 위치가 다르면 인코딩도 달라야 함
                assert_ne!(pos_i, pos_j, "Position {} and {} should have different encodings", i, j);
            }
        }
    }

    // ==================== 모델 일관성 테스트 ====================

    #[test]
    fn test_model_determinism() {
        let config = GPTConfig::mini();

        // 모델 생성 및 추론 모드로 설정 (dropout 비활성화)
        let mut model1 = GPT::new(config.clone());
        model1.set_training(false);

        // 같은 입력에 대해 같은 출력
        let tokens = vec![vec![1, 2, 3]];
        let logits1 = model1.forward(&tokens);

        // 같은 모델은 같은 입력에 같은 출력 (추론 모드)
        let logits1_again = model1.forward(&tokens);
        assert_eq!(logits1, logits1_again);
    }

    #[test]
    fn test_greedy_vs_sampling() {
        let config = GPTConfig::mini();
        let mut model = GPT::new(config);
        model.set_training(false);  // 추론 모드로 설정

        let start = vec![1, 2, 3];

        // Greedy 디코딩은 항상 같은 결과
        let greedy1 = model.generate_greedy(&start, 5);
        let greedy2 = model.generate_greedy(&start, 5);
        assert_eq!(greedy1, greedy2);

        // 시작 토큰은 항상 동일
        assert_eq!(&greedy1[..3], &start);
    }

    // ==================== 학습 모드 테스트 ====================

    #[test]
    fn test_training_mode_dropout() {
        let config = GPTConfig::mini();
        let mut model = GPT::new(config);

        let tokens = vec![vec![1, 2, 3, 4, 5]];

        // 추론 모드
        model.set_training(false);
        let logits_inference = model.forward(&tokens);
        let logits_inference2 = model.forward(&tokens);
        assert_eq!(logits_inference, logits_inference2);  // 동일

        // 학습 모드 (dropout 적용)
        model.set_training(true);
        let logits_train1 = model.forward(&tokens);
        let logits_train2 = model.forward(&tokens);
        // Dropout으로 인해 다를 수 있음 (하지만 확률적으로 테스트하기 어려움)
        // 대신 형태만 확인
        assert_eq!(logits_train1.shape(), logits_train2.shape());
    }
}
