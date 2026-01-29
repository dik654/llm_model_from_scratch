//! GPT 모델 (Decoder-only Transformer)
//!
//! 텍스트 생성을 위한 자기회귀(Auto-regressive) 언어 모델입니다.
//!
//! # 전체 구조
//! ```text
//!     토큰 ID 입력
//!          ↓
//!   Token Embedding
//!          +
//!   Position Embedding
//!          ↓
//!       Dropout
//!          ↓
//!   ┌─────────────┐
//!   │ Transformer │
//!   │   Block     │ × N
//!   └─────────────┘
//!          ↓
//!     Layer Norm
//!          ↓
//!     LM Head (Linear)
//!          ↓
//!     Logits (vocab_size)
//! ```
//!
//! # 언어 모델링
//! - 입력: [t₁, t₂, ..., tₙ]
//! - 출력: [t₂, t₃, ..., tₙ₊₁]의 확률 분포
//! - 목표: P(tᵢ | t₁, ..., tᵢ₋₁) 학습

use crate::tensor::{
    Tensor2D, Tensor3D,
    sinusoidal_positional_encoding, zeros_3d, softmax_2d,
};
use crate::nn::{Embedding, LayerNorm, Linear, Dropout, Module};
use super::TransformerBlocks;
use ndarray::s;
use rand::Rng;

/// GPT 모델 설정
#[derive(Clone, Debug)]
pub struct GPTConfig {
    /// 어휘 크기
    pub vocab_size: usize,
    /// 최대 시퀀스 길이
    pub max_seq_len: usize,
    /// 모델 차원 (d_model)
    pub d_model: usize,
    /// Attention Head 개수
    pub num_heads: usize,
    /// Transformer Block 개수
    pub num_layers: usize,
    /// FFN 내부 차원 (보통 d_model * 4)
    pub d_ff: usize,
    /// 드롭아웃 확률
    pub dropout: f32,
}

impl GPTConfig {
    /// 미니 GPT 설정 (테스트/데모용)
    pub fn mini() -> Self {
        Self {
            vocab_size: 256,       // ASCII
            max_seq_len: 128,
            d_model: 64,
            num_heads: 4,
            num_layers: 2,
            d_ff: 256,
            dropout: 0.1,
        }
    }

    /// 작은 GPT 설정
    pub fn small() -> Self {
        Self {
            vocab_size: 50257,     // GPT-2 토크나이저
            max_seq_len: 256,
            d_model: 384,
            num_heads: 6,
            num_layers: 6,
            d_ff: 1536,
            dropout: 0.1,
        }
    }

    /// 중간 GPT 설정
    pub fn medium() -> Self {
        Self {
            vocab_size: 50257,
            max_seq_len: 512,
            d_model: 768,
            num_heads: 12,
            num_layers: 12,
            d_ff: 3072,
            dropout: 0.1,
        }
    }
}

/// GPT 모델
///
/// # 예시
/// ```
/// let config = GPTConfig::mini();
/// let model = GPT::new(config);
///
/// // 순전파
/// let token_ids = vec![vec![1, 2, 3, 4, 5]];
/// let logits = model.forward(&token_ids);  // (1, 5, vocab_size)
///
/// // 텍스트 생성
/// let generated = model.generate(&[1, 2], 10, 1.0);
/// ```
#[derive(Clone)]
pub struct GPT {
    /// 설정
    pub config: GPTConfig,
    /// 토큰 임베딩
    pub token_embedding: Embedding,
    /// 위치 인코딩 (Sinusoidal)
    pub position_encoding: Tensor2D,
    /// 임베딩 드롭아웃
    pub embed_dropout: Dropout,
    /// Transformer 블록 스택
    pub transformer: TransformerBlocks,
    /// 최종 Layer Normalization
    pub ln_f: LayerNorm,
    /// 언어 모델 헤드 (d_model → vocab_size)
    pub lm_head: Linear,
}

impl GPT {
    /// 새로운 GPT 모델 생성
    pub fn new(config: GPTConfig) -> Self {
        // 토큰 임베딩
        let token_embedding = Embedding::new(config.vocab_size, config.d_model);

        // Sinusoidal 위치 인코딩
        let position_encoding = sinusoidal_positional_encoding(
            config.max_seq_len,
            config.d_model,
        );

        // 임베딩 드롭아웃
        let embed_dropout = Dropout::new(config.dropout);

        // Transformer 블록 스택
        let transformer = TransformerBlocks::new(
            config.num_layers,
            config.d_model,
            config.num_heads,
            config.d_ff,
            config.max_seq_len,
            config.dropout,
        );

        // 최종 Layer Norm
        let ln_f = LayerNorm::new(config.d_model);

        // LM Head (가중치 타이잉은 생략 - 교육 목적으로 단순하게)
        let lm_head = Linear::new(config.d_model, config.vocab_size);

        Self {
            config,
            token_embedding,
            position_encoding,
            embed_dropout,
            transformer,
            ln_f,
            lm_head,
        }
    }

    /// 학습/추론 모드 설정
    pub fn set_training(&mut self, training: bool) {
        self.embed_dropout.set_training(training);
        self.transformer.set_training(training);
    }

    /// 순전파: 토큰 ID → Logits
    ///
    /// # 동작 과정
    /// 1. 토큰 임베딩 조회
    /// 2. 위치 인코딩 추가
    /// 3. 드롭아웃 적용
    /// 4. Transformer 블록 통과
    /// 5. 최종 Layer Norm
    /// 6. LM Head로 로짓 계산
    pub fn forward(&self, token_ids: &[Vec<usize>]) -> Tensor3D {
        let batch_size = token_ids.len();
        let seq_len = token_ids[0].len();

        // 1. 토큰 임베딩
        let tok_emb = self.token_embedding.forward(token_ids);

        // 2. 위치 인코딩 추가
        let pos_enc = self.position_encoding.slice(s![..seq_len, ..]).to_owned();

        // 브로드캐스팅: (seq_len, d_model) → (batch, seq_len, d_model)
        let mut x = tok_emb.clone();
        for b in 0..batch_size {
            for s in 0..seq_len {
                for d in 0..self.config.d_model {
                    x[[b, s, d]] += pos_enc[[s, d]];
                }
            }
        }

        // 3. 드롭아웃
        let x = self.embed_dropout.forward(&x);

        // 4. Transformer 블록
        let x = self.transformer.forward(&x);

        // 5. 최종 Layer Norm
        let x = self.ln_f.forward(&x);

        // 6. LM Head
        self.lm_head.forward(&x)
    }

    /// 텍스트 생성 (Auto-regressive)
    ///
    /// # 인자
    /// - `start_tokens`: 시작 토큰 ID들
    /// - `max_new_tokens`: 생성할 최대 토큰 수
    /// - `temperature`: 샘플링 온도 (0.0~2.0)
    ///   - 낮을수록: 더 결정적 (greedy에 가까움)
    ///   - 높을수록: 더 다양함 (랜덤에 가까움)
    ///
    /// # 반환
    /// 생성된 전체 토큰 시퀀스
    pub fn generate(
        &self,
        start_tokens: &[usize],
        max_new_tokens: usize,
        temperature: f32,
    ) -> Vec<usize> {
        let mut rng = rand::thread_rng();
        let mut tokens = start_tokens.to_vec();

        for _ in 0..max_new_tokens {
            // 최대 길이 제한
            let context_len = tokens.len().min(self.config.max_seq_len);
            let context = &tokens[tokens.len() - context_len..];

            // 순전파
            let input = vec![context.to_vec()];
            let logits = self.forward(&input);

            // 마지막 위치의 로짓만 사용
            let seq_len = logits.shape()[1];
            let last_logits: Vec<f32> = (0..self.config.vocab_size)
                .map(|v| logits[[0, seq_len - 1, v]] / temperature)
                .collect();

            // Softmax로 확률 분포 계산
            let probs = {
                let max_val = last_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_vals: Vec<f32> = last_logits.iter().map(|&x| (x - max_val).exp()).collect();
                let sum: f32 = exp_vals.iter().sum();
                exp_vals.iter().map(|&x| x / sum).collect::<Vec<f32>>()
            };

            // 확률에 따라 샘플링
            let next_token = sample_from_distribution(&probs, &mut rng);
            tokens.push(next_token);
        }

        tokens
    }

    /// Greedy 디코딩 (가장 확률 높은 토큰 선택)
    pub fn generate_greedy(
        &self,
        start_tokens: &[usize],
        max_new_tokens: usize,
    ) -> Vec<usize> {
        let mut tokens = start_tokens.to_vec();

        for _ in 0..max_new_tokens {
            let context_len = tokens.len().min(self.config.max_seq_len);
            let context = &tokens[tokens.len() - context_len..];

            let input = vec![context.to_vec()];
            let logits = self.forward(&input);

            let seq_len = logits.shape()[1];

            // argmax
            let mut max_idx = 0;
            let mut max_val = f32::NEG_INFINITY;
            for v in 0..self.config.vocab_size {
                let val = logits[[0, seq_len - 1, v]];
                if val > max_val {
                    max_val = val;
                    max_idx = v;
                }
            }

            tokens.push(max_idx);
        }

        tokens
    }

    /// 파라미터 개수 반환
    pub fn num_parameters(&self) -> usize {
        self.token_embedding.num_parameters()
            + self.transformer.num_parameters()
            + self.ln_f.num_parameters()
            + self.lm_head.num_parameters()
    }

    /// 모델 요약 정보 출력
    pub fn summary(&self) -> String {
        let total_params = self.num_parameters();
        let embed_params = self.token_embedding.num_parameters();
        let transformer_params = self.transformer.num_parameters();
        let head_params = self.lm_head.num_parameters();

        format!(
            r#"GPT Model Summary
================
Config:
  vocab_size: {}
  max_seq_len: {}
  d_model: {}
  num_heads: {}
  num_layers: {}
  d_ff: {}
  dropout: {}

Parameters:
  Token Embedding: {:>12} ({:.1}%)
  Transformer:     {:>12} ({:.1}%)
  LM Head:         {:>12} ({:.1}%)
  --------------------------
  Total:           {:>12}

Memory (fp32):     {:>12.2} MB
"#,
            self.config.vocab_size,
            self.config.max_seq_len,
            self.config.d_model,
            self.config.num_heads,
            self.config.num_layers,
            self.config.d_ff,
            self.config.dropout,
            embed_params, (embed_params as f64 / total_params as f64) * 100.0,
            transformer_params, (transformer_params as f64 / total_params as f64) * 100.0,
            head_params, (head_params as f64 / total_params as f64) * 100.0,
            total_params,
            (total_params * 4) as f64 / (1024.0 * 1024.0),
        )
    }
}

/// 확률 분포에서 샘플링
fn sample_from_distribution<R: Rng>(probs: &[f32], rng: &mut R) -> usize {
    let r: f32 = rng.gen();
    let mut cumsum = 0.0;

    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i;
        }
    }

    probs.len() - 1
}

/// Top-k 샘플링
pub fn sample_top_k(logits: &[f32], k: usize, temperature: f32) -> usize {
    let mut rng = rand::thread_rng();

    // 로짓을 인덱스와 함께 정렬
    let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Top-k만 선택
    let top_k: Vec<(usize, f32)> = indexed.into_iter().take(k).collect();

    // Temperature 적용 및 softmax
    let max_val = top_k.iter().map(|(_, v)| *v).fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = top_k.iter().map(|(_, v)| ((v - max_val) / temperature).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    let probs: Vec<f32> = exp_vals.iter().map(|&x| x / sum).collect();

    // 샘플링
    let r: f32 = rng.gen();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return top_k[i].0;
        }
    }

    top_k[k - 1].0
}

/// Top-p (Nucleus) 샘플링
pub fn sample_top_p(logits: &[f32], p: f32, temperature: f32) -> usize {
    let mut rng = rand::thread_rng();

    // Temperature 적용
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();

    // Softmax
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    let probs: Vec<f32> = exp_vals.iter().map(|&x| x / sum).collect();

    // 인덱스와 함께 정렬
    let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // 누적 확률이 p를 넘을 때까지 선택
    let mut cumsum = 0.0;
    let mut selected: Vec<(usize, f32)> = Vec::new();

    for (idx, prob) in indexed {
        cumsum += prob;
        selected.push((idx, prob));
        if cumsum >= p {
            break;
        }
    }

    // 선택된 토큰 중에서 재정규화 후 샘플링
    let total: f32 = selected.iter().map(|(_, p)| p).sum();
    let r: f32 = rng.gen::<f32>() * total;
    let mut cumsum = 0.0;

    for (idx, prob) in &selected {
        cumsum += prob;
        if r < cumsum {
            return *idx;
        }
    }

    selected.last().unwrap().0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpt_forward_shape() {
        let config = GPTConfig::mini();
        let model = GPT::new(config.clone());

        let token_ids = vec![vec![1, 2, 3, 4, 5]];
        let logits = model.forward(&token_ids);

        assert_eq!(logits.shape(), &[1, 5, config.vocab_size]);
    }

    #[test]
    fn test_gpt_generate() {
        let config = GPTConfig::mini();
        let model = GPT::new(config);

        let start = vec![1, 2, 3];
        let generated = model.generate(&start, 5, 1.0);

        assert_eq!(generated.len(), 8); // 3 + 5
    }

    #[test]
    fn test_gpt_summary() {
        let config = GPTConfig::mini();
        let model = GPT::new(config);

        let summary = model.summary();
        assert!(summary.contains("GPT Model Summary"));
        assert!(summary.contains("Total:"));
    }

    #[test]
    fn test_sample_top_k() {
        let logits = vec![10.0, 5.0, 1.0, 0.1];
        let sampled = sample_top_k(&logits, 2, 1.0);
        assert!(sampled == 0 || sampled == 1);
    }

    #[test]
    fn test_sample_top_p() {
        let logits = vec![10.0, 5.0, 1.0, 0.1];
        let sampled = sample_top_p(&logits, 0.9, 1.0);
        // 대부분 상위 확률 토큰이 선택되어야 함
        assert!(sampled <= 2);
    }
}
