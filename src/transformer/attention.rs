//! Self-Attention 메커니즘
//!
//! Transformer의 핵심 연산입니다.
//!
//! # 핵심 아이디어
//! "모든 위치가 모든 위치를 직접 참조한다"
//!
//! 기존 RNN: A → B → C → D (순차적)
//! Attention: A ↔ B ↔ C ↔ D (모두 연결)
//!
//! # Query, Key, Value 비유
//! - Query (Q): "무엇을 찾고 있는가?" - 검색어
//! - Key (K): "무엇이 있는가?" - 책의 색인
//! - Value (V): "실제 내용은?" - 책의 내용
//!
//! # Scaled Dot-Product Attention
//! ```text
//! Attention(Q, K, V) = softmax(QK^T / √dk) · V
//! ```

use crate::tensor::{
    Tensor2D, Tensor3D, Tensor4D,
    reshape_for_multihead, reshape_from_multihead,
    transpose_4d_last2, batched_matmul_4d,
    softmax_4d_last, apply_mask, create_causal_mask,
    zeros_3d,
};
use crate::nn::{Linear, Module};
use ndarray::s;

/// Scaled Dot-Product Attention
///
/// # 수학적 정의
/// ```text
/// Attention(Q, K, V) = softmax(QK^T / √dk) · V
/// ```
///
/// # 단계별 동작
/// 1. QK^T: Query와 Key의 유사도 계산
/// 2. / √dk: 스케일링 (큰 dk에서 softmax 안정화)
/// 3. softmax: 확률 분포로 변환
/// 4. × V: Value의 가중 합
pub struct ScaledDotProductAttention {
    /// 드롭아웃 확률
    pub dropout_p: f32,
    /// 학습 모드
    pub training: bool,
}

impl ScaledDotProductAttention {
    pub fn new(dropout_p: f32) -> Self {
        Self {
            dropout_p,
            training: true,
        }
    }

    /// Attention 계산
    ///
    /// # 인자
    /// - `q`: Query (batch, heads, seq_q, head_dim)
    /// - `k`: Key (batch, heads, seq_k, head_dim)
    /// - `v`: Value (batch, heads, seq_v, head_dim)
    /// - `mask`: 선택적 마스크 (seq_q, seq_k)
    ///
    /// # 반환
    /// - output: (batch, heads, seq_q, head_dim)
    /// - attention_weights: (batch, heads, seq_q, seq_k)
    pub fn forward(
        &self,
        q: &Tensor4D,
        k: &Tensor4D,
        v: &Tensor4D,
        mask: Option<&Tensor2D>,
    ) -> (Tensor4D, Tensor4D) {
        let head_dim = q.shape()[3] as f32;

        // Step 1: QK^T 계산
        // (batch, heads, seq_q, head_dim) @ (batch, heads, head_dim, seq_k)
        // = (batch, heads, seq_q, seq_k)
        let k_t = transpose_4d_last2(k);
        let mut scores = batched_matmul_4d(q, &k_t);

        // Step 2: 스케일링 (√dk로 나눔)
        // 왜? dk가 크면 dot product 값도 커져서 softmax가 극단적이 됨
        let scale = 1.0 / head_dim.sqrt();
        scores.mapv_inplace(|v| v * scale);

        // Step 3: 마스킹 (선택적)
        if let Some(mask) = mask {
            scores = apply_mask(&scores, mask);
        }

        // Step 4: Softmax (확률 분포로 변환)
        let attention_weights = softmax_4d_last(&scores);

        // Step 5: Value의 가중 합
        // (batch, heads, seq_q, seq_k) @ (batch, heads, seq_v, head_dim)
        // = (batch, heads, seq_q, head_dim)
        let output = batched_matmul_4d(&attention_weights, v);

        (output, attention_weights)
    }
}

/// Multi-Head Attention
///
/// # 왜 여러 개의 Head?
/// - 단일 Attention: 하나의 "관점"만 학습
/// - Multi-Head: 여러 관점에서 관계 학습
///   - Head 1: 문법적 관계 (주어-동사)
///   - Head 2: 의미적 관계 (동의어)
///   - Head 3: 위치 관계 (인접 단어)
///   - ...
///
/// # 구조
/// ```text
///        입력 X
///           ↓
///    ┌──────┼──────┐
///    ↓      ↓      ↓
///  Head₁  Head₂  Head₃ ...
///    ↓      ↓      ↓
///    └──────┼──────┘
///        Concat
///           ↓
///      Linear (Wo)
///           ↓
///         출력
/// ```
#[derive(Clone)]
pub struct MultiHeadAttention {
    /// Query 투영 (d_model → d_model)
    pub wq: Linear,
    /// Key 투영 (d_model → d_model)
    pub wk: Linear,
    /// Value 투영 (d_model → d_model)
    pub wv: Linear,
    /// 출력 투영 (d_model → d_model)
    pub wo: Linear,

    /// 모델 차원
    pub d_model: usize,
    /// Head 개수
    pub num_heads: usize,
    /// 각 Head의 차원 (d_model / num_heads)
    pub head_dim: usize,
    /// 드롭아웃 확률
    pub dropout_p: f32,
}

impl MultiHeadAttention {
    /// 새로운 Multi-Head Attention 생성
    ///
    /// # 인자
    /// - `d_model`: 모델 차원 (예: 512)
    /// - `num_heads`: Head 개수 (예: 8)
    /// - `dropout_p`: 드롭아웃 확률
    pub fn new(d_model: usize, num_heads: usize, dropout_p: f32) -> Self {
        assert!(
            d_model % num_heads == 0,
            "d_model ({}) must be divisible by num_heads ({})",
            d_model, num_heads
        );

        let head_dim = d_model / num_heads;

        Self {
            wq: Linear::new(d_model, d_model),
            wk: Linear::new(d_model, d_model),
            wv: Linear::new(d_model, d_model),
            wo: Linear::new(d_model, d_model),
            d_model,
            num_heads,
            head_dim,
            dropout_p,
        }
    }

    /// Multi-Head Attention 순전파
    ///
    /// # 인자
    /// - `query`: Query 입력 (batch, seq_q, d_model)
    /// - `key`: Key 입력 (batch, seq_k, d_model)
    /// - `value`: Value 입력 (batch, seq_v, d_model)
    /// - `mask`: 선택적 마스크
    ///
    /// # 반환
    /// - output: (batch, seq_q, d_model)
    pub fn forward(
        &self,
        query: &Tensor3D,
        key: &Tensor3D,
        value: &Tensor3D,
        mask: Option<&Tensor2D>,
    ) -> Tensor3D {
        // 1. Linear 투영
        let q = self.wq.forward(query); // (batch, seq_q, d_model)
        let k = self.wk.forward(key);   // (batch, seq_k, d_model)
        let v = self.wv.forward(value); // (batch, seq_v, d_model)

        // 2. Head로 분리
        // (batch, seq, d_model) → (batch, num_heads, seq, head_dim)
        let q = reshape_for_multihead(&q, self.num_heads);
        let k = reshape_for_multihead(&k, self.num_heads);
        let v = reshape_for_multihead(&v, self.num_heads);

        // 3. Scaled Dot-Product Attention
        let attention = ScaledDotProductAttention::new(self.dropout_p);
        let (context, _weights) = attention.forward(&q, &k, &v, mask);

        // 4. Head 결합
        // (batch, num_heads, seq, head_dim) → (batch, seq, d_model)
        let context = reshape_from_multihead(&context);

        // 5. 출력 투영
        self.wo.forward(&context)
    }

    /// 파라미터 개수 반환
    pub fn num_parameters(&self) -> usize {
        self.wq.num_parameters()
            + self.wk.num_parameters()
            + self.wv.num_parameters()
            + self.wo.num_parameters()
    }
}

/// Causal Self-Attention (GPT 스타일)
///
/// 미래 토큰을 볼 수 없도록 마스킹합니다.
/// 언어 모델의 auto-regressive 생성에 필수입니다.
///
/// # 마스크 예시 (seq_len=4)
/// ```text
/// [[1, 0, 0, 0],   ← 첫 토큰: 자신만 참조
///  [1, 1, 0, 0],   ← 두 번째: 1-2번 참조
///  [1, 1, 1, 0],   ← 세 번째: 1-3번 참조
///  [1, 1, 1, 1]]   ← 네 번째: 모두 참조
/// ```
#[derive(Clone)]
pub struct CausalSelfAttention {
    /// 내부 Multi-Head Attention
    pub attention: MultiHeadAttention,
    /// 최대 시퀀스 길이
    pub max_seq_len: usize,
    /// 캐시된 causal mask
    causal_mask: Tensor2D,
}

impl CausalSelfAttention {
    /// 새로운 Causal Self-Attention 생성
    pub fn new(d_model: usize, num_heads: usize, max_seq_len: usize, dropout_p: f32) -> Self {
        let attention = MultiHeadAttention::new(d_model, num_heads, dropout_p);
        let causal_mask = create_causal_mask(max_seq_len);

        Self {
            attention,
            max_seq_len,
            causal_mask,
        }
    }

    /// Causal Self-Attention 순전파
    ///
    /// Query, Key, Value가 모두 같은 Self-Attention입니다.
    /// 자동으로 causal mask가 적용됩니다.
    pub fn forward(&self, x: &Tensor3D) -> Tensor3D {
        let seq_len = x.shape()[1];

        // 현재 시퀀스 길이에 맞는 마스크 추출
        let mask = self.causal_mask.slice(s![..seq_len, ..seq_len]).to_owned();

        // Self-Attention: Q=K=V=x
        self.attention.forward(x, x, x, Some(&mask))
    }

    /// 파라미터 개수 반환
    pub fn num_parameters(&self) -> usize {
        self.attention.num_parameters()
    }
}

/// Attention 가중치 시각화를 위한 유틸리티
///
/// 어텐션 패턴을 문자열로 반환합니다.
pub fn visualize_attention(weights: &Tensor4D, tokens: &[&str]) -> String {
    let mut result = String::new();
    let seq_len = weights.shape()[2];

    // 첫 번째 배치, 첫 번째 헤드만 시각화
    result.push_str("Attention Pattern (Batch 0, Head 0):\n");
    result.push_str("        ");

    for token in tokens.iter().take(seq_len) {
        result.push_str(&format!("{:>8}", token));
    }
    result.push('\n');

    for (i, token) in tokens.iter().take(seq_len).enumerate() {
        result.push_str(&format!("{:>6}: ", token));
        for j in 0..seq_len {
            let w = weights[[0, 0, i, j]];
            let bar = match w {
                w if w > 0.5 => "████",
                w if w > 0.3 => "███░",
                w if w > 0.1 => "██░░",
                w if w > 0.05 => "█░░░",
                _ => "░░░░",
            };
            result.push_str(&format!("{:>8}", bar));
        }
        result.push('\n');
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::randn_3d;

    #[test]
    fn test_multihead_attention_shape() {
        let mha = MultiHeadAttention::new(64, 8, 0.0);
        let x = randn_3d(2, 10, 64);
        let y = mha.forward(&x, &x, &x, None);

        assert_eq!(y.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_causal_self_attention_shape() {
        let attn = CausalSelfAttention::new(64, 8, 100, 0.0);
        let x = randn_3d(2, 10, 64);
        let y = attn.forward(&x);

        assert_eq!(y.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_attention_mask() {
        let mask = create_causal_mask(4);

        // 하삼각 행렬이어야 함
        assert_eq!(mask[[0, 1]], 0.0);  // 미래는 못 봄
        assert_eq!(mask[[1, 0]], 1.0);  // 과거는 볼 수 있음
        assert_eq!(mask[[3, 3]], 1.0);  // 자기 자신은 볼 수 있음
    }
}
