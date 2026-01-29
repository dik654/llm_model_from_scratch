//! Attention 시각화
//!
//! Attention 패턴을 시각화 데이터로 변환합니다.
//! 웹앱에서 렌더링할 수 있는 형태로 출력합니다.

use crate::tensor::Tensor4D;
use serde::{Serialize, Deserialize};

/// Attention 가중치 시각화 데이터
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AttentionData {
    /// Head별 attention 가중치
    pub heads: Vec<HeadAttention>,
    /// 토큰 레이블
    pub tokens: Vec<String>,
    /// 배치 인덱스
    pub batch_idx: usize,
}

/// 단일 Head의 Attention 데이터
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HeadAttention {
    /// Head 인덱스
    pub head_idx: usize,
    /// Attention 가중치 행렬 (query_len × key_len)
    pub weights: Vec<Vec<f32>>,
}

/// Attention 시각화 유틸리티
pub struct AttentionViz;

impl AttentionViz {
    /// 4D Attention 텐서를 시각화 데이터로 변환
    ///
    /// # 인자
    /// - `attention`: (batch, heads, seq_q, seq_k) 형태의 텐서
    /// - `tokens`: 토큰 문자열
    /// - `batch_idx`: 시각화할 배치 인덱스
    pub fn from_tensor(
        attention: &Tensor4D,
        tokens: &[String],
        batch_idx: usize,
    ) -> AttentionData {
        let num_heads = attention.shape()[1];
        let seq_len_q = attention.shape()[2];
        let seq_len_k = attention.shape()[3];

        let heads: Vec<HeadAttention> = (0..num_heads)
            .map(|h| {
                let weights: Vec<Vec<f32>> = (0..seq_len_q)
                    .map(|q| {
                        (0..seq_len_k)
                            .map(|k| attention[[batch_idx, h, q, k]])
                            .collect()
                    })
                    .collect();

                HeadAttention {
                    head_idx: h,
                    weights,
                }
            })
            .collect();

        AttentionData {
            heads,
            tokens: tokens.to_vec(),
            batch_idx,
        }
    }

    /// JSON으로 직렬화
    pub fn to_json(data: &AttentionData) -> String {
        serde_json::to_string_pretty(data).unwrap_or_default()
    }

    /// 특정 Head의 attention 패턴 요약
    pub fn summarize_head(head: &HeadAttention, tokens: &[String]) -> String {
        let mut result = format!("Head {}: 주요 Attention 패턴\n", head.head_idx);

        for (i, row) in head.weights.iter().enumerate() {
            if i >= tokens.len() {
                break;
            }

            // 가장 높은 attention을 받는 토큰 찾기
            let max_idx = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            let max_val = row[max_idx];

            if max_idx < tokens.len() {
                result.push_str(&format!(
                    "  {} → {} ({:.2})\n",
                    tokens[i], tokens[max_idx], max_val
                ));
            }
        }

        result
    }

    /// Attention 엔트로피 계산
    ///
    /// 높은 엔트로피 = 균등하게 분산된 attention
    /// 낮은 엔트로피 = 특정 토큰에 집중
    pub fn compute_entropy(weights: &[f32]) -> f32 {
        weights
            .iter()
            .filter(|&&w| w > 0.0)
            .map(|&w| -w * w.ln())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array4;

    #[test]
    fn test_attention_viz() {
        let attention = Array4::from_elem((1, 2, 3, 3), 0.33);
        let tokens = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        let data = AttentionViz::from_tensor(&attention, &tokens, 0);

        assert_eq!(data.heads.len(), 2);
        assert_eq!(data.heads[0].weights.len(), 3);
    }

    #[test]
    fn test_entropy() {
        // 균등 분포
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        let entropy_uniform = AttentionViz::compute_entropy(&uniform);

        // 집중 분포
        let focused = vec![0.9, 0.05, 0.025, 0.025];
        let entropy_focused = AttentionViz::compute_entropy(&focused);

        assert!(entropy_uniform > entropy_focused);
    }
}
