//! 임베딩 레이어 (Embedding Layer)
//!
//! 이산적인 토큰 ID를 연속적인 벡터로 변환합니다.
//!
//! # 개념
//! ```text
//! 토큰 ID: 42  →  임베딩 벡터: [0.1, -0.3, 0.5, ...]
//!                            (embed_dim 차원)
//! ```
//!
//! # 내부 구조
//! - 룩업 테이블: (vocab_size, embed_dim) 크기의 행렬
//! - 토큰 ID로 해당 행을 인덱싱
//!
//! # One-hot과의 비교
//! - One-hot: [0, 0, 1, 0, ..., 0] (vocab_size 차원, sparse)
//! - Embedding: [0.1, -0.3, ...] (embed_dim 차원, dense)
//! - Embedding이 훨씬 효율적이고 의미를 학습할 수 있음

use crate::tensor::{Tensor2D, Tensor3D, randn_2d};
use ndarray::s;

/// 임베딩 레이어
///
/// # 예시
/// ```ignore
/// let embedding = Embedding::new(50000, 768); // 어휘 크기 50000, 임베딩 768차원
/// let token_ids = vec![vec![100, 2500, 345]]; // 배치 크기 1, 시퀀스 길이 3
/// let vectors = embedding.forward(&token_ids); // (1, 3, 768)
/// ```
#[derive(Clone)]
pub struct Embedding {
    /// 임베딩 테이블 (vocab_size, embed_dim)
    /// 각 행이 하나의 토큰에 대한 임베딩 벡터
    pub weight: Tensor2D,
    /// 어휘 크기 (고유 토큰 개수)
    pub vocab_size: usize,
    /// 임베딩 차원
    pub embed_dim: usize,
}

impl Embedding {
    /// 새로운 임베딩 레이어 생성
    ///
    /// # 인자
    /// - `vocab_size`: 어휘 크기 (토큰 개수)
    /// - `embed_dim`: 임베딩 벡터 차원
    ///
    /// 가중치는 표준정규분포로 초기화됩니다.
    pub fn new(vocab_size: usize, embed_dim: usize) -> Self {
        // 표준정규분포 초기화 후 스케일 조정
        let mut weight = randn_2d(vocab_size, embed_dim);
        let scale = 1.0 / (embed_dim as f32).sqrt();
        weight.mapv_inplace(|v| v * scale);

        Self {
            weight,
            vocab_size,
            embed_dim,
        }
    }

    /// 토큰 ID를 임베딩 벡터로 변환
    ///
    /// # 인자
    /// - `token_ids`: 2D 벡터 (batch_size, seq_len)
    ///
    /// # 반환
    /// - 3D 텐서 (batch_size, seq_len, embed_dim)
    ///
    /// # 동작 과정
    /// 각 토큰 ID에 대해 임베딩 테이블에서 해당 행을 가져옵니다.
    /// ```text
    /// token_id = 42
    /// → embedding.weight[42, :] 반환
    /// ```
    pub fn forward(&self, token_ids: &[Vec<usize>]) -> Tensor3D {
        let batch_size = token_ids.len();
        let seq_len = token_ids[0].len();

        let mut output = Tensor3D::zeros((batch_size, seq_len, self.embed_dim));

        for (b, seq) in token_ids.iter().enumerate() {
            for (s, &token_id) in seq.iter().enumerate() {
                assert!(
                    token_id < self.vocab_size,
                    "Token ID {} is out of range (vocab_size: {})",
                    token_id,
                    self.vocab_size
                );

                // 임베딩 테이블에서 해당 행을 복사
                let embedding_vector = self.weight.slice(s![token_id, ..]);
                for (e, &val) in embedding_vector.iter().enumerate() {
                    output[[b, s, e]] = val;
                }
            }
        }

        output
    }

    /// 파라미터 개수 반환
    pub fn num_parameters(&self) -> usize {
        self.vocab_size * self.embed_dim
    }
}

/// 위치 임베딩 (Positional Embedding)
///
/// 각 위치에 대한 학습 가능한 임베딩입니다.
/// Sinusoidal encoding의 대안으로 사용됩니다.
#[derive(Clone)]
pub struct PositionalEmbedding {
    /// 위치 임베딩 테이블 (max_seq_len, embed_dim)
    pub weight: Tensor2D,
    /// 최대 시퀀스 길이
    pub max_seq_len: usize,
    /// 임베딩 차원
    pub embed_dim: usize,
}

impl PositionalEmbedding {
    /// 새로운 위치 임베딩 생성
    pub fn new(max_seq_len: usize, embed_dim: usize) -> Self {
        let mut weight = randn_2d(max_seq_len, embed_dim);
        let scale = 1.0 / (embed_dim as f32).sqrt();
        weight.mapv_inplace(|v| v * scale);

        Self {
            weight,
            max_seq_len,
            embed_dim,
        }
    }

    /// 위치 임베딩 반환
    ///
    /// # 인자
    /// - `seq_len`: 시퀀스 길이
    ///
    /// # 반환
    /// - 2D 텐서 (seq_len, embed_dim)
    pub fn forward(&self, seq_len: usize) -> Tensor2D {
        assert!(
            seq_len <= self.max_seq_len,
            "Sequence length {} exceeds maximum {}",
            seq_len,
            self.max_seq_len
        );

        self.weight.slice(s![..seq_len, ..]).to_owned()
    }

    /// 파라미터 개수 반환
    pub fn num_parameters(&self) -> usize {
        self.max_seq_len * self.embed_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_shape() {
        let embedding = Embedding::new(1000, 64);
        let token_ids = vec![vec![1, 2, 3, 4, 5]];
        let output = embedding.forward(&token_ids);

        assert_eq!(output.shape(), &[1, 5, 64]);
    }

    #[test]
    fn test_embedding_lookup() {
        let embedding = Embedding::new(1000, 64);

        // 같은 토큰 ID는 같은 임베딩 벡터를 반환해야 함
        let ids1 = vec![vec![42]];
        let ids2 = vec![vec![42]];

        let out1 = embedding.forward(&ids1);
        let out2 = embedding.forward(&ids2);

        for i in 0..64 {
            assert_eq!(out1[[0, 0, i]], out2[[0, 0, i]]);
        }
    }

    #[test]
    fn test_positional_embedding() {
        let pos_emb = PositionalEmbedding::new(512, 64);
        let output = pos_emb.forward(100);

        assert_eq!(output.shape(), &[100, 64]);
    }

    #[test]
    #[should_panic]
    fn test_embedding_out_of_range() {
        let embedding = Embedding::new(100, 64);
        let token_ids = vec![vec![200]]; // vocab_size보다 큰 ID
        embedding.forward(&token_ids);
    }
}
