//! 선형 레이어 (Linear Layer / Fully Connected Layer)
//!
//! 신경망의 가장 기본적인 연산입니다.
//!
//! # 수학적 정의
//! y = xW^T + b
//!
//! 여기서:
//! - x: 입력 텐서 (*, in_features)
//! - W: 가중치 행렬 (out_features, in_features)
//! - b: 바이어스 벡터 (out_features)
//! - y: 출력 텐서 (*, out_features)
//!
//! # 역할
//! - 차원 변환: in_features → out_features
//! - 선형 조합: 입력 특성들의 가중 합

use crate::tensor::{Tensor1D, Tensor2D, Tensor3D, xavier_uniform, zeros_1d};
use ndarray::s;
use super::Module;

/// 선형 레이어
///
/// # 예시
/// ```
/// let linear = Linear::new(768, 3072); // 768 → 3072 차원 변환
/// let x = zeros_3d(32, 10, 768);       // (batch=32, seq=10, dim=768)
/// let y = linear.forward(&x);          // (batch=32, seq=10, dim=3072)
/// ```
#[derive(Clone)]
pub struct Linear {
    /// 가중치 행렬 (out_features, in_features)
    pub weight: Tensor2D,
    /// 바이어스 벡터 (out_features)
    pub bias: Tensor1D,
    /// 입력 차원
    pub in_features: usize,
    /// 출력 차원
    pub out_features: usize,
}

impl Linear {
    /// 새로운 선형 레이어 생성
    ///
    /// # 인자
    /// - `in_features`: 입력 차원
    /// - `out_features`: 출력 차원
    ///
    /// 가중치는 Xavier 초기화, 바이어스는 0으로 초기화됩니다.
    pub fn new(in_features: usize, out_features: usize) -> Self {
        // Xavier 초기화: 입출력 분산을 균등하게
        let weight = xavier_uniform(in_features, out_features);
        let bias = zeros_1d(out_features);

        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }

    /// 바이어스 없는 선형 레이어 생성
    ///
    /// Attention의 Q, K, V 투영에 주로 사용됩니다.
    pub fn new_no_bias(in_features: usize, out_features: usize) -> Self {
        let weight = xavier_uniform(in_features, out_features);
        let bias = zeros_1d(out_features);

        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }

    /// 2D 입력에 대한 순전파
    ///
    /// (batch, in_features) → (batch, out_features)
    pub fn forward_2d(&self, x: &Tensor2D) -> Tensor2D {
        // y = x @ W^T + b
        let y = x.dot(&self.weight.t());

        // 바이어스 추가
        let mut result = y.clone();
        for mut row in result.rows_mut() {
            row += &self.bias;
        }
        result
    }
}

impl Module for Linear {
    /// 3D 입력에 대한 순전파
    ///
    /// (batch, seq_len, in_features) → (batch, seq_len, out_features)
    ///
    /// # 동작 과정
    /// 1. 각 배치, 각 위치에서 독립적으로 선형 변환 적용
    /// 2. y[b, s, :] = x[b, s, :] @ W^T + b
    fn forward(&self, x: &Tensor3D) -> Tensor3D {
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];

        let mut output = Tensor3D::zeros((batch_size, seq_len, self.out_features));

        for b in 0..batch_size {
            for s in 0..seq_len {
                let x_slice = x.slice(s![b, s, ..]);

                // y = x @ W^T + b
                for o in 0..self.out_features {
                    let mut sum = self.bias[o];
                    for i in 0..self.in_features {
                        sum += x_slice[i] * self.weight[[o, i]];
                    }
                    output[[b, s, o]] = sum;
                }
            }
        }

        output
    }

    fn num_parameters(&self) -> usize {
        self.in_features * self.out_features + self.out_features
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::zeros_3d;

    #[test]
    fn test_linear_shape() {
        let linear = Linear::new(64, 128);
        let x = zeros_3d(2, 10, 64);
        let y = linear.forward(&x);

        assert_eq!(y.shape(), &[2, 10, 128]);
    }

    #[test]
    fn test_linear_parameters() {
        let linear = Linear::new(64, 128);
        assert_eq!(linear.num_parameters(), 64 * 128 + 128);
    }
}
