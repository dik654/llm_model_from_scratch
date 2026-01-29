//! Layer Normalization
//!
//! 신경망 학습을 안정화시키는 정규화 기법입니다.
//!
//! # 수학적 정의
//! ```text
//! LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β
//!
//! μ = mean(x)      (평균)
//! σ² = var(x)      (분산)
//! γ, β = 학습 파라미터 (스케일, 시프트)
//! ε = 작은 상수 (수치 안정성)
//! ```
//!
//! # Batch Norm vs Layer Norm
//! - **Batch Norm**: 배치 방향으로 정규화 (각 feature별)
//! - **Layer Norm**: 레이어 방향으로 정규화 (각 sample별)
//!
//! NLP에서 Layer Norm이 선호되는 이유:
//! 1. 가변 길이 시퀀스 처리 용이
//! 2. 배치 크기에 독립적
//! 3. 추론 시 일관된 동작
//!
//! # Transformer에서의 역할
//! - Attention과 FFN 출력을 정규화
//! - Gradient 흐름 개선
//! - 학습 안정성 향상

use crate::tensor::{Tensor1D, Tensor3D, ones_1d, zeros_1d};
use ndarray::s;
use super::Module;

/// Layer Normalization
///
/// # 예시
/// ```ignore
/// let ln = LayerNorm::new(768);
/// let x = randn_3d(32, 10, 768);  // (batch, seq, dim)
/// let y = ln.forward(&x);         // 정규화된 (batch, seq, dim)
/// ```
#[derive(Clone)]
pub struct LayerNorm {
    /// 스케일 파라미터 γ (학습 가능)
    /// 정규화 후 출력의 스케일을 조절
    pub gamma: Tensor1D,

    /// 시프트 파라미터 β (학습 가능)
    /// 정규화 후 출력의 평균을 조절
    pub beta: Tensor1D,

    /// 정규화할 차원
    pub normalized_shape: usize,

    /// 수치 안정성을 위한 작은 상수
    pub eps: f32,
}

impl LayerNorm {
    /// 새로운 Layer Normalization 생성
    ///
    /// # 인자
    /// - `normalized_shape`: 정규화할 마지막 차원의 크기
    ///
    /// γ는 1로, β는 0으로 초기화됩니다.
    /// 이렇게 하면 초기에는 identity 변환에 가까워집니다.
    pub fn new(normalized_shape: usize) -> Self {
        Self {
            gamma: ones_1d(normalized_shape),
            beta: zeros_1d(normalized_shape),
            normalized_shape,
            eps: 1e-5,
        }
    }

    /// epsilon 값을 지정하여 생성
    pub fn new_with_eps(normalized_shape: usize, eps: f32) -> Self {
        Self {
            gamma: ones_1d(normalized_shape),
            beta: zeros_1d(normalized_shape),
            normalized_shape,
            eps,
        }
    }
}

impl Module for LayerNorm {
    /// Layer Normalization 적용
    ///
    /// # 동작 과정
    /// 1. 마지막 차원에 대해 평균 계산
    /// 2. 마지막 차원에 대해 분산 계산
    /// 3. (x - mean) / sqrt(var + eps)로 정규화
    /// 4. gamma로 스케일, beta로 시프트
    fn forward(&self, x: &Tensor3D) -> Tensor3D {
        let shape = x.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let dim = shape[2];

        assert_eq!(
            dim, self.normalized_shape,
            "Input dimension {} doesn't match normalized_shape {}",
            dim, self.normalized_shape
        );

        let mut output = Tensor3D::zeros((batch_size, seq_len, dim));

        for b in 0..batch_size {
            for s in 0..seq_len {
                let slice = x.slice(s![b, s, ..]);

                // 1. 평균 계산
                let mean: f32 = slice.iter().sum::<f32>() / dim as f32;

                // 2. 분산 계산
                let variance: f32 = slice.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / dim as f32;

                // 3. 정규화 (역 표준편차 미리 계산)
                let std_inv = 1.0 / (variance + self.eps).sqrt();

                // 4. 스케일과 시프트 적용
                for (d, &val) in slice.iter().enumerate() {
                    let normalized = (val - mean) * std_inv;
                    output[[b, s, d]] = normalized * self.gamma[d] + self.beta[d];
                }
            }
        }

        output
    }

    fn num_parameters(&self) -> usize {
        // gamma + beta
        self.normalized_shape * 2
    }
}

/// RMS Normalization (Root Mean Square Normalization)
///
/// Layer Norm의 간소화 버전으로, 평균을 빼지 않습니다.
/// Llama, GPT-NeoX 등 최신 모델에서 사용됩니다.
///
/// # 수학적 정의
/// ```text
/// RMSNorm(x) = x / √(mean(x²) + ε) * γ
/// ```
///
/// # 장점
/// - 계산이 더 간단함 (평균 계산 불필요)
/// - 실험적으로 Layer Norm과 유사한 성능
#[derive(Clone)]
pub struct RMSNorm {
    /// 스케일 파라미터 γ
    pub gamma: Tensor1D,
    /// 정규화할 차원
    pub normalized_shape: usize,
    /// 수치 안정성을 위한 작은 상수
    pub eps: f32,
}

impl RMSNorm {
    /// 새로운 RMS Normalization 생성
    pub fn new(normalized_shape: usize) -> Self {
        Self {
            gamma: ones_1d(normalized_shape),
            normalized_shape,
            eps: 1e-5,
        }
    }
}

impl Module for RMSNorm {
    /// RMS Normalization 적용
    fn forward(&self, x: &Tensor3D) -> Tensor3D {
        let shape = x.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let dim = shape[2];

        let mut output = Tensor3D::zeros((batch_size, seq_len, dim));

        for b in 0..batch_size {
            for s in 0..seq_len {
                let slice = x.slice(s![b, s, ..]);

                // RMS 계산: sqrt(mean(x²))
                let rms = (slice.iter().map(|&v| v * v).sum::<f32>() / dim as f32).sqrt();

                // 정규화 및 스케일 적용
                let rms_inv = 1.0 / (rms + self.eps);
                for (d, &val) in slice.iter().enumerate() {
                    output[[b, s, d]] = val * rms_inv * self.gamma[d];
                }
            }
        }

        output
    }

    fn num_parameters(&self) -> usize {
        self.normalized_shape
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::randn_3d;
    use approx::assert_relative_eq;

    #[test]
    fn test_layer_norm_shape() {
        let ln = LayerNorm::new(64);
        let x = randn_3d(2, 10, 64);
        let y = ln.forward(&x);

        assert_eq!(y.shape(), x.shape());
    }

    #[test]
    fn test_layer_norm_statistics() {
        let ln = LayerNorm::new(64);
        let x = randn_3d(1, 1, 64);
        let y = ln.forward(&x);

        // 정규화 후 평균은 beta(0)에 가깝고, 분산은 gamma²(1)에 가까워야 함
        let mean: f32 = y.iter().sum::<f32>() / 64.0;
        let variance: f32 = y.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / 64.0;

        assert_relative_eq!(mean, 0.0, epsilon = 0.1);
        assert_relative_eq!(variance, 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_rms_norm_shape() {
        let rms = RMSNorm::new(64);
        let x = randn_3d(2, 10, 64);
        let y = rms.forward(&x);

        assert_eq!(y.shape(), x.shape());
    }
}
