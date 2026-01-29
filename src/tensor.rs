//! 텐서 연산 모듈
//!
//! LLM의 모든 연산은 텐서(다차원 배열) 위에서 이루어집니다.
//! 이 모듈은 ndarray를 기반으로 한 텐서 연산을 제공합니다.
//!
//! # 핵심 개념
//! - 스칼라 (0D): 단일 숫자
//! - 벡터 (1D): [1, 2, 3]
//! - 행렬 (2D): [[1, 2], [3, 4]]
//! - 3D 텐서: LLM 입력 (batch, seq_len, embed_dim)

use ndarray::{Array1, Array2, Array3, Array4, Axis, s};
use rand::Rng;
use rand_distr::{Distribution, Normal, Uniform};
use std::f32::consts::PI;

/// 1차원 텐서 (벡터) - 바이어스, 임베딩 벡터 등
pub type Tensor1D = Array1<f32>;

/// 2차원 텐서 (행렬) - 가중치 행렬, 임베딩 테이블 등
pub type Tensor2D = Array2<f32>;

/// 3차원 텐서 - LLM 기본 형태 (batch, seq_len, embed_dim)
pub type Tensor3D = Array3<f32>;

/// 4차원 텐서 - Multi-Head Attention (batch, num_heads, seq_len, head_dim)
pub type Tensor4D = Array4<f32>;

// ============================================================================
// 텐서 생성 함수들
// ============================================================================

/// 영 텐서 생성
pub fn zeros_1d(size: usize) -> Tensor1D {
    Array1::zeros(size)
}

pub fn zeros_2d(rows: usize, cols: usize) -> Tensor2D {
    Array2::zeros((rows, cols))
}

pub fn zeros_3d(dim0: usize, dim1: usize, dim2: usize) -> Tensor3D {
    Array3::zeros((dim0, dim1, dim2))
}

/// 1로 채워진 텐서 생성
pub fn ones_1d(size: usize) -> Tensor1D {
    Array1::ones(size)
}

pub fn ones_2d(rows: usize, cols: usize) -> Tensor2D {
    Array2::ones((rows, cols))
}

/// 정규분포로 초기화된 텐서
///
/// 신경망 가중치 초기화에 주로 사용됩니다.
/// mean=0, std=1이 기본이며, Xavier/He 초기화에서는 std를 조절합니다.
pub fn randn_1d(size: usize) -> Tensor1D {
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();
    Array1::from_iter((0..size).map(|_| normal.sample(&mut rng)))
}

pub fn randn_2d(rows: usize, cols: usize) -> Tensor2D {
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();
    Array2::from_shape_fn((rows, cols), |_| normal.sample(&mut rng))
}

pub fn randn_3d(dim0: usize, dim1: usize, dim2: usize) -> Tensor3D {
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();
    Array3::from_shape_fn((dim0, dim1, dim2), |_| normal.sample(&mut rng))
}

/// Xavier (Glorot) 초기화
///
/// W ~ U(-sqrt(6/(fan_in + fan_out)), sqrt(6/(fan_in + fan_out)))
///
/// 선형 레이어의 가중치 초기화에 권장됩니다.
/// 입력과 출력의 분산을 일정하게 유지합니다.
pub fn xavier_uniform(fan_in: usize, fan_out: usize) -> Tensor2D {
    let mut rng = rand::thread_rng();
    let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
    let uniform = Uniform::new(-limit, limit);
    Array2::from_shape_fn((fan_out, fan_in), |_| uniform.sample(&mut rng))
}

/// He (Kaiming) 초기화
///
/// W ~ N(0, sqrt(2/fan_in))
///
/// ReLU 활성화 함수와 함께 사용할 때 권장됩니다.
pub fn he_normal(fan_in: usize, fan_out: usize) -> Tensor2D {
    let mut rng = rand::thread_rng();
    let std = (2.0 / fan_in as f32).sqrt();
    let normal = Normal::new(0.0, std).unwrap();
    Array2::from_shape_fn((fan_out, fan_in), |_| normal.sample(&mut rng))
}

// ============================================================================
// 행렬 연산
// ============================================================================

/// 2D 행렬 곱 (Matrix Multiplication)
///
/// C = A @ B
/// A: (m, k), B: (k, n) -> C: (m, n)
///
/// 신경망의 핵심 연산입니다. y = xW + b에서 xW 부분입니다.
pub fn matmul_2d(a: &Tensor2D, b: &Tensor2D) -> Tensor2D {
    a.dot(b)
}

/// 3D 배치 행렬 곱
///
/// 각 배치에 대해 독립적으로 행렬 곱을 수행합니다.
/// A: (batch, m, k), B: (batch, k, n) -> C: (batch, m, n)
pub fn batched_matmul(a: &Tensor3D, b: &Tensor3D) -> Tensor3D {
    let batch_size = a.shape()[0];
    let m = a.shape()[1];
    let n = b.shape()[2];

    let mut result = zeros_3d(batch_size, m, n);

    for i in 0..batch_size {
        let a_slice = a.slice(s![i, .., ..]);
        let b_slice = b.slice(s![i, .., ..]);
        let c = a_slice.dot(&b_slice);
        result.slice_mut(s![i, .., ..]).assign(&c);
    }

    result
}

/// 4D 배치 행렬 곱 (Multi-Head Attention용)
///
/// A: (batch, heads, seq, dim), B: (batch, heads, dim, seq)
/// -> C: (batch, heads, seq, seq)
pub fn batched_matmul_4d(a: &Tensor4D, b: &Tensor4D) -> Tensor4D {
    let batch_size = a.shape()[0];
    let num_heads = a.shape()[1];
    let seq_len_a = a.shape()[2];
    let seq_len_b = b.shape()[3];

    let mut result = Array4::zeros((batch_size, num_heads, seq_len_a, seq_len_b));

    for batch in 0..batch_size {
        for head in 0..num_heads {
            let a_slice = a.slice(s![batch, head, .., ..]);
            let b_slice = b.slice(s![batch, head, .., ..]);
            let c = a_slice.dot(&b_slice);
            result.slice_mut(s![batch, head, .., ..]).assign(&c);
        }
    }

    result
}

// ============================================================================
// 텐서 변환 연산
// ============================================================================

/// 전치 (Transpose)
///
/// 마지막 두 축을 교환합니다.
pub fn transpose_2d(a: &Tensor2D) -> Tensor2D {
    a.t().to_owned()
}

/// 3D 텐서의 마지막 두 축 전치
pub fn transpose_3d_last2(a: &Tensor3D) -> Tensor3D {
    a.clone().permuted_axes([0, 2, 1])
}

/// 4D 텐서의 마지막 두 축 전치
pub fn transpose_4d_last2(a: &Tensor4D) -> Tensor4D {
    a.clone().permuted_axes([0, 1, 3, 2])
}

/// 3D → 4D 변환 (Multi-Head Attention용)
///
/// (batch, seq_len, embed_dim) -> (batch, num_heads, seq_len, head_dim)
pub fn reshape_for_multihead(
    tensor: &Tensor3D,
    num_heads: usize,
) -> Tensor4D {
    let batch_size = tensor.shape()[0];
    let seq_len = tensor.shape()[1];
    let embed_dim = tensor.shape()[2];
    let head_dim = embed_dim / num_heads;

    // 수동으로 변환 (메모리 레이아웃 문제 방지)
    let mut result = Array4::zeros((batch_size, num_heads, seq_len, head_dim));

    for b in 0..batch_size {
        for s in 0..seq_len {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    result[[b, h, s, d]] = tensor[[b, s, h * head_dim + d]];
                }
            }
        }
    }

    result
}

/// 4D → 3D 변환 (Multi-Head Attention 결과 병합)
///
/// (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, embed_dim)
pub fn reshape_from_multihead(tensor: &Tensor4D) -> Tensor3D {
    let batch_size = tensor.shape()[0];
    let num_heads = tensor.shape()[1];
    let seq_len = tensor.shape()[2];
    let head_dim = tensor.shape()[3];
    let embed_dim = num_heads * head_dim;

    // 수동으로 변환 (메모리 레이아웃 문제 방지)
    let mut result = zeros_3d(batch_size, seq_len, embed_dim);

    for b in 0..batch_size {
        for s in 0..seq_len {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    result[[b, s, h * head_dim + d]] = tensor[[b, h, s, d]];
                }
            }
        }
    }

    result
}

// ============================================================================
// 활성화 함수
// ============================================================================

/// Softmax 함수
///
/// softmax(x_i) = exp(x_i) / sum(exp(x_j))
///
/// 수치 안정성을 위해 최댓값을 빼서 계산합니다.
/// 결과는 확률 분포 (합이 1, 모든 값이 양수)
pub fn softmax_1d(x: &Tensor1D) -> Tensor1D {
    let max_val = x.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_x = x.mapv(|v| (v - max_val).exp());
    let sum = exp_x.sum();
    exp_x / sum
}

/// 2D 텐서의 마지막 축에 대한 Softmax
pub fn softmax_2d(x: &Tensor2D) -> Tensor2D {
    let mut result = x.clone();
    for mut row in result.rows_mut() {
        let max_val = row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        row.mapv_inplace(|v| (v - max_val).exp());
        let sum = row.sum();
        row.mapv_inplace(|v| v / sum);
    }
    result
}

/// 3D 텐서의 마지막 축에 대한 Softmax (Attention Score용)
pub fn softmax_3d_last(x: &Tensor3D) -> Tensor3D {
    let mut result = x.clone();
    let shape = result.shape().to_vec();

    for i in 0..shape[0] {
        for j in 0..shape[1] {
            let mut slice = result.slice_mut(s![i, j, ..]);
            let max_val = slice.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            slice.mapv_inplace(|v| (v - max_val).exp());
            let sum = slice.sum();
            slice.mapv_inplace(|v| v / sum);
        }
    }
    result
}

/// 4D 텐서의 마지막 축에 대한 Softmax
pub fn softmax_4d_last(x: &Tensor4D) -> Tensor4D {
    let mut result = x.clone();
    let shape = result.shape().to_vec();

    for b in 0..shape[0] {
        for h in 0..shape[1] {
            for s in 0..shape[2] {
                let mut slice = result.slice_mut(s![b, h, s, ..]);
                let max_val = slice.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                slice.mapv_inplace(|v| (v - max_val).exp());
                let sum = slice.sum();
                slice.mapv_inplace(|v| v / sum);
            }
        }
    }
    result
}

/// ReLU (Rectified Linear Unit)
///
/// ReLU(x) = max(0, x)
///
/// 가장 단순하지만 효과적인 활성화 함수입니다.
pub fn relu(x: &Tensor3D) -> Tensor3D {
    x.mapv(|v| v.max(0.0))
}

/// GELU (Gaussian Error Linear Unit)
///
/// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
///
/// Transformer에서 주로 사용되는 활성화 함수입니다.
/// ReLU보다 부드러운 곡선을 가지며, 음수 입력도 일부 통과시킵니다.
pub fn gelu(x: &Tensor3D) -> Tensor3D {
    let sqrt_2_over_pi = (2.0 / PI).sqrt();
    x.mapv(|v| {
        0.5 * v * (1.0 + (sqrt_2_over_pi * (v + 0.044715 * v.powi(3))).tanh())
    })
}

/// Tanh 활성화 함수
pub fn tanh(x: &Tensor3D) -> Tensor3D {
    x.mapv(|v| v.tanh())
}

/// Sigmoid 함수
///
/// σ(x) = 1 / (1 + exp(-x))
pub fn sigmoid(x: &Tensor3D) -> Tensor3D {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

// ============================================================================
// 정규화 연산
// ============================================================================

/// Layer Normalization
///
/// LayerNorm(x) = (x - μ) / √(σ² + ε) * γ + β
///
/// 각 샘플의 feature 차원을 정규화합니다.
/// Batch Norm과 달리 배치 크기에 독립적입니다.
pub fn layer_norm(
    x: &Tensor3D,
    gamma: &Tensor1D,  // 스케일 파라미터
    beta: &Tensor1D,   // 시프트 파라미터
    eps: f32,
) -> Tensor3D {
    let shape = x.shape();
    let mut result = zeros_3d(shape[0], shape[1], shape[2]);

    for i in 0..shape[0] {
        for j in 0..shape[1] {
            let slice = x.slice(s![i, j, ..]);

            // 평균 계산
            let mean = slice.mean().unwrap();

            // 분산 계산
            let variance = slice.mapv(|v| (v - mean).powi(2)).mean().unwrap();

            // 정규화
            let std_inv = 1.0 / (variance + eps).sqrt();

            for (k, &v) in slice.iter().enumerate() {
                let normalized = (v - mean) * std_inv;
                result[[i, j, k]] = normalized * gamma[k] + beta[k];
            }
        }
    }

    result
}

// ============================================================================
// 마스킹 연산
// ============================================================================

/// Causal Mask 생성 (Auto-regressive 모델용)
///
/// 미래 토큰을 볼 수 없도록 하삼각 마스크를 생성합니다.
///
/// 예: seq_len=4
/// [[1, 0, 0, 0],
///  [1, 1, 0, 0],
///  [1, 1, 1, 0],
///  [1, 1, 1, 1]]
pub fn create_causal_mask(seq_len: usize) -> Tensor2D {
    let mut mask = zeros_2d(seq_len, seq_len);
    for i in 0..seq_len {
        for j in 0..=i {
            mask[[i, j]] = 1.0;
        }
    }
    mask
}

/// 마스크 적용 (Attention Score에 적용)
///
/// mask가 0인 위치를 매우 작은 값(-inf)으로 설정하여
/// softmax 후 0에 가깝게 만듭니다.
pub fn apply_mask(scores: &Tensor4D, mask: &Tensor2D) -> Tensor4D {
    let mut result = scores.clone();
    let shape = scores.shape();

    for b in 0..shape[0] {
        for h in 0..shape[1] {
            for i in 0..shape[2] {
                for j in 0..shape[3] {
                    if mask[[i, j]] == 0.0 {
                        result[[b, h, i, j]] = f32::NEG_INFINITY;
                    }
                }
            }
        }
    }

    result
}

// ============================================================================
// Positional Encoding
// ============================================================================

/// Sinusoidal Positional Encoding
///
/// PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
/// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
///
/// 위치 정보를 임베딩에 추가합니다.
/// Attention은 위치를 구분하지 못하므로 필수적입니다.
pub fn sinusoidal_positional_encoding(max_len: usize, d_model: usize) -> Tensor2D {
    let mut pe = zeros_2d(max_len, d_model);

    for pos in 0..max_len {
        for i in 0..(d_model / 2) {
            let div_term = (pos as f32) / (10000.0_f32).powf(2.0 * i as f32 / d_model as f32);
            pe[[pos, 2 * i]] = div_term.sin();
            pe[[pos, 2 * i + 1]] = div_term.cos();
        }
    }

    pe
}

// ============================================================================
// 유틸리티 함수
// ============================================================================

/// 텐서 요소의 합
pub fn sum(x: &Tensor3D) -> f32 {
    x.sum()
}

/// 텐서 요소의 평균
pub fn mean(x: &Tensor3D) -> f32 {
    x.mean().unwrap()
}

/// 스칼라 곱
pub fn scale(x: &Tensor3D, scalar: f32) -> Tensor3D {
    x.mapv(|v| v * scalar)
}

/// 텐서 덧셈
pub fn add(a: &Tensor3D, b: &Tensor3D) -> Tensor3D {
    a + b
}

/// 브로드캐스팅 덧셈 (3D + 1D)
///
/// 바이어스 추가에 사용됩니다.
/// (batch, seq, dim) + (dim,) -> (batch, seq, dim)
pub fn add_bias(x: &Tensor3D, bias: &Tensor1D) -> Tensor3D {
    let mut result = x.clone();
    let shape = x.shape();

    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                result[[i, j, k]] += bias[k];
            }
        }
    }

    result
}

/// Cross-Entropy Loss
///
/// L = -sum(y_true * log(y_pred))
///
/// 분류 문제의 손실 함수입니다.
/// y_pred는 softmax 출력 (확률), y_true는 one-hot 인코딩된 정답입니다.
pub fn cross_entropy_loss(predictions: &Tensor2D, targets: &[usize]) -> f32 {
    let batch_size = predictions.shape()[0];
    let mut total_loss = 0.0;

    for (i, &target) in targets.iter().enumerate() {
        let pred = predictions[[i, target]].max(1e-10); // 수치 안정성
        total_loss -= pred.ln();
    }

    total_loss / batch_size as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_matmul_2d() {
        let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let c = matmul_2d(&a, &b);

        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c[[0, 0]], 22.0); // 1*1 + 2*3 + 3*5
        assert_eq!(c[[0, 1]], 28.0); // 1*2 + 2*4 + 3*6
    }

    #[test]
    fn test_softmax() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y = softmax_1d(&x);

        // 합이 1이어야 함
        assert_relative_eq!(y.sum(), 1.0, epsilon = 1e-6);

        // 모든 값이 양수여야 함
        assert!(y.iter().all(|&v| v > 0.0));

        // 순서가 유지되어야 함
        assert!(y[2] > y[1] && y[1] > y[0]);
    }

    #[test]
    fn test_causal_mask() {
        let mask = create_causal_mask(4);

        // 하삼각 행렬이어야 함
        assert_eq!(mask[[0, 0]], 1.0);
        assert_eq!(mask[[0, 1]], 0.0);
        assert_eq!(mask[[1, 0]], 1.0);
        assert_eq!(mask[[1, 1]], 1.0);
        assert_eq!(mask[[3, 3]], 1.0);
    }

    #[test]
    fn test_positional_encoding() {
        let pe = sinusoidal_positional_encoding(100, 64);

        assert_eq!(pe.shape(), &[100, 64]);

        // 첫 번째 위치의 첫 번째 값은 sin(0) = 0
        assert_relative_eq!(pe[[0, 0]], 0.0, epsilon = 1e-6);

        // 모든 값은 -1과 1 사이
        assert!(pe.iter().all(|&v| v >= -1.0 && v <= 1.0));
    }
}
