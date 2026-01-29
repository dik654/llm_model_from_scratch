//! 역전파 (Backpropagation) 구현
//!
//! 신경망 학습의 핵심입니다. 손실 함수에서 각 파라미터로
//! 기울기를 계산합니다.
//!
//! # 수학적 배경
//!
//! ## Chain Rule (연쇄 법칙)
//! 합성 함수의 미분:
//! ```text
//! y = f(g(x))
//! dy/dx = dy/dg · dg/dx
//! ```
//!
//! ## 신경망에서의 역전파
//! ```text
//! x → Linear → ReLU → Linear → Loss
//!
//! 역전파:
//! ∂L/∂W₂ = ∂L/∂out · ∂out/∂W₂
//! ∂L/∂W₁ = ∂L/∂out · ∂out/∂h · ∂h/∂W₁
//! ```

use crate::tensor::{Tensor1D, Tensor2D, Tensor3D, zeros_1d, zeros_2d, zeros_3d};
use ndarray::s;

/// 선형 레이어의 역전파
///
/// y = xW^T + b
///
/// # 기울기
/// - ∂L/∂x = ∂L/∂y · W
/// - ∂L/∂W = (∂L/∂y)^T · x
/// - ∂L/∂b = sum(∂L/∂y, dim=0)
pub fn linear_backward(
    grad_output: &Tensor3D,  // ∂L/∂y: (batch, seq, out_features)
    input: &Tensor3D,        // x: (batch, seq, in_features)
    weight: &Tensor2D,       // W: (out_features, in_features)
) -> (Tensor3D, Tensor2D, Tensor1D) {
    let batch_size = input.shape()[0];
    let seq_len = input.shape()[1];
    let in_features = input.shape()[2];
    let out_features = weight.shape()[0];

    // ∂L/∂x = ∂L/∂y @ W
    let mut grad_input = zeros_3d(batch_size, seq_len, in_features);
    for b in 0..batch_size {
        for s in 0..seq_len {
            for i in 0..in_features {
                let mut sum = 0.0;
                for o in 0..out_features {
                    sum += grad_output[[b, s, o]] * weight[[o, i]];
                }
                grad_input[[b, s, i]] = sum;
            }
        }
    }

    // ∂L/∂W = (∂L/∂y)^T @ x
    let mut grad_weight = zeros_2d(out_features, in_features);
    for b in 0..batch_size {
        for s in 0..seq_len {
            for o in 0..out_features {
                for i in 0..in_features {
                    grad_weight[[o, i]] += grad_output[[b, s, o]] * input[[b, s, i]];
                }
            }
        }
    }

    // ∂L/∂b = sum(∂L/∂y)
    let mut grad_bias = zeros_1d(out_features);
    for b in 0..batch_size {
        for s in 0..seq_len {
            for o in 0..out_features {
                grad_bias[o] += grad_output[[b, s, o]];
            }
        }
    }

    (grad_input, grad_weight, grad_bias)
}

/// ReLU의 역전파
///
/// y = max(0, x)
/// ∂y/∂x = 1 if x > 0 else 0
pub fn relu_backward(grad_output: &Tensor3D, input: &Tensor3D) -> Tensor3D {
    let mut grad_input = grad_output.clone();

    for (grad, &inp) in grad_input.iter_mut().zip(input.iter()) {
        if inp <= 0.0 {
            *grad = 0.0;
        }
    }

    grad_input
}

/// GELU의 역전파 (근사)
///
/// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
pub fn gelu_backward(grad_output: &Tensor3D, input: &Tensor3D) -> Tensor3D {
    use std::f32::consts::PI;

    let sqrt_2_over_pi = (2.0 / PI).sqrt();
    let c = 0.044715;

    input.mapv(|x| {
        let inner = sqrt_2_over_pi * (x + c * x.powi(3));
        let tanh_inner = inner.tanh();
        let sech2 = 1.0 - tanh_inner.powi(2);
        let d_inner = sqrt_2_over_pi * (1.0 + 3.0 * c * x.powi(2));

        0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner
    }) * grad_output
}

/// Softmax의 역전파
///
/// y = softmax(x)
/// Jacobian이 필요하지만, cross-entropy와 함께 사용하면
/// 간단해집니다: ∂L/∂x = y - target
pub fn softmax_cross_entropy_backward(
    softmax_output: &Tensor2D,  // softmax 출력
    targets: &[usize],          // 정답 클래스 인덱스
) -> Tensor2D {
    let mut grad = softmax_output.clone();

    for (i, &target) in targets.iter().enumerate() {
        grad[[i, target]] -= 1.0;
    }

    grad
}

/// Layer Normalization의 역전파
///
/// LayerNorm(x) = (x - μ) / σ * γ + β
pub fn layer_norm_backward(
    grad_output: &Tensor3D,
    input: &Tensor3D,
    gamma: &Tensor1D,
    eps: f32,
) -> (Tensor3D, Tensor1D, Tensor1D) {
    let shape = input.shape();
    let batch_size = shape[0];
    let seq_len = shape[1];
    let dim = shape[2];
    let n = dim as f32;

    let mut grad_input = zeros_3d(batch_size, seq_len, dim);
    let mut grad_gamma = zeros_1d(dim);
    let mut grad_beta = zeros_1d(dim);

    for b in 0..batch_size {
        for s in 0..seq_len {
            let x = input.slice(s![b, s, ..]);
            let dy = grad_output.slice(s![b, s, ..]);

            // 평균과 분산
            let mean: f32 = x.iter().sum::<f32>() / n;
            let var: f32 = x.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;
            let std_inv = 1.0 / (var + eps).sqrt();

            // 정규화된 값
            let x_norm: Vec<f32> = x.iter().map(|&v| (v - mean) * std_inv).collect();

            // ∂L/∂γ, ∂L/∂β
            for d in 0..dim {
                grad_gamma[d] += dy[d] * x_norm[d];
                grad_beta[d] += dy[d];
            }

            // ∂L/∂x (복잡한 식)
            let dy_gamma: Vec<f32> = (0..dim).map(|d| dy[d] * gamma[d]).collect();
            let sum_dy_gamma: f32 = dy_gamma.iter().sum();
            let sum_dy_gamma_x_norm: f32 = dy_gamma.iter().zip(&x_norm).map(|(a, b)| a * b).sum();

            for d in 0..dim {
                grad_input[[b, s, d]] = std_inv / n
                    * (n * dy_gamma[d] - sum_dy_gamma - x_norm[d] * sum_dy_gamma_x_norm);
            }
        }
    }

    (grad_input, grad_gamma, grad_beta)
}

/// Embedding의 역전파
///
/// 임베딩은 룩업이므로, 해당 인덱스의 기울기만 업데이트됩니다.
pub fn embedding_backward(
    grad_output: &Tensor3D,  // (batch, seq, embed_dim)
    token_ids: &[Vec<usize>],
    vocab_size: usize,
    embed_dim: usize,
) -> Tensor2D {
    let mut grad_weight = zeros_2d(vocab_size, embed_dim);

    for (b, seq) in token_ids.iter().enumerate() {
        for (s, &token_id) in seq.iter().enumerate() {
            for d in 0..embed_dim {
                grad_weight[[token_id, d]] += grad_output[[b, s, d]];
            }
        }
    }

    grad_weight
}

/// Dropout의 역전파
///
/// 순전파에서 사용한 마스크를 그대로 적용합니다.
pub fn dropout_backward(
    grad_output: &Tensor3D,
    mask: &Tensor3D,  // 순전파에서 저장한 마스크
    scale: f32,       // 1 / (1 - p)
) -> Tensor3D {
    grad_output * mask * scale
}

/// 잔차 연결의 역전파
///
/// y = x + f(x)
/// ∂L/∂x = ∂L/∂y + ∂L/∂f(x) · ∂f(x)/∂x
///
/// 잔차 연결에서 기울기는 그냥 더해집니다.
pub fn residual_backward(
    grad_output: &Tensor3D,
    grad_sublayer: &Tensor3D,
) -> Tensor3D {
    grad_output + grad_sublayer
}

/// Attention 역전파를 위한 중간값 저장 구조체
#[derive(Clone)]
pub struct AttentionCache {
    pub q: Tensor3D,
    pub k: Tensor3D,
    pub v: Tensor3D,
    pub attention_weights: Tensor3D,
    pub scale: f32,
}

/// Self-Attention의 역전파 (단순화 버전)
///
/// Attention(Q, K, V) = softmax(QK^T / √d) · V
///
/// 전체 역전파는 복잡하므로, 여기서는 구조만 설명합니다.
/// 실제 구현시에는 JAX/PyTorch의 autograd를 참조하세요.
pub fn attention_backward_notes() -> &'static str {
    r#"
Self-Attention 역전파 개요:

1. 출력에서 V로의 기울기:
   ∂L/∂V = Attention_weights^T · ∂L/∂output

2. Attention weights에서 QK로의 기울기:
   ∂L/∂(QK^T) = softmax의 jacobian × ∂L/∂attention_weights

3. Q, K로의 기울기:
   ∂L/∂Q = ∂L/∂(QK^T) · K / √d
   ∂L/∂K = Q^T · ∂L/∂(QK^T) / √d

4. 투영 행렬로의 기울기:
   ∂L/∂Wq = X^T · ∂L/∂Q
   ∂L/∂Wk = X^T · ∂L/∂K
   ∂L/∂Wv = X^T · ∂L/∂V
"#
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::randn_3d;

    #[test]
    fn test_relu_backward() {
        let input = Tensor3D::from_shape_vec(
            (1, 1, 4),
            vec![-1.0, 0.0, 1.0, 2.0],
        ).unwrap();
        let grad_output = Tensor3D::ones((1, 1, 4));

        let grad_input = relu_backward(&grad_output, &input);

        assert_eq!(grad_input[[0, 0, 0]], 0.0);  // x < 0
        assert_eq!(grad_input[[0, 0, 1]], 0.0);  // x = 0
        assert_eq!(grad_input[[0, 0, 2]], 1.0);  // x > 0
        assert_eq!(grad_input[[0, 0, 3]], 1.0);  // x > 0
    }

    #[test]
    fn test_softmax_cross_entropy_backward() {
        let softmax_output = Tensor2D::from_shape_vec(
            (2, 3),
            vec![0.7, 0.2, 0.1, 0.1, 0.1, 0.8],
        ).unwrap();
        let targets = vec![0, 2];

        let grad = softmax_cross_entropy_backward(&softmax_output, &targets);

        // 정답 위치에서 -1이 빠져야 함
        assert!((grad[[0, 0]] - (-0.3)).abs() < 1e-6);  // 0.7 - 1 = -0.3
        assert!((grad[[1, 2]] - (-0.2)).abs() < 1e-6);  // 0.8 - 1 = -0.2
    }
}
