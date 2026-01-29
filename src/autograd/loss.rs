//! 손실 함수 (Loss Functions)
//!
//! 모델의 예측과 실제 정답 사이의 차이를 측정합니다.
//! 이 값을 최소화하는 것이 학습의 목표입니다.
//!
//! # 주요 손실 함수
//! - Cross-Entropy: 분류 문제
//! - MSE: 회귀 문제
//! - Perplexity: 언어 모델 평가

use crate::tensor::{Tensor2D, Tensor3D, softmax_2d};
use ndarray::s;

/// Cross-Entropy Loss
///
/// 분류 문제의 표준 손실 함수입니다.
///
/// # 수학적 정의
/// ```text
/// L = -1/N * Σ log(p[target])
///
/// p = softmax(logits)
/// ```
///
/// # 언어 모델에서의 사용
/// - 각 위치에서 다음 토큰을 예측
/// - 정답 토큰의 확률을 최대화
pub fn cross_entropy_loss(
    logits: &Tensor3D,      // (batch, seq_len, vocab_size)
    targets: &[Vec<usize>], // (batch, seq_len) - 정답 토큰 ID
) -> f32 {
    let batch_size = logits.shape()[0];
    let seq_len = logits.shape()[1];
    let vocab_size = logits.shape()[2];

    let mut total_loss = 0.0;
    let mut count = 0;

    for b in 0..batch_size {
        for s in 0..seq_len {
            // Softmax 계산 (수치 안정성을 위해 최댓값 빼기)
            let logits_slice: Vec<f32> = (0..vocab_size)
                .map(|v| logits[[b, s, v]])
                .collect();

            let max_val = logits_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = logits_slice.iter().map(|&x| (x - max_val).exp()).sum();

            // 정답 토큰의 log probability
            let target = targets[b][s];
            let log_prob = (logits_slice[target] - max_val) - exp_sum.ln();

            total_loss -= log_prob;
            count += 1;
        }
    }

    total_loss / count as f32
}

/// Cross-Entropy Loss with ignore index
///
/// 패딩 토큰 등 특정 인덱스를 무시합니다.
pub fn cross_entropy_loss_ignore(
    logits: &Tensor3D,
    targets: &[Vec<usize>],
    ignore_index: usize,
) -> f32 {
    let batch_size = logits.shape()[0];
    let seq_len = logits.shape()[1];
    let vocab_size = logits.shape()[2];

    let mut total_loss = 0.0;
    let mut count = 0;

    for b in 0..batch_size {
        for s in 0..seq_len {
            let target = targets[b][s];

            // 무시할 인덱스는 건너뜀
            if target == ignore_index {
                continue;
            }

            let logits_slice: Vec<f32> = (0..vocab_size)
                .map(|v| logits[[b, s, v]])
                .collect();

            let max_val = logits_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = logits_slice.iter().map(|&x| (x - max_val).exp()).sum();

            let log_prob = (logits_slice[target] - max_val) - exp_sum.ln();

            total_loss -= log_prob;
            count += 1;
        }
    }

    if count == 0 {
        return 0.0;
    }

    total_loss / count as f32
}

/// Perplexity (혼란도)
///
/// 언어 모델의 평가 지표입니다.
///
/// # 정의
/// ```text
/// PPL = exp(Cross-Entropy Loss)
/// ```
///
/// # 의미
/// - PPL = 1: 완벽한 예측
/// - PPL = N: N개 중 랜덤 선택과 비슷
/// - 낮을수록 좋음
pub fn perplexity(loss: f32) -> f32 {
    loss.exp()
}

/// 언어 모델 손실 계산
///
/// 다음 토큰 예측 태스크의 손실을 계산합니다.
///
/// # 입력
/// - input_ids: [t₁, t₂, ..., tₙ]
/// - targets: [t₂, t₃, ..., tₙ₊₁] (한 칸 시프트)
pub fn language_model_loss(
    logits: &Tensor3D,      // 모델 출력 (batch, seq_len, vocab_size)
    input_ids: &[Vec<usize>], // 입력 토큰
) -> f32 {
    // 타겟은 입력을 한 칸 시프트한 것
    let targets: Vec<Vec<usize>> = input_ids
        .iter()
        .map(|seq| seq[1..].to_vec())
        .collect();

    // 로짓도 마지막 위치 제외 (마지막 예측은 평가 불가)
    let seq_len = logits.shape()[1];
    let logits_shifted = logits.slice(s![.., ..(seq_len - 1), ..]).to_owned();

    cross_entropy_loss(&logits_shifted, &targets)
}

/// Label Smoothing
///
/// 정답에 100% 확신하는 대신, 일부 확률을 다른 클래스에 분배합니다.
/// 과적합을 방지하고 일반화 성능을 향상시킵니다.
///
/// # 수학적 정의
/// ```text
/// target_smooth = (1 - ε) * one_hot(target) + ε / num_classes
/// ```
pub fn cross_entropy_with_label_smoothing(
    logits: &Tensor3D,
    targets: &[Vec<usize>],
    smoothing: f32,  // 보통 0.1
) -> f32 {
    let batch_size = logits.shape()[0];
    let seq_len = logits.shape()[1];
    let vocab_size = logits.shape()[2];

    let mut total_loss = 0.0;
    let mut count = 0;

    for b in 0..batch_size {
        for s in 0..seq_len {
            let logits_slice: Vec<f32> = (0..vocab_size)
                .map(|v| logits[[b, s, v]])
                .collect();

            // Softmax (수치 안정성)
            let max_val = logits_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_vals: Vec<f32> = logits_slice.iter().map(|&x| (x - max_val).exp()).collect();
            let exp_sum: f32 = exp_vals.iter().sum();
            let log_probs: Vec<f32> = logits_slice
                .iter()
                .map(|&x| (x - max_val) - exp_sum.ln())
                .collect();

            // Label smoothing 적용
            let target = targets[b][s];
            let smooth_target = (1.0 - smoothing) / 1.0;  // 정답
            let smooth_other = smoothing / vocab_size as f32;  // 다른 클래스

            let mut loss = 0.0;
            for (i, &log_p) in log_probs.iter().enumerate() {
                if i == target {
                    loss -= smooth_target * log_p;
                }
                loss -= smooth_other * log_p;
            }

            total_loss += loss;
            count += 1;
        }
    }

    total_loss / count as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::randn_3d;

    #[test]
    fn test_cross_entropy_perfect_prediction() {
        // 정답에 매우 높은 로짓을 주면 손실이 낮아야 함
        let mut logits = Tensor3D::zeros((1, 2, 4));
        logits[[0, 0, 0]] = 10.0;  // 첫 번째 위치에서 클래스 0 예측
        logits[[0, 1, 1]] = 10.0;  // 두 번째 위치에서 클래스 1 예측

        let targets = vec![vec![0, 1]];
        let loss = cross_entropy_loss(&logits, &targets);

        assert!(loss < 0.01, "Loss should be low for perfect prediction");
    }

    #[test]
    fn test_cross_entropy_random_prediction() {
        // 균등 분포면 손실이 log(vocab_size)에 가까워야 함
        let logits = Tensor3D::zeros((1, 2, 4));  // 균등 로짓
        let targets = vec![vec![0, 1]];
        let loss = cross_entropy_loss(&logits, &targets);

        let expected = (4.0_f32).ln();  // log(4)
        assert!(
            (loss - expected).abs() < 0.1,
            "Loss should be close to log(vocab_size)"
        );
    }

    #[test]
    fn test_perplexity() {
        let loss = 2.0;
        let ppl = perplexity(loss);
        assert!((ppl - loss.exp()).abs() < 1e-6);
    }
}
