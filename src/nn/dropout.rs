//! Dropout 레이어
//!
//! 과적합(Overfitting)을 방지하기 위한 정규화 기법입니다.
//!
//! # 동작 원리
//! - **학습 시**: 무작위로 뉴런을 비활성화 (0으로 설정)
//! - **추론 시**: 모든 뉴런 활성화 (변화 없음)
//!
//! # 수학적 정의
//! ```text
//! 학습 시:
//! mask ~ Bernoulli(1-p)
//! y = x * mask / (1-p)
//!
//! 추론 시:
//! y = x
//! ```
//!
//! # 스케일링 (Inverted Dropout)
//! 학습 시 1/(1-p)로 스케일하면 추론 시 추가 연산이 필요 없습니다.
//! - 학습: 일부 비활성화 + 나머지 확대
//! - 추론: 그대로 사용
//!
//! # 효과
//! - 뉴런 간 공적응(co-adaptation) 방지
//! - 앙상블 효과 (다양한 서브네트워크 학습)
//! - 더 강건한 특징 학습

use crate::tensor::Tensor3D;
use rand::Rng;
use super::Module;

/// Dropout 레이어
///
/// # 예시
/// ```
/// let dropout = Dropout::new(0.1);  // 10% 확률로 비활성화
/// dropout.set_training(true);       // 학습 모드
/// let y = dropout.forward(&x);      // 일부 값이 0이 됨
///
/// dropout.set_training(false);      // 추론 모드
/// let y = dropout.forward(&x);      // 변화 없음
/// ```
#[derive(Clone)]
pub struct Dropout {
    /// 드롭아웃 확률 (비활성화될 확률)
    pub p: f32,
    /// 학습 모드 여부
    pub training: bool,
}

impl Dropout {
    /// 새로운 Dropout 레이어 생성
    ///
    /// # 인자
    /// - `p`: 드롭아웃 확률 (0.0 ~ 1.0)
    ///   - 0.0: 드롭아웃 없음
    ///   - 0.1: 10% 확률로 비활성화 (일반적인 값)
    ///   - 0.5: 50% 확률로 비활성화 (강한 정규화)
    pub fn new(p: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&p),
            "Dropout probability must be between 0 and 1, got {}",
            p
        );

        Self { p, training: true }
    }

    /// 학습/추론 모드 설정
    ///
    /// - `true`: 학습 모드 (드롭아웃 적용)
    /// - `false`: 추론 모드 (드롭아웃 비활성화)
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// 학습 모드 여부 반환
    pub fn is_training(&self) -> bool {
        self.training
    }
}

impl Module for Dropout {
    /// Dropout 적용
    ///
    /// # 학습 모드
    /// 1. 각 요소에 대해 Bernoulli(1-p) 마스크 생성
    /// 2. 마스크가 0인 위치는 0으로 설정
    /// 3. 마스크가 1인 위치는 1/(1-p)로 스케일
    ///
    /// # 추론 모드
    /// 입력을 그대로 반환
    fn forward(&self, x: &Tensor3D) -> Tensor3D {
        // p가 0이면 드롭아웃 없음
        if self.p == 0.0 {
            return x.clone();
        }

        // 추론 모드면 그대로 반환
        if !self.training {
            return x.clone();
        }

        // 학습 모드: 드롭아웃 적용
        let mut rng = rand::thread_rng();
        let scale = 1.0 / (1.0 - self.p);

        x.mapv(|v| {
            if rng.gen::<f32>() < self.p {
                0.0  // 비활성화
            } else {
                v * scale  // 스케일 업
            }
        })
    }

    fn num_parameters(&self) -> usize {
        0  // Dropout은 학습 파라미터가 없음
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::ones_1d;
    use ndarray::Array3;

    #[test]
    fn test_dropout_training_mode() {
        let dropout = Dropout::new(0.5);
        let x = Array3::ones((10, 10, 100));
        let y = dropout.forward(&x);

        // 일부는 0이어야 함
        let zeros = y.iter().filter(|&&v| v == 0.0).count();
        let nonzeros = y.iter().filter(|&&v| v != 0.0).count();

        assert!(zeros > 0, "Should have some zeros");
        assert!(nonzeros > 0, "Should have some non-zeros");
    }

    #[test]
    fn test_dropout_inference_mode() {
        let mut dropout = Dropout::new(0.5);
        dropout.set_training(false);

        let x = Array3::ones((2, 3, 4));
        let y = dropout.forward(&x);

        // 추론 모드에서는 변화 없음
        assert_eq!(x, y);
    }

    #[test]
    fn test_dropout_scale() {
        let dropout = Dropout::new(0.5);
        let x = Array3::ones((100, 100, 100));
        let y = dropout.forward(&x);

        // 평균이 원래 값(1.0)에 가까워야 함 (스케일링 때문)
        let mean = y.mean().unwrap();
        assert!(
            (mean - 1.0).abs() < 0.1,
            "Mean should be close to 1.0, got {}",
            mean
        );
    }

    #[test]
    fn test_dropout_zero_probability() {
        let dropout = Dropout::new(0.0);
        let x = Array3::from_elem((2, 3, 4), 5.0);
        let y = dropout.forward(&x);

        assert_eq!(x, y);
    }
}
