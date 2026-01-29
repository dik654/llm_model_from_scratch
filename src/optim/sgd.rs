//! SGD (Stochastic Gradient Descent) 옵티마이저
//!
//! 가장 기본적인 최적화 알고리즘입니다.
//!
//! # 기본 SGD
//! ```text
//! θ = θ - lr · ∇L(θ)
//! ```
//!
//! # Momentum SGD
//! ```text
//! v = μ · v - lr · ∇L(θ)
//! θ = θ + v
//! ```
//!
//! 모멘텀은 이전 업데이트 방향을 기억하여
//! 진동을 줄이고 수렴을 가속합니다.

use crate::tensor::{Tensor1D, Tensor2D};

/// SGD 옵티마이저
///
/// # 예시
/// ```
/// let mut sgd = SGD::new(0.01, 0.9);
///
/// // 기울기 계산 후
/// sgd.step_2d(&mut weight, &grad_weight);
/// sgd.step_1d(&mut bias, &grad_bias);
/// ```
pub struct SGD {
    /// 학습률
    pub lr: f32,
    /// 모멘텀 계수 (0이면 기본 SGD)
    pub momentum: f32,
    /// 가중치 감쇠 (L2 정규화)
    pub weight_decay: f32,
    /// 모멘텀 버퍼 (2D)
    velocity_2d: Vec<Tensor2D>,
    /// 모멘텀 버퍼 (1D)
    velocity_1d: Vec<Tensor1D>,
}

impl SGD {
    /// 새로운 SGD 옵티마이저 생성
    ///
    /// # 인자
    /// - `lr`: 학습률 (예: 0.01)
    /// - `momentum`: 모멘텀 계수 (예: 0.9, 0이면 비활성화)
    pub fn new(lr: f32, momentum: f32) -> Self {
        Self {
            lr,
            momentum,
            weight_decay: 0.0,
            velocity_2d: Vec::new(),
            velocity_1d: Vec::new(),
        }
    }

    /// Weight decay 설정
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// 2D 파라미터 업데이트
    pub fn step_2d(&mut self, param: &mut Tensor2D, grad: &Tensor2D) {
        // Weight decay
        if self.weight_decay > 0.0 {
            param.mapv_inplace(|p| p * (1.0 - self.lr * self.weight_decay));
        }

        if self.momentum == 0.0 {
            // 기본 SGD
            *param = &*param - &(grad * self.lr);
        } else {
            // Momentum SGD
            // 버퍼가 없으면 초기화
            if self.velocity_2d.is_empty() {
                self.velocity_2d.push(Tensor2D::zeros(param.raw_dim()));
            }

            let idx = self.velocity_2d.len() - 1;
            let v = &mut self.velocity_2d[idx];

            // v = momentum * v + grad
            *v = &(&*v * self.momentum) + grad;

            // param = param - lr * v
            *param = &*param - &(&*v * self.lr);
        }
    }

    /// 1D 파라미터 업데이트
    pub fn step_1d(&mut self, param: &mut Tensor1D, grad: &Tensor1D) {
        if self.momentum == 0.0 {
            *param = &*param - &(grad * self.lr);
        } else {
            if self.velocity_1d.is_empty() {
                self.velocity_1d.push(Tensor1D::zeros(param.raw_dim()));
            }

            let idx = self.velocity_1d.len() - 1;
            let v = &mut self.velocity_1d[idx];

            *v = &(&*v * self.momentum) + grad;
            *param = &*param - &(&*v * self.lr);
        }
    }

    /// 모멘텀 버퍼 초기화
    pub fn zero_momentum(&mut self) {
        for v in &mut self.velocity_2d {
            v.fill(0.0);
        }
        for v in &mut self.velocity_1d {
            v.fill(0.0);
        }
    }
}

/// Nesterov Momentum SGD
///
/// 모멘텀 방향으로 먼저 이동한 후 기울기를 계산합니다.
/// 표준 모멘텀보다 더 빠른 수렴을 보입니다.
///
/// # 수학적 정의
/// ```text
/// v = μ · v + ∇L(θ + μ · v)
/// θ = θ - lr · v
/// ```
pub struct NesterovSGD {
    /// 학습률
    pub lr: f32,
    /// 모멘텀 계수
    pub momentum: f32,
    /// 모멘텀 버퍼
    velocity_2d: Vec<Tensor2D>,
    velocity_1d: Vec<Tensor1D>,
}

impl NesterovSGD {
    pub fn new(lr: f32, momentum: f32) -> Self {
        Self {
            lr,
            momentum,
            velocity_2d: Vec::new(),
            velocity_1d: Vec::new(),
        }
    }

    /// 2D 파라미터 업데이트 (Nesterov)
    pub fn step_2d(&mut self, param: &mut Tensor2D, grad: &Tensor2D) {
        if self.velocity_2d.is_empty() {
            self.velocity_2d.push(Tensor2D::zeros(param.raw_dim()));
        }

        let idx = self.velocity_2d.len() - 1;
        let v = &mut self.velocity_2d[idx];

        // v_prev = v
        let v_prev = v.clone();

        // v = momentum * v - lr * grad
        *v = &(&v_prev * self.momentum) - &(grad * self.lr);

        // param = param + (-momentum * v_prev) + (1 + momentum) * v
        //       = param + v + momentum * (v - v_prev)
        let update = &*v + &(&(&*v - &v_prev) * self.momentum);
        *param = &*param + &update;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::randn_2d;

    #[test]
    fn test_sgd_step() {
        let mut param = Tensor2D::ones((3, 3));
        let grad = Tensor2D::ones((3, 3)) * 0.1;

        let mut sgd = SGD::new(0.1, 0.0);
        sgd.step_2d(&mut param, &grad);

        // param = 1.0 - 0.1 * 0.1 = 0.99
        assert!((param[[0, 0]] - 0.99).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_momentum() {
        let mut param = Tensor2D::ones((3, 3));
        let grad = Tensor2D::ones((3, 3)) * 0.1;

        let mut sgd = SGD::new(0.1, 0.9);

        // 첫 번째 스텝
        sgd.step_2d(&mut param, &grad);

        // 두 번째 스텝 - 모멘텀으로 더 많이 움직임
        let before = param[[0, 0]];
        sgd.step_2d(&mut param, &grad);
        let after = param[[0, 0]];

        assert!(before - after > 0.01, "Momentum should accelerate");
    }
}
