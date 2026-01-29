//! Adam 옵티마이저
//!
//! Adaptive Moment Estimation - 가장 널리 사용되는 옵티마이저입니다.
//!
//! # 핵심 아이디어
//! - 1차 모멘트 (평균): 기울기의 방향
//! - 2차 모멘트 (분산): 기울기의 크기
//! - 각 파라미터마다 적응적 학습률
//!
//! # 수학적 정의
//! ```text
//! m = β₁ · m + (1 - β₁) · g        (1차 모멘트)
//! v = β₂ · v + (1 - β₂) · g²       (2차 모멘트)
//!
//! m̂ = m / (1 - β₁^t)              (편향 보정)
//! v̂ = v / (1 - β₂^t)
//!
//! θ = θ - lr · m̂ / (√v̂ + ε)
//! ```
//!
//! # 기본 하이퍼파라미터
//! - lr = 0.001
//! - β₁ = 0.9
//! - β₂ = 0.999
//! - ε = 1e-8

use crate::tensor::{Tensor1D, Tensor2D};

/// Adam 옵티마이저
#[derive(Clone)]
pub struct Adam {
    /// 학습률
    pub lr: f32,
    /// 1차 모멘트 계수
    pub beta1: f32,
    /// 2차 모멘트 계수
    pub beta2: f32,
    /// 수치 안정성을 위한 작은 값
    pub eps: f32,
    /// 현재 스텝
    pub t: usize,
    /// 1차 모멘트 (2D)
    m_2d: Vec<Tensor2D>,
    /// 2차 모멘트 (2D)
    v_2d: Vec<Tensor2D>,
    /// 1차 모멘트 (1D)
    m_1d: Vec<Tensor1D>,
    /// 2차 모멘트 (1D)
    v_1d: Vec<Tensor1D>,
}

impl Adam {
    /// 새로운 Adam 옵티마이저 생성
    ///
    /// # 인자
    /// - `lr`: 학습률 (기본: 0.001)
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0,
            m_2d: Vec::new(),
            v_2d: Vec::new(),
            m_1d: Vec::new(),
            v_1d: Vec::new(),
        }
    }

    /// 하이퍼파라미터 설정
    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// 2D 파라미터 업데이트
    pub fn step_2d(&mut self, param: &mut Tensor2D, grad: &Tensor2D, idx: usize) {
        // 상태 초기화
        while self.m_2d.len() <= idx {
            self.m_2d.push(Tensor2D::zeros(param.raw_dim()));
            self.v_2d.push(Tensor2D::zeros(param.raw_dim()));
        }

        let m = &mut self.m_2d[idx];
        let v = &mut self.v_2d[idx];

        // m = β₁ · m + (1 - β₁) · g
        *m = &(&*m * self.beta1) + &(grad * (1.0 - self.beta1));

        // v = β₂ · v + (1 - β₂) · g²
        let grad_sq = grad.mapv(|x| x * x);
        *v = &(&*v * self.beta2) + &(grad_sq * (1.0 - self.beta2));

        // 편향 보정
        let t = (self.t + 1) as f32;
        let m_hat = &*m / (1.0 - self.beta1.powf(t));
        let v_hat = &*v / (1.0 - self.beta2.powf(t));

        // θ = θ - lr · m̂ / (√v̂ + ε)
        let update = m_hat / (v_hat.mapv(|x| x.sqrt() + self.eps));
        *param = &*param - &(update * self.lr);
    }

    /// 1D 파라미터 업데이트
    pub fn step_1d(&mut self, param: &mut Tensor1D, grad: &Tensor1D, idx: usize) {
        while self.m_1d.len() <= idx {
            self.m_1d.push(Tensor1D::zeros(param.raw_dim()));
            self.v_1d.push(Tensor1D::zeros(param.raw_dim()));
        }

        let m = &mut self.m_1d[idx];
        let v = &mut self.v_1d[idx];

        *m = &(&*m * self.beta1) + &(grad * (1.0 - self.beta1));

        let grad_sq = grad.mapv(|x| x * x);
        *v = &(&*v * self.beta2) + &(grad_sq * (1.0 - self.beta2));

        let t = (self.t + 1) as f32;
        let m_hat = &*m / (1.0 - self.beta1.powf(t));
        let v_hat = &*v / (1.0 - self.beta2.powf(t));

        let update = m_hat / (v_hat.mapv(|x| x.sqrt() + self.eps));
        *param = &*param - &(update * self.lr);
    }

    /// 스텝 카운터 증가
    pub fn increment_step(&mut self) {
        self.t += 1;
    }

    /// 상태 초기화
    pub fn reset(&mut self) {
        self.t = 0;
        self.m_2d.clear();
        self.v_2d.clear();
        self.m_1d.clear();
        self.v_1d.clear();
    }
}

/// AdamW 옵티마이저
///
/// Adam에 decoupled weight decay를 추가한 버전입니다.
/// LLM 학습의 표준 옵티마이저입니다.
///
/// # Adam vs AdamW
/// - Adam: L2 정규화가 적응적 학습률과 결합됨
/// - AdamW: weight decay를 별도로 적용 (분리)
///
/// # 수학적 정의
/// ```text
/// Adam 업데이트 후:
/// θ = θ - λ · θ    (weight decay)
/// ```
#[derive(Clone)]
pub struct AdamW {
    /// 내부 Adam
    pub adam: Adam,
    /// Weight decay 계수
    pub weight_decay: f32,
}

impl AdamW {
    /// 새로운 AdamW 옵티마이저 생성
    ///
    /// # 인자
    /// - `lr`: 학습률
    /// - `weight_decay`: 가중치 감쇠 (예: 0.01)
    pub fn new(lr: f32, weight_decay: f32) -> Self {
        Self {
            adam: Adam::new(lr),
            weight_decay,
        }
    }

    /// 하이퍼파라미터 설정
    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.adam = self.adam.with_betas(beta1, beta2);
        self
    }

    /// 2D 파라미터 업데이트
    pub fn step_2d(&mut self, param: &mut Tensor2D, grad: &Tensor2D, idx: usize) {
        // Adam 업데이트
        self.adam.step_2d(param, grad, idx);

        // Decoupled weight decay
        param.mapv_inplace(|p| p * (1.0 - self.adam.lr * self.weight_decay));
    }

    /// 1D 파라미터 업데이트
    pub fn step_1d(&mut self, param: &mut Tensor1D, grad: &Tensor1D, idx: usize) {
        self.adam.step_1d(param, grad, idx);
        param.mapv_inplace(|p| p * (1.0 - self.adam.lr * self.weight_decay));
    }

    /// 스텝 카운터 증가
    pub fn increment_step(&mut self) {
        self.adam.increment_step();
    }

    /// 상태 초기화
    pub fn reset(&mut self) {
        self.adam.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adam_step() {
        let mut param = Tensor2D::ones((3, 3));
        let grad = Tensor2D::ones((3, 3)) * 0.1;

        let mut adam = Adam::new(0.01);

        // 여러 스텝 실행
        for _ in 0..10 {
            adam.step_2d(&mut param, &grad, 0);
            adam.increment_step();
        }

        // 파라미터가 감소해야 함
        assert!(param[[0, 0]] < 1.0);
    }

    #[test]
    fn test_adamw_weight_decay() {
        let mut param1 = Tensor2D::ones((3, 3)) * 2.0;
        let mut param2 = param1.clone();
        let grad = Tensor2D::zeros((3, 3));  // 기울기 없음

        let mut adam = Adam::new(0.01);
        let mut adamw = AdamW::new(0.01, 0.1);

        adam.step_2d(&mut param1, &grad, 0);
        adamw.step_2d(&mut param2, &grad, 0);

        // Adam은 기울기 없으면 변화 없음
        assert!((param1[[0, 0]] - 2.0).abs() < 1e-6);

        // AdamW는 weight decay로 감소
        assert!(param2[[0, 0]] < 2.0);
    }
}
