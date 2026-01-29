//! 학습률 스케줄러 (Learning Rate Scheduler)
//!
//! 학습 과정에서 학습률을 동적으로 조절합니다.
//!
//! # 왜 필요한가?
//! - 초기: 높은 학습률로 빠르게 탐색
//! - 후기: 낮은 학습률로 미세 조정
//!
//! # LLM에서의 표준 스케줄
//! 1. Warmup: 0 → lr_max (선형 증가)
//! 2. Decay: lr_max → lr_min (코사인 감소)

use std::f32::consts::PI;

/// 학습률 스케줄러 트레잇
pub trait LRScheduler {
    /// 현재 스텝의 학습률 반환
    fn get_lr(&self, step: usize) -> f32;

    /// 스케줄 시각화 (디버깅용)
    fn visualize(&self, total_steps: usize) -> String {
        let mut result = String::new();
        result.push_str("Learning Rate Schedule:\n");

        let width = 60;
        let height = 10;

        // 최대/최소 학습률 찾기
        let mut max_lr = f32::NEG_INFINITY;
        let mut min_lr = f32::INFINITY;
        for step in 0..total_steps {
            let lr = self.get_lr(step);
            max_lr = max_lr.max(lr);
            min_lr = min_lr.min(lr);
        }

        // ASCII 차트 생성
        for row in 0..height {
            let threshold = max_lr - (max_lr - min_lr) * (row as f32 / height as f32);
            result.push_str(&format!("{:>8.6} |", threshold));

            for col in 0..width {
                let step = col * total_steps / width;
                let lr = self.get_lr(step);

                if lr >= threshold {
                    result.push('█');
                } else {
                    result.push(' ');
                }
            }
            result.push('\n');
        }

        result.push_str("         |");
        result.push_str(&"-".repeat(width));
        result.push('\n');
        result.push_str(&format!("          0{:>width$}\n", total_steps, width = width - 1));

        result
    }
}

/// Constant 학습률
///
/// 학습률이 일정합니다.
pub struct ConstantLR {
    pub lr: f32,
}

impl ConstantLR {
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }
}

impl LRScheduler for ConstantLR {
    fn get_lr(&self, _step: usize) -> f32 {
        self.lr
    }
}

/// Linear Warmup 스케줄러
///
/// 학습률을 선형으로 증가시킵니다.
///
/// # 수학적 정의
/// ```text
/// lr(t) = lr_start + (lr_end - lr_start) * t / warmup_steps
/// ```
pub struct LinearWarmup {
    pub lr_start: f32,
    pub lr_end: f32,
    pub warmup_steps: usize,
}

impl LinearWarmup {
    pub fn new(lr_end: f32, warmup_steps: usize) -> Self {
        Self {
            lr_start: 0.0,
            lr_end,
            warmup_steps,
        }
    }

    pub fn with_start(mut self, lr_start: f32) -> Self {
        self.lr_start = lr_start;
        self
    }
}

impl LRScheduler for LinearWarmup {
    fn get_lr(&self, step: usize) -> f32 {
        if step >= self.warmup_steps {
            return self.lr_end;
        }

        let progress = step as f32 / self.warmup_steps as f32;
        self.lr_start + (self.lr_end - self.lr_start) * progress
    }
}

/// Cosine Annealing 스케줄러
///
/// 학습률을 코사인 곡선으로 감소시킵니다.
///
/// # 수학적 정의
/// ```text
/// lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T))
/// ```
///
/// # 장점
/// - 부드러운 감소
/// - 학습 후반에 미세 조정
pub struct CosineAnnealing {
    pub lr_max: f32,
    pub lr_min: f32,
    pub total_steps: usize,
}

impl CosineAnnealing {
    pub fn new(lr_max: f32, total_steps: usize) -> Self {
        Self {
            lr_max,
            lr_min: 0.0,
            total_steps,
        }
    }

    pub fn with_min_lr(mut self, lr_min: f32) -> Self {
        self.lr_min = lr_min;
        self
    }
}

impl LRScheduler for CosineAnnealing {
    fn get_lr(&self, step: usize) -> f32 {
        if step >= self.total_steps {
            return self.lr_min;
        }

        let progress = step as f32 / self.total_steps as f32;
        self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + (PI * progress).cos())
    }
}

/// Warmup + Cosine Decay 스케줄러
///
/// LLM 학습의 표준 스케줄입니다.
///
/// # 단계
/// 1. Warmup: 0 → lr_max (선형)
/// 2. Decay: lr_max → lr_min (코사인)
///
/// # 시각화
/// ```text
/// LR
///  │    ╱─────╲
///  │   ╱       ╲
///  │  ╱         ╲
///  │ ╱           ╲
///  │╱             ╲____
///  └─────────────────────→ Steps
///    Warmup    Decay
/// ```
pub struct WarmupCosineDecay {
    pub lr_max: f32,
    pub lr_min: f32,
    pub warmup_steps: usize,
    pub total_steps: usize,
}

impl WarmupCosineDecay {
    pub fn new(lr_max: f32, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            lr_max,
            lr_min: 0.0,
            warmup_steps,
            total_steps,
        }
    }

    pub fn with_min_lr(mut self, lr_min: f32) -> Self {
        self.lr_min = lr_min;
        self
    }
}

impl LRScheduler for WarmupCosineDecay {
    fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            // Warmup 단계
            let progress = step as f32 / self.warmup_steps as f32;
            self.lr_max * progress
        } else if step >= self.total_steps {
            // 완료
            self.lr_min
        } else {
            // Cosine decay 단계
            let decay_steps = self.total_steps - self.warmup_steps;
            let decay_step = step - self.warmup_steps;
            let progress = decay_step as f32 / decay_steps as f32;

            self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + (PI * progress).cos())
        }
    }
}

/// Step Decay 스케줄러
///
/// 특정 스텝마다 학습률을 감소시킵니다.
///
/// # 예시
/// ```text
/// lr = lr_init * gamma^(step / step_size)
/// ```
pub struct StepDecay {
    pub lr_init: f32,
    pub gamma: f32,      // 감소 비율
    pub step_size: usize, // 감소 주기
}

impl StepDecay {
    pub fn new(lr_init: f32, gamma: f32, step_size: usize) -> Self {
        Self {
            lr_init,
            gamma,
            step_size,
        }
    }
}

impl LRScheduler for StepDecay {
    fn get_lr(&self, step: usize) -> f32 {
        let num_decays = step / self.step_size;
        self.lr_init * self.gamma.powi(num_decays as i32)
    }
}

/// Exponential Decay 스케줄러
///
/// 학습률을 지수적으로 감소시킵니다.
///
/// # 수학적 정의
/// ```text
/// lr(t) = lr_init * gamma^t
/// ```
pub struct ExponentialDecay {
    pub lr_init: f32,
    pub gamma: f32, // 보통 0.95-0.99
}

impl ExponentialDecay {
    pub fn new(lr_init: f32, gamma: f32) -> Self {
        Self { lr_init, gamma }
    }
}

impl LRScheduler for ExponentialDecay {
    fn get_lr(&self, step: usize) -> f32 {
        self.lr_init * self.gamma.powi(step as i32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_warmup() {
        let scheduler = LinearWarmup::new(0.001, 1000);

        assert!((scheduler.get_lr(0) - 0.0).abs() < 1e-6);
        assert!((scheduler.get_lr(500) - 0.0005).abs() < 1e-6);
        assert!((scheduler.get_lr(1000) - 0.001).abs() < 1e-6);
        assert!((scheduler.get_lr(2000) - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_annealing() {
        let scheduler = CosineAnnealing::new(0.001, 1000);

        assert!((scheduler.get_lr(0) - 0.001).abs() < 1e-6);
        assert!((scheduler.get_lr(500) - 0.0005).abs() < 1e-4);
        assert!((scheduler.get_lr(1000) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_warmup_cosine_decay() {
        let scheduler = WarmupCosineDecay::new(0.001, 100, 1000);

        // Warmup
        assert!(scheduler.get_lr(0) < scheduler.get_lr(50));
        assert!(scheduler.get_lr(50) < scheduler.get_lr(100));

        // Peak
        assert!((scheduler.get_lr(100) - 0.001).abs() < 1e-6);

        // Decay
        assert!(scheduler.get_lr(100) > scheduler.get_lr(500));
        assert!(scheduler.get_lr(500) > scheduler.get_lr(1000));
    }

    #[test]
    fn test_step_decay() {
        let scheduler = StepDecay::new(0.1, 0.1, 10);

        assert!((scheduler.get_lr(0) - 0.1).abs() < 1e-6);
        assert!((scheduler.get_lr(5) - 0.1).abs() < 1e-6);
        assert!((scheduler.get_lr(10) - 0.01).abs() < 1e-6);
        assert!((scheduler.get_lr(20) - 0.001).abs() < 1e-6);
    }
}
