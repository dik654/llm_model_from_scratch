//! Feed-Forward Network (FFN)
//!
//! Transformer의 위치별 Feed-Forward 네트워크입니다.
//!
//! # 구조
//! ```text
//! FFN(x) = W₂ · GELU(W₁ · x + b₁) + b₂
//! ```
//!
//! # 차원
//! - 입력: d_model (예: 512)
//! - 확장: d_ff (예: 2048, 보통 4배)
//! - 출력: d_model (예: 512)
//!
//! # 역할
//! - Attention은 토큰 간 관계를 학습
//! - FFN은 각 토큰의 표현을 변환
//! - 비선형성 추가 (GELU/ReLU)

use crate::tensor::{Tensor3D, gelu};
use crate::nn::{Linear, Dropout, Module};

/// Position-wise Feed-Forward Network
///
/// # 예시
/// ```
/// let ffn = FeedForward::new(512, 2048, 0.1);
/// let x = randn_3d(32, 10, 512);  // (batch, seq, d_model)
/// let y = ffn.forward(&x);        // (batch, seq, d_model)
/// ```
#[derive(Clone)]
pub struct FeedForward {
    /// 첫 번째 선형 레이어 (d_model → d_ff)
    pub fc1: Linear,
    /// 두 번째 선형 레이어 (d_ff → d_model)
    pub fc2: Linear,
    /// 드롭아웃
    pub dropout: Dropout,
    /// 모델 차원
    pub d_model: usize,
    /// FFN 확장 차원
    pub d_ff: usize,
}

impl FeedForward {
    /// 새로운 Feed-Forward Network 생성
    ///
    /// # 인자
    /// - `d_model`: 모델 차원
    /// - `d_ff`: FFN 내부 차원 (보통 d_model의 4배)
    /// - `dropout_p`: 드롭아웃 확률
    pub fn new(d_model: usize, d_ff: usize, dropout_p: f32) -> Self {
        Self {
            fc1: Linear::new(d_model, d_ff),
            fc2: Linear::new(d_ff, d_model),
            dropout: Dropout::new(dropout_p),
            d_model,
            d_ff,
        }
    }

    /// 학습/추론 모드 설정
    pub fn set_training(&mut self, training: bool) {
        self.dropout.set_training(training);
    }

    /// 파라미터 개수 반환
    pub fn num_parameters(&self) -> usize {
        self.fc1.num_parameters() + self.fc2.num_parameters()
    }
}

impl Module for FeedForward {
    /// FFN 순전파
    ///
    /// # 동작 과정
    /// 1. 첫 번째 선형 변환: d_model → d_ff
    /// 2. GELU 활성화 함수
    /// 3. 드롭아웃
    /// 4. 두 번째 선형 변환: d_ff → d_model
    fn forward(&self, x: &Tensor3D) -> Tensor3D {
        // x: (batch, seq, d_model)

        // 1. 첫 번째 선형 변환
        let hidden = self.fc1.forward(x);  // (batch, seq, d_ff)

        // 2. GELU 활성화
        let activated = gelu(&hidden);

        // 3. 드롭아웃
        let dropped = self.dropout.forward(&activated);

        // 4. 두 번째 선형 변환
        self.fc2.forward(&dropped)  // (batch, seq, d_model)
    }

    fn num_parameters(&self) -> usize {
        self.fc1.num_parameters() + self.fc2.num_parameters()
    }
}

/// SwiGLU Feed-Forward (Llama 스타일)
///
/// 더 현대적인 FFN 변형으로, gating 메커니즘을 추가합니다.
///
/// # 구조
/// ```text
/// SwiGLU(x) = (xW₁ ⊙ SiLU(xWg)) W₂
/// ```
///
/// - W₁: gate 투영
/// - Wg: up 투영
/// - W₂: down 투영
/// - SiLU(x) = x * sigmoid(x)
#[derive(Clone)]
pub struct SwiGLU {
    /// Gate 투영
    pub w_gate: Linear,
    /// Up 투영
    pub w_up: Linear,
    /// Down 투영
    pub w_down: Linear,
    /// 드롭아웃
    pub dropout: Dropout,
}

impl SwiGLU {
    /// 새로운 SwiGLU 생성
    pub fn new(d_model: usize, d_ff: usize, dropout_p: f32) -> Self {
        Self {
            w_gate: Linear::new(d_model, d_ff),
            w_up: Linear::new(d_model, d_ff),
            w_down: Linear::new(d_ff, d_model),
            dropout: Dropout::new(dropout_p),
        }
    }

    /// 학습/추론 모드 설정
    pub fn set_training(&mut self, training: bool) {
        self.dropout.set_training(training);
    }

    /// 파라미터 개수 반환
    pub fn num_parameters(&self) -> usize {
        self.w_gate.num_parameters()
            + self.w_up.num_parameters()
            + self.w_down.num_parameters()
    }
}

impl Module for SwiGLU {
    fn forward(&self, x: &Tensor3D) -> Tensor3D {
        // Gate: SiLU(xW_gate)
        let gate = self.w_gate.forward(x);
        let gate = gate.mapv(|v| v * (1.0 / (1.0 + (-v).exp()))); // SiLU

        // Up: xW_up
        let up = self.w_up.forward(x);

        // Element-wise multiply
        let hidden = &gate * &up;

        // Dropout
        let hidden = self.dropout.forward(&hidden);

        // Down projection
        self.w_down.forward(&hidden)
    }

    fn num_parameters(&self) -> usize {
        self.w_gate.num_parameters()
            + self.w_up.num_parameters()
            + self.w_down.num_parameters()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::randn_3d;

    #[test]
    fn test_feed_forward_shape() {
        let ffn = FeedForward::new(64, 256, 0.0);
        let x = randn_3d(2, 10, 64);
        let y = ffn.forward(&x);

        assert_eq!(y.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_swiglu_shape() {
        let swiglu = SwiGLU::new(64, 256, 0.0);
        let x = randn_3d(2, 10, 64);
        let y = swiglu.forward(&x);

        assert_eq!(y.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_ffn_parameters() {
        let ffn = FeedForward::new(512, 2048, 0.1);
        // fc1: 512*2048 + 2048, fc2: 2048*512 + 512
        let expected = 512 * 2048 + 2048 + 2048 * 512 + 512;
        assert_eq!(ffn.num_parameters(), expected);
    }
}
