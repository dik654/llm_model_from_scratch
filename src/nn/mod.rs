//! 신경망 레이어 모듈
//!
//! LLM을 구성하는 기본 레이어들을 정의합니다.
//!
//! # 주요 레이어
//! - `Linear`: 선형 변환 (y = xW + b)
//! - `Embedding`: 이산 토큰을 연속 벡터로 변환
//! - `LayerNorm`: 레이어 정규화
//! - `Dropout`: 과적합 방지

mod linear;
mod embedding;
mod layer_norm;
mod dropout;

pub use linear::Linear;
pub use embedding::Embedding;
pub use layer_norm::LayerNorm;
pub use dropout::Dropout;

use crate::tensor::Tensor3D;

/// 모든 신경망 레이어가 구현해야 하는 트레잇
///
/// forward 메서드를 통해 입력을 출력으로 변환합니다.
pub trait Module {
    /// 순전파 (Forward Pass)
    ///
    /// 입력 텐서를 받아 출력 텐서를 반환합니다.
    fn forward(&self, x: &Tensor3D) -> Tensor3D;

    /// 파라미터 개수 반환
    fn num_parameters(&self) -> usize;
}
