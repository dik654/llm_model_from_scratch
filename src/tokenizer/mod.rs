//! 토크나이저 모듈
//!
//! 텍스트를 토큰 ID로 변환하고, 역변환합니다.
//!
//! # 토큰화 방식
//! - 문자 단위: 가장 단순, 긴 시퀀스
//! - 단어 단위: OOV 문제
//! - 서브워드 (BPE): 균형잡힌 접근법
//!
//! # BPE (Byte Pair Encoding)
//! 가장 빈번한 문자 쌍을 반복적으로 병합하여
//! 적절한 크기의 어휘를 구축합니다.

pub mod bpe;
pub mod simple;

pub use bpe::BPETokenizer;
pub use simple::{CharTokenizer, SimpleTokenizer};

/// 토크나이저 트레잇
pub trait Tokenizer {
    /// 텍스트를 토큰 ID로 변환
    fn encode(&self, text: &str) -> Vec<usize>;

    /// 토큰 ID를 텍스트로 변환
    fn decode(&self, ids: &[usize]) -> String;

    /// 어휘 크기 반환
    fn vocab_size(&self) -> usize;

    /// 특수 토큰 ID 반환
    fn special_tokens(&self) -> SpecialTokens;
}

/// 특수 토큰 ID 모음
#[derive(Clone, Debug)]
pub struct SpecialTokens {
    /// 패딩 토큰
    pub pad: Option<usize>,
    /// 시작 토큰
    pub bos: Option<usize>,
    /// 종료 토큰
    pub eos: Option<usize>,
    /// 미지 토큰
    pub unk: Option<usize>,
    /// 마스크 토큰 (BERT용)
    pub mask: Option<usize>,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            pad: None,
            bos: None,
            eos: None,
            unk: None,
            mask: None,
        }
    }
}
