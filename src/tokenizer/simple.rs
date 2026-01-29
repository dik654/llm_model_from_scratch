//! 간단한 토크나이저 구현
//!
//! 교육 목적으로 단순한 토크나이저를 제공합니다.

use std::collections::HashMap;
use super::{Tokenizer, SpecialTokens};

/// 문자 단위 토크나이저
///
/// 각 문자(또는 바이트)를 하나의 토큰으로 처리합니다.
///
/// # 장점
/// - 구현이 매우 단순
/// - OOV 없음
///
/// # 단점
/// - 시퀀스가 길어짐
/// - 의미 단위 학습이 어려움
#[derive(Clone)]
pub struct CharTokenizer {
    /// 문자 → ID 매핑
    char_to_id: HashMap<char, usize>,
    /// ID → 문자 매핑
    id_to_char: HashMap<usize, char>,
    /// 어휘 크기
    vocab_size: usize,
    /// 특수 토큰
    special_tokens: SpecialTokens,
}

impl CharTokenizer {
    /// 새로운 문자 토크나이저 생성 (ASCII 기반)
    pub fn new() -> Self {
        let mut char_to_id = HashMap::new();
        let mut id_to_char = HashMap::new();

        // ASCII 문자 (0-127)
        for i in 0..128u8 {
            let c = i as char;
            char_to_id.insert(c, i as usize);
            id_to_char.insert(i as usize, c);
        }

        // 특수 토큰 추가
        let special_tokens = SpecialTokens {
            pad: Some(128),
            bos: Some(129),
            eos: Some(130),
            unk: Some(131),
            mask: None,
        };

        char_to_id.insert('\0', 128); // PAD
        id_to_char.insert(128, '\0');
        // BOS, EOS, UNK는 특수 제어 문자로

        Self {
            char_to_id,
            id_to_char,
            vocab_size: 256, // 확장 ASCII
            special_tokens,
        }
    }

    /// 텍스트에서 어휘 구축
    pub fn from_text(text: &str) -> Self {
        let mut char_to_id = HashMap::new();
        let mut id_to_char = HashMap::new();

        // 특수 토큰 먼저 추가
        char_to_id.insert('\0', 0); // PAD
        id_to_char.insert(0, '\0');

        let mut next_id = 1;

        for c in text.chars() {
            if !char_to_id.contains_key(&c) {
                char_to_id.insert(c, next_id);
                id_to_char.insert(next_id, c);
                next_id += 1;
            }
        }

        let special_tokens = SpecialTokens {
            pad: Some(0),
            bos: None,
            eos: None,
            unk: None,
            mask: None,
        };

        Self {
            vocab_size: next_id,
            char_to_id,
            id_to_char,
            special_tokens,
        }
    }
}

impl Default for CharTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for CharTokenizer {
    fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .map(|c| {
                *self.char_to_id.get(&c).unwrap_or(
                    &self.special_tokens.unk.unwrap_or(0)
                )
            })
            .collect()
    }

    fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .filter_map(|&id| self.id_to_char.get(&id))
            .collect()
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn special_tokens(&self) -> SpecialTokens {
        self.special_tokens.clone()
    }
}

/// 단어 단위 토크나이저
///
/// 공백으로 분리된 단어를 토큰으로 처리합니다.
///
/// # 장점
/// - 의미 단위 처리
/// - 짧은 시퀀스
///
/// # 단점
/// - OOV 문제
/// - 어휘 크기 폭발
pub struct SimpleTokenizer {
    /// 단어 → ID 매핑
    word_to_id: HashMap<String, usize>,
    /// ID → 단어 매핑
    id_to_word: HashMap<usize, String>,
    /// 특수 토큰
    special_tokens: SpecialTokens,
}

impl SimpleTokenizer {
    /// 텍스트에서 어휘 구축
    pub fn from_text(text: &str, min_freq: usize) -> Self {
        // 단어 빈도 계산
        let mut word_freq: HashMap<String, usize> = HashMap::new();
        for word in text.split_whitespace() {
            let word = word.to_lowercase();
            *word_freq.entry(word).or_insert(0) += 1;
        }

        let mut word_to_id = HashMap::new();
        let mut id_to_word = HashMap::new();

        // 특수 토큰 먼저 추가
        let special = vec!["<pad>", "<bos>", "<eos>", "<unk>"];
        for (i, &token) in special.iter().enumerate() {
            word_to_id.insert(token.to_string(), i);
            id_to_word.insert(i, token.to_string());
        }

        let mut next_id = special.len();

        // 빈도가 높은 단어만 추가
        for (word, freq) in word_freq {
            if freq >= min_freq {
                word_to_id.insert(word.clone(), next_id);
                id_to_word.insert(next_id, word);
                next_id += 1;
            }
        }

        let special_tokens = SpecialTokens {
            pad: Some(0),
            bos: Some(1),
            eos: Some(2),
            unk: Some(3),
            mask: None,
        };

        Self {
            word_to_id,
            id_to_word,
            special_tokens,
        }
    }
}

impl Tokenizer for SimpleTokenizer {
    fn encode(&self, text: &str) -> Vec<usize> {
        text.split_whitespace()
            .map(|word| {
                let word = word.to_lowercase();
                *self.word_to_id.get(&word).unwrap_or(
                    &self.special_tokens.unk.unwrap()
                )
            })
            .collect()
    }

    fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .filter_map(|&id| self.id_to_word.get(&id))
            .cloned()
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn vocab_size(&self) -> usize {
        self.word_to_id.len()
    }

    fn special_tokens(&self) -> SpecialTokens {
        self.special_tokens.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_char_tokenizer() {
        let tokenizer = CharTokenizer::new();

        let text = "Hello";
        let encoded = tokenizer.encode(text);
        let decoded = tokenizer.decode(&encoded);

        assert_eq!(decoded, text);
    }

    #[test]
    fn test_char_tokenizer_from_text() {
        let text = "hello world";
        let tokenizer = CharTokenizer::from_text(text);

        let encoded = tokenizer.encode("hello");
        let decoded = tokenizer.decode(&encoded);

        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_simple_tokenizer() {
        let text = "hello world hello rust world";
        let tokenizer = SimpleTokenizer::from_text(text, 1);

        let encoded = tokenizer.encode("hello world");
        let decoded = tokenizer.decode(&encoded);

        assert_eq!(decoded, "hello world");
    }
}
