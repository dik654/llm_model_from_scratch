//! BPE (Byte Pair Encoding) 토크나이저
//!
//! 서브워드 토크나이저의 대표적인 방법입니다.
//!
//! # 알고리즘 개요
//!
//! ## 학습 단계
//! 1. 문자 단위로 시작
//! 2. 가장 빈번한 인접 쌍을 찾음
//! 3. 해당 쌍을 새 토큰으로 병합
//! 4. 원하는 어휘 크기까지 반복
//!
//! ## 인코딩 단계
//! 1. 텍스트를 문자 단위로 분리
//! 2. 학습된 병합 규칙을 순서대로 적용
//!
//! # 예시
//! ```text
//! 학습:
//! "low lower lowest" →
//!   빈번한 쌍: ('l', 'o'), ('o', 'w'), ('e', 'r'), ...
//!   병합: 'l' + 'o' → 'lo'
//!   병합: 'lo' + 'w' → 'low'
//!   ...
//!
//! 인코딩:
//! "lower" → ['low', 'er']
//! ```

use std::collections::HashMap;
use super::{Tokenizer, SpecialTokens};
use serde::{Serialize, Deserialize};

/// BPE 토크나이저
#[derive(Clone, Serialize, Deserialize)]
pub struct BPETokenizer {
    /// 토큰 → ID 매핑
    token_to_id: HashMap<String, usize>,
    /// ID → 토큰 매핑
    id_to_token: HashMap<usize, String>,
    /// 병합 규칙 (순서대로 적용)
    /// (토큰1, 토큰2) → 병합된 토큰
    merges: Vec<(String, String)>,
    /// 특수 토큰
    #[serde(skip)]
    special_tokens: SpecialTokens,
}

impl BPETokenizer {
    /// 텍스트에서 BPE 토크나이저 학습
    ///
    /// # 인자
    /// - `text`: 학습 텍스트
    /// - `vocab_size`: 목표 어휘 크기
    pub fn train(text: &str, vocab_size: usize) -> Self {
        println!("BPE 학습 시작... 목표 어휘 크기: {}", vocab_size);

        // 1. 초기 어휘: 개별 문자 + 특수 토큰
        let mut token_to_id: HashMap<String, usize> = HashMap::new();
        let mut id_to_token: HashMap<usize, String> = HashMap::new();

        // 특수 토큰 추가
        let special = vec!["<pad>", "<unk>", "<bos>", "<eos>"];
        for (i, &token) in special.iter().enumerate() {
            token_to_id.insert(token.to_string(), i);
            id_to_token.insert(i, token.to_string());
        }

        let mut next_id = special.len();

        // 모든 문자를 어휘에 추가
        for c in text.chars() {
            let s = c.to_string();
            if !token_to_id.contains_key(&s) {
                token_to_id.insert(s.clone(), next_id);
                id_to_token.insert(next_id, s);
                next_id += 1;
            }
        }

        // 2. 단어를 문자 시퀀스로 분리
        let mut word_freqs: HashMap<Vec<String>, usize> = HashMap::new();

        for word in text.split_whitespace() {
            let chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            *word_freqs.entry(chars).or_insert(0) += 1;
        }

        let mut merges: Vec<(String, String)> = Vec::new();

        // 3. 병합 반복
        while next_id < vocab_size {
            // 인접 쌍 빈도 계산
            let mut pair_freqs: HashMap<(String, String), usize> = HashMap::new();

            for (word_tokens, freq) in &word_freqs {
                if word_tokens.len() < 2 {
                    continue;
                }

                for i in 0..word_tokens.len() - 1 {
                    let pair = (word_tokens[i].clone(), word_tokens[i + 1].clone());
                    *pair_freqs.entry(pair).or_insert(0) += freq;
                }
            }

            if pair_freqs.is_empty() {
                break;
            }

            // 가장 빈번한 쌍 찾기
            let best_pair = pair_freqs
                .iter()
                .max_by_key(|(_, &freq)| freq)
                .map(|(pair, _)| pair.clone());

            let Some((a, b)) = best_pair else {
                break;
            };

            // 새 토큰 생성
            let new_token = format!("{}{}", a, b);

            // 어휘에 추가
            token_to_id.insert(new_token.clone(), next_id);
            id_to_token.insert(next_id, new_token.clone());
            merges.push((a.clone(), b.clone()));
            next_id += 1;

            // 단어에서 쌍 병합
            let mut new_word_freqs: HashMap<Vec<String>, usize> = HashMap::new();

            for (word_tokens, freq) in word_freqs {
                let merged = merge_pair(&word_tokens, &a, &b, &new_token);
                *new_word_freqs.entry(merged).or_insert(0) += freq;
            }

            word_freqs = new_word_freqs;

            if merges.len() % 100 == 0 {
                println!("  {} 병합 완료, 어휘 크기: {}", merges.len(), next_id);
            }
        }

        println!("BPE 학습 완료! 최종 어휘 크기: {}, 병합 규칙: {}", next_id, merges.len());

        let special_tokens = SpecialTokens {
            pad: Some(0),
            unk: Some(1),
            bos: Some(2),
            eos: Some(3),
            mask: None,
        };

        Self {
            token_to_id,
            id_to_token,
            merges,
            special_tokens,
        }
    }

    /// 미리 정의된 어휘와 병합 규칙으로 생성
    pub fn from_vocab_and_merges(
        vocab: HashMap<String, usize>,
        merges: Vec<(String, String)>,
    ) -> Self {
        let id_to_token: HashMap<usize, String> = vocab
            .iter()
            .map(|(k, &v)| (v, k.clone()))
            .collect();

        Self {
            token_to_id: vocab,
            id_to_token,
            merges,
            special_tokens: SpecialTokens::default(),
        }
    }

    /// JSON으로 저장
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)
    }

    /// JSON에서 로드
    pub fn load(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let mut tokenizer: Self = serde_json::from_str(&json)?;
        tokenizer.special_tokens = SpecialTokens {
            pad: tokenizer.token_to_id.get("<pad>").copied(),
            unk: tokenizer.token_to_id.get("<unk>").copied(),
            bos: tokenizer.token_to_id.get("<bos>").copied(),
            eos: tokenizer.token_to_id.get("<eos>").copied(),
            mask: tokenizer.token_to_id.get("<mask>").copied(),
        };
        Ok(tokenizer)
    }

    /// 텍스트를 토큰 문자열로 변환 (디버깅용)
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let mut result = Vec::new();

        for word in text.split_whitespace() {
            // 문자 단위로 분리
            let mut tokens: Vec<String> = word.chars().map(|c| c.to_string()).collect();

            // 병합 규칙 순서대로 적용
            for (a, b) in &self.merges {
                let new_token = format!("{}{}", a, b);
                tokens = merge_pair(&tokens, a, b, &new_token);
            }

            result.extend(tokens);
        }

        result
    }
}

impl Tokenizer for BPETokenizer {
    fn encode(&self, text: &str) -> Vec<usize> {
        let tokens = self.tokenize(text);

        tokens
            .iter()
            .map(|token| {
                *self.token_to_id.get(token).unwrap_or(
                    &self.special_tokens.unk.unwrap_or(0)
                )
            })
            .collect()
    }

    fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .filter_map(|&id| self.id_to_token.get(&id))
            .cloned()
            .collect::<Vec<_>>()
            .join("")
    }

    fn vocab_size(&self) -> usize {
        self.token_to_id.len()
    }

    fn special_tokens(&self) -> SpecialTokens {
        self.special_tokens.clone()
    }
}

/// 토큰 시퀀스에서 특정 쌍을 병합
fn merge_pair(tokens: &[String], a: &str, b: &str, new_token: &str) -> Vec<String> {
    if tokens.len() < 2 {
        return tokens.to_vec();
    }

    let mut result = Vec::new();
    let mut i = 0;

    while i < tokens.len() {
        if i < tokens.len() - 1 && tokens[i] == a && tokens[i + 1] == b {
            result.push(new_token.to_string());
            i += 2;
        } else {
            result.push(tokens[i].clone());
            i += 1;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpe_training() {
        let text = "low lower lowest low lower lowest low low";
        let tokenizer = BPETokenizer::train(text, 20);

        assert!(tokenizer.vocab_size() <= 20);
        assert!(tokenizer.merges.len() > 0);
    }

    #[test]
    fn test_bpe_encode_decode() {
        let text = "low lower lowest low lower lowest";
        let tokenizer = BPETokenizer::train(text, 30);

        let encoded = tokenizer.encode("low");
        let decoded = tokenizer.decode(&encoded);

        assert_eq!(decoded, "low");
    }

    #[test]
    fn test_merge_pair() {
        let tokens = vec!["l".to_string(), "o".to_string(), "w".to_string()];
        let merged = merge_pair(&tokens, "l", "o", "lo");

        assert_eq!(merged, vec!["lo", "w"]);
    }

    #[test]
    fn test_tokenize() {
        let text = "aa bb aa bb aa bb";
        let tokenizer = BPETokenizer::train(text, 10);

        let tokens = tokenizer.tokenize("aa");
        // "aa"는 자주 등장하므로 하나의 토큰으로 병합되어야 함
        assert!(tokens.len() <= 2);
    }
}
