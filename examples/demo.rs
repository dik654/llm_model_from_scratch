//! 기본 데모
//!
//! GPT 모델의 기본 사용법을 보여줍니다.

use llm_from_scratch::prelude::*;

fn main() {
    println!("=== GPT 모델 기본 데모 ===\n");

    // 1. 모델 생성
    let config = GPTConfig::mini();
    let model = GPT::new(config);
    println!("모델 요약:\n{}", model.summary());

    // 2. 순전파
    let tokens = vec![vec![10, 20, 30, 40, 50]];
    let logits = model.forward(&tokens);
    println!("입력: {:?}", tokens[0]);
    println!("출력 형태: {:?}\n", logits.shape());

    // 3. 생성
    let start = vec![1, 2, 3];
    println!("시작 토큰: {:?}", start);

    let generated = model.generate(&start, 20, 1.0);
    println!("생성된 토큰: {:?}", generated);
}
