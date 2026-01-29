//! LLM from Scratch - Demo
//!
//! 교육 목적의 Transformer 구현 데모입니다.

use llm_from_scratch::prelude::*;

fn main() {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║       LLM from Scratch - Educational Implementation       ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // 1. 모델 설정 출력
    println!("1. 모델 생성");
    println!("────────────────────────────────────────");
    let config = GPTConfig::mini();
    println!("   Config: {:?}\n", config);

    let model = GPT::new(config.clone());
    println!("{}", model.summary());

    // 2. 토크나이저 데모
    println!("\n2. 토크나이저");
    println!("────────────────────────────────────────");
    let tokenizer = CharTokenizer::new();
    let text = "Hello, world!";
    let tokens = tokenizer.encode(text);
    println!("   입력 텍스트: \"{}\"", text);
    println!("   토큰 ID: {:?}", tokens);
    println!("   디코딩: \"{}\"", tokenizer.decode(&tokens));

    // 3. 순전파 데모
    println!("\n3. 순전파 (Forward Pass)");
    println!("────────────────────────────────────────");
    let input_tokens = vec![vec![1, 2, 3, 4, 5]];
    println!("   입력: {:?}", input_tokens[0]);
    println!("   입력 형태: (batch=1, seq_len=5)");

    let logits = model.forward(&input_tokens);
    println!("   출력 형태: {:?}", logits.shape());
    println!("   → (batch=1, seq_len=5, vocab_size={})", config.vocab_size);

    // 4. 텍스트 생성 데모
    println!("\n4. 텍스트 생성");
    println!("────────────────────────────────────────");
    let start = tokenizer.encode("Hi");
    println!("   시작 토큰: {:?} (\"Hi\")", start);

    println!("\n   Temperature별 생성 결과:");
    for temp in [0.5, 1.0, 1.5] {
        let generated = model.generate(&start, 10, temp);
        let text = tokenizer.decode(&generated);
        println!("   T={:.1}: \"{}\" ({:?})", temp, text, generated);
    }

    // 5. Attention 시각화 데이터 생성
    println!("\n5. Attention 시각화 (JSON 출력)");
    println!("────────────────────────────────────────");
    println!("   웹앱에서 시각화할 수 있는 JSON 데이터를 생성합니다.");
    println!("   (실제 Attention 가중치는 별도 추출 필요)");

    // 6. 학습 루프 구조
    println!("\n6. 학습 루프 구조");
    println!("────────────────────────────────────────");
    println!(r#"
   for epoch in 0..num_epochs {{
       for (x, y) in dataloader {{
           // 1. Forward
           let logits = model.forward(&x);

           // 2. Loss 계산
           let loss = cross_entropy_loss(&logits, &y);

           // 3. Backward (역전파)
           // ... 기울기 계산 ...

           // 4. Optimizer step
           optimizer.step();
           scheduler.step();
       }}
   }}
"#);

    println!("════════════════════════════════════════════════════════════");
    println!("데모 완료! 자세한 내용은 examples/ 디렉토리를 참고하세요.");
    println!("════════════════════════════════════════════════════════════");
}
