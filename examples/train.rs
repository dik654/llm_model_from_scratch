//! 학습 예제
//!
//! 간단한 텍스트 데이터로 GPT 모델을 학습합니다.

use llm_from_scratch::prelude::*;
use llm_from_scratch::viz::TrainingViz;

fn main() {
    println!("=== GPT 학습 예제 ===\n");

    // 1. 데이터 준비
    let text = "hello world hello rust hello deep learning";
    let tokenizer = CharTokenizer::from_text(text);

    println!("어휘 크기: {}", tokenizer.vocab_size());

    // 토큰화
    let tokens = tokenizer.encode(text);
    println!("토큰: {:?}\n", tokens);

    // 2. 모델 생성
    let config = GPTConfig {
        vocab_size: tokenizer.vocab_size(),
        max_seq_len: 32,
        d_model: 32,
        num_heads: 2,
        num_layers: 2,
        d_ff: 64,
        dropout: 0.0,
    };

    let model = GPT::new(config.clone());
    println!("모델 생성 완료");
    println!("파라미터 수: {}\n", model.num_parameters());

    // 3. 학습 설정
    let lr = 0.001;
    let num_steps = 10;
    let batch_size = 1;
    let seq_len = 8;

    let mut viz = TrainingViz::new(num_steps);

    println!("학습 설정:");
    println!("  - 학습률: {}", lr);
    println!("  - 스텝 수: {}", num_steps);
    println!("  - 배치 크기: {}", batch_size);
    println!("  - 시퀀스 길이: {}\n", seq_len);

    // 4. 학습 루프 (시뮬레이션)
    println!("학습 시작...\n");

    for step in 0..num_steps {
        // 배치 생성 (간단하게 처음부터 seq_len만큼)
        let start_idx = step % (tokens.len().saturating_sub(seq_len));
        let input: Vec<usize> = tokens[start_idx..start_idx + seq_len].to_vec();
        let target: Vec<usize> = tokens[start_idx + 1..start_idx + seq_len + 1]
            .iter()
            .cloned()
            .chain(std::iter::once(0))
            .take(seq_len)
            .collect();

        // Forward
        let logits = model.forward(&[input.clone()]);

        // Loss 계산
        let loss = cross_entropy_loss(&logits, &[target]);
        let ppl = perplexity(loss);

        // 시각화 업데이트
        viz.update(step + 1, loss, lr, seq_len);

        println!(
            "Step {:>3}/{}: Loss = {:.4}, PPL = {:.2}",
            step + 1,
            num_steps,
            loss,
            ppl
        );
    }

    println!("\n학습 완료!");
    println!("{}", viz.summary());

    // 5. 생성 테스트
    println!("\n=== 생성 테스트 ===");
    let start = tokenizer.encode("hello");
    println!("시작: \"hello\" → {:?}", start);

    let generated = model.generate(&start, 10, 1.0);
    let generated_text = tokenizer.decode(&generated);
    println!("생성: \"{}\" ← {:?}", generated_text, generated);
}
