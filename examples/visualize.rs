//! 시각화 데모
//!
//! Attention 패턴과 학습 통계를 JSON으로 출력합니다.
//! 이 데이터를 웹앱에서 시각화할 수 있습니다.

use llm_from_scratch::prelude::*;
use llm_from_scratch::viz::{AttentionViz, AttentionData, HeadAttention, TrainingViz};
use ndarray::Array4;

fn main() {
    println!("=== 시각화 데모 ===\n");

    // 1. Attention 시각화 데이터 생성
    println!("1. Attention 시각화");
    println!("────────────────────────────────────────");

    // 샘플 Attention 가중치 생성
    let tokens = vec![
        "The".to_string(),
        "cat".to_string(),
        "sat".to_string(),
        "on".to_string(),
        "mat".to_string(),
    ];

    // 가상의 Attention 가중치 (batch=1, heads=2, seq=5, seq=5)
    let mut attention = Array4::<f32>::zeros((1, 2, 5, 5));

    // Causal mask 적용 + 가상 패턴
    for h in 0..2 {
        for i in 0..5 {
            for j in 0..=i {
                // 자기 자신에게 높은 attention
                if i == j {
                    attention[[0, h, i, j]] = 0.5;
                } else {
                    attention[[0, h, i, j]] = 0.5 / (i as f32);
                }
            }
        }
    }

    // 정규화 (행 합 = 1)
    for h in 0..2 {
        for i in 0..5 {
            let sum: f32 = (0..5).map(|j| attention[[0, h, i, j]]).sum();
            if sum > 0.0 {
                for j in 0..5 {
                    attention[[0, h, i, j]] /= sum;
                }
            }
        }
    }

    let attention_data = AttentionViz::from_tensor(&attention, &tokens, 0);

    println!("토큰: {:?}", tokens);
    println!("\nHead 0 Attention 패턴:");
    print_attention_matrix(&attention_data.heads[0], &tokens);

    println!("\nHead 1 Attention 패턴:");
    print_attention_matrix(&attention_data.heads[1], &tokens);

    // JSON 출력
    println!("\nJSON 출력 (웹앱용):");
    println!("────────────────────────────────────────");
    let json = AttentionViz::to_json(&attention_data);
    println!("{}", json);

    // 2. 학습 통계 시각화
    println!("\n2. 학습 통계 시각화");
    println!("────────────────────────────────────────");

    let mut viz = TrainingViz::new(100);

    // 가상의 학습 데이터
    for step in 0..20 {
        let loss = 5.0 * (-0.1 * step as f32).exp() + 0.5;
        let lr = 0.001 * (1.0 - step as f32 / 100.0);
        viz.update(step + 1, loss, lr, 1024);
    }

    println!("{}", viz.summary());

    println!("\n학습 통계 JSON:");
    println!("────────────────────────────────────────");
    println!("{}", viz.to_json());

    // 3. 학습률 스케줄 시각화
    println!("\n3. 학습률 스케줄");
    println!("────────────────────────────────────────");

    let scheduler = WarmupCosineDecay::new(0.001, 10, 100);

    println!("Warmup + Cosine Decay 스케줄:");
    println!("Step |    LR    | 시각화");
    println!("-----|----------|{}", "-".repeat(30));

    for step in (0..100).step_by(10) {
        let lr = scheduler.get_lr(step);
        let bar_len = (lr / 0.001 * 25.0) as usize;
        println!(
            "{:>4} | {:.6} | {}",
            step,
            lr,
            "█".repeat(bar_len)
        );
    }
}

/// Attention 행렬을 텍스트로 출력
fn print_attention_matrix(head: &HeadAttention, tokens: &[String]) {
    let width = 8;

    // 헤더
    print!("{:>8}", "");
    for token in tokens {
        print!("{:>width$}", token, width = width);
    }
    println!();

    // 행
    for (i, row) in head.weights.iter().enumerate() {
        if i < tokens.len() {
            print!("{:>8}", tokens[i]);
            for &val in row {
                let block = if val > 0.4 {
                    "████"
                } else if val > 0.2 {
                    "▓▓▓░"
                } else if val > 0.1 {
                    "▒▒░░"
                } else if val > 0.0 {
                    "░░░░"
                } else {
                    "    "
                };
                print!("{:>width$}", block, width = width);
            }
            println!();
        }
    }
}
