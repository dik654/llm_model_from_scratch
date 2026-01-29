//! 터미널 기반 시각화
//!
//! ratatui를 사용하여 터미널에서 시각화합니다.

use crate::tensor::{Tensor2D, Tensor3D, Tensor4D};
use std::io::{self, Write};

/// 터미널 시각화 유틸리티
pub struct TerminalViz;

impl TerminalViz {
    /// 텐서 형태를 시각적으로 표시
    pub fn show_tensor_shape(name: &str, shape: &[usize]) -> String {
        let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        let shape_str = dims.join(" × ");

        let visual = match shape.len() {
            1 => format!("[{}]", "■".repeat(shape[0].min(20))),
            2 => {
                let rows = shape[0].min(5);
                let cols = shape[1].min(10);
                (0..rows)
                    .map(|_| format!("│{}│", "■".repeat(cols)))
                    .collect::<Vec<_>>()
                    .join("\n")
            }
            3 => format!(
                "┌─────────────────┐\n│ Batch: {} │\n│ Seq: {} │\n│ Dim: {} │\n└─────────────────┘",
                shape[0], shape[1], shape[2]
            ),
            4 => format!(
                "┌─────────────────┐\n│ Batch: {} │\n│ Heads: {} │\n│ Seq: {} │\n│ Dim: {} │\n└─────────────────┘",
                shape[0], shape[1], shape[2], shape[3]
            ),
            _ => "...".to_string(),
        };

        format!(
            "╔══════════════════════════════════════╗\n\
             ║ {:<36} ║\n\
             ║ Shape: {:<29} ║\n\
             ╠══════════════════════════════════════╣\n\
             {}\n\
             ╚══════════════════════════════════════╝",
            name, shape_str, visual
        )
    }

    /// Transformer 데이터 흐름 시각화
    pub fn show_transformer_flow(batch: usize, seq_len: usize, d_model: usize, vocab_size: usize) -> String {
        format!(
            r#"
╔═══════════════════════════════════════════════════════════════╗
║                    TRANSFORMER 데이터 흐름                     ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║   입력 토큰 ID        ({}, {})                              ║
║         │                                                     ║
║         ▼                                                     ║
║   ┌─────────────┐                                             ║
║   │  Embedding  │    ({}, {}, {})                         ║
║   └─────────────┘                                             ║
║         │                                                     ║
║         + ◄──── Position Encoding                             ║
║         │                                                     ║
║         ▼                                                     ║
║   ┌─────────────────────────────────────┐                     ║
║   │         Transformer Block           │                     ║
║   │  ┌─────────────┐  ┌─────────────┐   │                     ║
║   │  │  Attention  │  │     FFN     │   │                     ║
║   │  └─────────────┘  └─────────────┘   │                     ║
║   └─────────────────────────────────────┘                     ║
║         │                                                     ║
║         ▼                                                     ║
║   ┌─────────────┐                                             ║
║   │  Layer Norm │    ({}, {}, {})                         ║
║   └─────────────┘                                             ║
║         │                                                     ║
║         ▼                                                     ║
║   ┌─────────────┐                                             ║
║   │   LM Head   │    ({}, {}, {})                        ║
║   └─────────────┘                                             ║
║         │                                                     ║
║         ▼                                                     ║
║   출력 Logits        ({}, {}, {})                        ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"#,
            batch, seq_len,
            batch, seq_len, d_model,
            batch, seq_len, d_model,
            batch, seq_len, vocab_size,
            batch, seq_len, vocab_size
        )
    }

    /// Attention 연산 시각화
    pub fn show_attention_computation(seq_len: usize, d_model: usize, num_heads: usize) -> String {
        let head_dim = d_model / num_heads;

        format!(
            r#"
╔═══════════════════════════════════════════════════════════════╗
║                    MULTI-HEAD ATTENTION                        ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║   입력 X: (batch, {}, {})                                  ║
║      │                                                        ║
║      ├──────────────┬──────────────┐                          ║
║      ▼              ▼              ▼                          ║
║   ┌─────┐       ┌─────┐       ┌─────┐                         ║
║   │ Wq  │       │ Wk  │       │ Wv  │    Linear Projections   ║
║   └─────┘       └─────┘       └─────┘                         ║
║      │              │              │                          ║
║      ▼              ▼              ▼                          ║
║   Q: ({},{})   K: ({},{})   V: ({},{})                  ║
║      │              │              │                          ║
║      ├──────────────┴──────────────┤                          ║
║      │      Split into {} heads     │                          ║
║      ▼                                                        ║
║   ┌───────────────────────────────────┐                       ║
║   │  Q·K^T / √{:<3}  →  Softmax  →  ·V  │   Scaled Dot-Product ║
║   └───────────────────────────────────┘                       ║
║      │                                                        ║
║      ▼                                                        ║
║   Concat all heads: (batch, {}, {})                        ║
║      │                                                        ║
║      ▼                                                        ║
║   ┌─────┐                                                     ║
║   │ Wo  │    Output Projection                                ║
║   └─────┘                                                     ║
║      │                                                        ║
║      ▼                                                        ║
║   출력: (batch, {}, {})                                    ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"#,
            seq_len, d_model,
            seq_len, d_model, seq_len, d_model, seq_len, d_model,
            num_heads,
            head_dim,
            seq_len, d_model,
            seq_len, d_model
        )
    }

    /// 히트맵 형식으로 2D 텐서 출력
    pub fn heatmap(tensor: &Tensor2D, title: &str) -> String {
        let rows = tensor.shape()[0];
        let cols = tensor.shape()[1];

        // 값 범위 계산
        let min_val = tensor.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = tensor.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;

        let blocks = ['░', '▒', '▓', '█'];

        let mut result = format!("┌─ {} ─┐\n", title);

        let display_rows = rows.min(15);
        let display_cols = cols.min(40);

        for i in 0..display_rows {
            result.push('│');
            for j in 0..display_cols {
                let val = tensor[[i * rows / display_rows, j * cols / display_cols]];
                let normalized = if range > 0.0 {
                    ((val - min_val) / range).min(1.0).max(0.0)
                } else {
                    0.5
                };
                let idx = (normalized * 3.0) as usize;
                result.push(blocks[idx.min(3)]);
            }
            if cols > display_cols {
                result.push_str("...");
            }
            result.push_str("│\n");
        }

        if rows > display_rows {
            result.push_str("│ ... │\n");
        }

        result.push_str(&format!("└{}┘\n", "─".repeat(display_cols.min(40) + 2)));
        result.push_str(&format!("  Range: [{:.3}, {:.3}]\n", min_val, max_val));

        result
    }

    /// 진행 바
    pub fn progress_bar(current: usize, total: usize, width: usize) -> String {
        let progress = current as f64 / total as f64;
        let filled = (progress * width as f64) as usize;
        let empty = width - filled;

        format!(
            "[{}{}] {}/{} ({:.1}%)",
            "█".repeat(filled),
            "░".repeat(empty),
            current,
            total,
            progress * 100.0
        )
    }

    /// 손실 그래프 (ASCII)
    pub fn loss_graph(losses: &[f32], width: usize, height: usize) -> String {
        if losses.is_empty() {
            return "No data".to_string();
        }

        let min_loss = losses.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_loss = losses.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max_loss - min_loss;

        let mut result = String::new();
        result.push_str(&format!("{:.4} │", max_loss));

        for row in 0..height {
            let threshold = max_loss - range * (row as f32 / height as f32);

            if row > 0 {
                result.push_str(&format!("\n       │"));
            }

            for col in 0..width {
                let idx = col * losses.len() / width;
                let loss = losses[idx];

                if loss >= threshold && (row == 0 || losses[idx] < max_loss - range * ((row - 1) as f32 / height as f32)) {
                    result.push('●');
                } else if loss >= threshold {
                    result.push('│');
                } else {
                    result.push(' ');
                }
            }
        }

        result.push_str(&format!("\n{:.4} │", min_loss));
        result.push_str(&format!("\n       └{}─", "─".repeat(width)));
        result.push_str(&format!("\n        0{}{}steps", " ".repeat(width / 2 - 3), losses.len()));

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::randn_2d;

    #[test]
    fn test_tensor_shape_visualization() {
        let viz = TerminalViz::show_tensor_shape("Input", &[32, 10, 768]);
        assert!(viz.contains("32 × 10 × 768"));
    }

    #[test]
    fn test_heatmap() {
        let tensor = randn_2d(5, 10);
        let viz = TerminalViz::heatmap(&tensor, "Test");
        assert!(viz.contains("Test"));
    }

    #[test]
    fn test_progress_bar() {
        let bar = TerminalViz::progress_bar(50, 100, 20);
        assert!(bar.contains("50/100"));
        assert!(bar.contains("50.0%"));
    }
}
