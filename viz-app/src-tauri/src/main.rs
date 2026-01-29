#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use tauri::State;

use llm_from_scratch::prelude::*;
use llm_from_scratch::tensor::{Tensor2D, Tensor3D};

// 앱 상태 관리
struct AppState {
    model: Mutex<Option<GPT>>,
    tokenizer: Mutex<CharTokenizer>,
    training_history: Mutex<Vec<TrainingStep>>,
}

#[derive(Serialize, Clone)]
struct TrainingStep {
    step: usize,
    loss: f32,
    perplexity: f32,
    learning_rate: f32,
}

#[derive(Serialize)]
struct ModelInfo {
    vocab_size: usize,
    max_seq_len: usize,
    d_model: usize,
    num_heads: usize,
    num_layers: usize,
    d_ff: usize,
    total_params: usize,
    memory_mb: f32,
}

#[derive(Serialize)]
struct AttentionWeights {
    heads: Vec<HeadWeights>,
    tokens: Vec<String>,
}

#[derive(Serialize)]
struct HeadWeights {
    head_idx: usize,
    weights: Vec<Vec<f32>>,
}

#[derive(Serialize)]
struct GenerationResult {
    tokens: Vec<usize>,
    text: String,
    steps: Vec<GenerationStep>,
}

#[derive(Serialize)]
struct GenerationStep {
    step: usize,
    token_id: usize,
    token_text: String,
    top_probs: Vec<TokenProb>,
}

#[derive(Serialize)]
struct TokenProb {
    token_id: usize,
    token_text: String,
    probability: f32,
}

#[derive(Serialize)]
struct ForwardPassResult {
    input_tokens: Vec<usize>,
    input_text: String,
    output_shape: Vec<usize>,
    embeddings_sample: Vec<f32>,
    logits_sample: Vec<f32>,
    predicted_next: String,
}

// ==================== Tauri 커맨드 ====================

/// 모델 생성
#[tauri::command]
fn create_model(config_type: &str, state: State<AppState>) -> Result<ModelInfo, String> {
    let config = match config_type {
        "mini" => GPTConfig::mini(),
        "small" => GPTConfig::small(),
        "medium" => GPTConfig::medium(),
        _ => GPTConfig::mini(),
    };

    let model = GPT::new(config.clone());
    let total_params = model.num_parameters();

    let info = ModelInfo {
        vocab_size: config.vocab_size,
        max_seq_len: config.max_seq_len,
        d_model: config.d_model,
        num_heads: config.num_heads,
        num_layers: config.num_layers,
        d_ff: config.d_ff,
        total_params,
        memory_mb: (total_params * 4) as f32 / (1024.0 * 1024.0),
    };

    *state.model.lock().unwrap() = Some(model);

    Ok(info)
}

/// 텍스트 토큰화
#[tauri::command]
fn tokenize(text: &str, state: State<AppState>) -> Result<Vec<usize>, String> {
    let tokenizer = state.tokenizer.lock().unwrap();
    Ok(tokenizer.encode(text))
}

/// 토큰 디코딩
#[tauri::command]
fn decode_tokens(tokens: Vec<usize>, state: State<AppState>) -> Result<String, String> {
    let tokenizer = state.tokenizer.lock().unwrap();
    Ok(tokenizer.decode(&tokens))
}

/// 순전파 실행
#[tauri::command]
fn forward_pass(text: &str, state: State<AppState>) -> Result<ForwardPassResult, String> {
    let model_guard = state.model.lock().unwrap();
    let model = model_guard.as_ref().ok_or("Model not created")?;

    let tokenizer = state.tokenizer.lock().unwrap();
    let tokens = tokenizer.encode(text);

    if tokens.is_empty() {
        return Err("Empty input".to_string());
    }

    let logits = model.forward(&[tokens.clone()]);
    let shape = logits.shape();

    // 마지막 위치의 logits에서 가장 높은 확률의 토큰 찾기
    let last_pos = shape[1] - 1;
    let vocab_size = shape[2];

    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;
    for v in 0..vocab_size {
        let val = logits[[0, last_pos, v]];
        if val > max_val {
            max_val = val;
            max_idx = v;
        }
    }

    let predicted_char = if max_idx < 128 {
        char::from_u32(max_idx as u32).unwrap_or('?').to_string()
    } else {
        format!("[{}]", max_idx)
    };

    // 샘플 데이터 추출
    let logits_sample: Vec<f32> = (0..vocab_size.min(20))
        .map(|v| logits[[0, last_pos, v]])
        .collect();

    Ok(ForwardPassResult {
        input_tokens: tokens.clone(),
        input_text: text.to_string(),
        output_shape: vec![shape[0], shape[1], shape[2]],
        embeddings_sample: vec![], // 임베딩은 별도 추출 필요
        logits_sample,
        predicted_next: predicted_char,
    })
}

/// 텍스트 생성
#[tauri::command]
fn generate_text(
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    state: State<AppState>,
) -> Result<GenerationResult, String> {
    let model_guard = state.model.lock().unwrap();
    let model = model_guard.as_ref().ok_or("Model not created")?;

    let tokenizer = state.tokenizer.lock().unwrap();
    let start_tokens = tokenizer.encode(prompt);

    let mut tokens = start_tokens.clone();
    let mut steps = Vec::new();

    for step in 0..max_tokens {
        let context_len = tokens.len().min(128); // max_seq_len
        let context = &tokens[tokens.len() - context_len..];

        let logits = model.forward(&[context.to_vec()]);
        let seq_len = logits.shape()[1];
        let vocab_size = logits.shape()[2];

        // 마지막 위치의 logits
        let last_logits: Vec<f32> = (0..vocab_size)
            .map(|v| logits[[0, seq_len - 1, v]] / temperature)
            .collect();

        // Softmax
        let max_val = last_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = last_logits.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();
        let probs: Vec<f32> = exp_vals.iter().map(|&x| x / sum).collect();

        // Top-5 토큰
        let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let top_probs: Vec<TokenProb> = indexed
            .iter()
            .take(5)
            .map(|(idx, prob)| {
                let token_text = if *idx < 128 {
                    char::from_u32(*idx as u32).unwrap_or('?').to_string()
                } else {
                    format!("[{}]", idx)
                };
                TokenProb {
                    token_id: *idx,
                    token_text,
                    probability: *prob,
                }
            })
            .collect();

        // 샘플링 (간단한 구현)
        let mut rng_val: f32 = (step as f32 * 0.1).sin().abs();
        let mut cumsum = 0.0;
        let mut next_token = 0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if rng_val < cumsum {
                next_token = i;
                break;
            }
        }

        let token_text = if next_token < 128 {
            char::from_u32(next_token as u32).unwrap_or('?').to_string()
        } else {
            format!("[{}]", next_token)
        };

        steps.push(GenerationStep {
            step,
            token_id: next_token,
            token_text: token_text.clone(),
            top_probs,
        });

        tokens.push(next_token);
    }

    let final_text = tokenizer.decode(&tokens);

    Ok(GenerationResult {
        tokens,
        text: final_text,
        steps,
    })
}

/// Attention 가중치 추출 (시뮬레이션)
#[tauri::command]
fn get_attention_weights(text: &str, state: State<AppState>) -> Result<AttentionWeights, String> {
    let tokenizer = state.tokenizer.lock().unwrap();
    let tokens = tokenizer.encode(text);
    let seq_len = tokens.len();

    // 실제 모델에서 추출하려면 forward 수정 필요
    // 여기서는 causal attention 패턴 시뮬레이션
    let num_heads = 4;
    let mut heads = Vec::new();

    for h in 0..num_heads {
        let mut weights = Vec::new();
        for i in 0..seq_len {
            let mut row = vec![0.0; seq_len];
            // Causal mask + 헤드별 다른 패턴
            let total: f32 = (i + 1) as f32;
            for j in 0..=i {
                // 헤드별로 다른 attention 패턴
                let base = 1.0 / total;
                let bias = match h {
                    0 => if j == i { 0.3 } else { 0.0 },  // 현재 위치 집중
                    1 => if j == 0 { 0.2 } else { 0.0 },  // 첫 토큰 집중
                    2 => (j as f32 / total) * 0.2,        // 위치 비례
                    _ => 0.0,
                };
                row[j] = base + bias;
            }
            // 정규화
            let sum: f32 = row.iter().sum();
            if sum > 0.0 {
                for v in &mut row {
                    *v /= sum;
                }
            }
            weights.push(row);
        }
        heads.push(HeadWeights {
            head_idx: h,
            weights,
        });
    }

    let token_strs: Vec<String> = tokens
        .iter()
        .map(|&t| {
            if t < 128 {
                char::from_u32(t as u32).unwrap_or('?').to_string()
            } else {
                format!("[{}]", t)
            }
        })
        .collect();

    Ok(AttentionWeights {
        heads,
        tokens: token_strs,
    })
}

/// 학습 시뮬레이션 (데모용)
#[tauri::command]
fn simulate_training(num_steps: usize, state: State<AppState>) -> Result<Vec<TrainingStep>, String> {
    let mut history = state.training_history.lock().unwrap();
    history.clear();

    let scheduler = WarmupCosineDecay::new(0.001, 10, num_steps);

    for step in 0..num_steps {
        let lr = scheduler.get_lr(step);

        // 손실 시뮬레이션 (감소하는 패턴)
        let base_loss = 5.0 * (-0.03 * step as f32).exp() + 0.5;
        let noise = ((step as f32 * 0.7).sin() * 0.1).abs();
        let loss = base_loss + noise;

        let ppl = loss.exp();

        history.push(TrainingStep {
            step,
            loss,
            perplexity: ppl,
            learning_rate: lr,
        });
    }

    Ok(history.clone())
}

/// Positional Encoding 데이터
#[tauri::command]
fn get_positional_encoding(max_len: usize, d_model: usize) -> Result<Vec<Vec<f32>>, String> {
    let pos_enc = sinusoidal_positional_encoding(max_len, d_model);

    let mut result = Vec::new();
    for i in 0..max_len {
        let row: Vec<f32> = (0..d_model).map(|j| pos_enc[[i, j]]).collect();
        result.push(row);
    }

    Ok(result)
}

fn main() {
    tauri::Builder::default()
        .manage(AppState {
            model: Mutex::new(None),
            tokenizer: Mutex::new(CharTokenizer::new()),
            training_history: Mutex::new(Vec::new()),
        })
        .invoke_handler(tauri::generate_handler![
            create_model,
            tokenize,
            decode_tokens,
            forward_pass,
            generate_text,
            get_attention_weights,
            simulate_training,
            get_positional_encoding,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
