//! LLM Visualizer - Web Server
//!
//! 브라우저에서 LLM을 시각화하는 웹 서버입니다.
//! 실행 후 http://localhost:3000 에서 확인하세요.

use axum::{
    extract::State,
    http::StatusCode,
    response::{Html, IntoResponse, Json},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tower_http::cors::CorsLayer;

use llm_from_scratch::prelude::*;

// ==================== 앱 상태 ====================
struct AppState {
    model: Mutex<Option<GPT>>,
    tokenizer: CharTokenizer,
}

// ==================== API 타입 ====================
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

#[derive(Deserialize)]
struct CreateModelRequest {
    config_type: String,
}

#[derive(Deserialize)]
struct TokenizeRequest {
    text: String,
}

#[derive(Serialize)]
struct TokenizeResponse {
    tokens: Vec<usize>,
    token_strs: Vec<String>,
}

#[derive(Deserialize)]
struct ForwardRequest {
    text: String,
}

#[derive(Serialize)]
struct ForwardResponse {
    input_tokens: Vec<usize>,
    output_shape: Vec<usize>,
    predicted_next: String,
    top_predictions: Vec<TokenPrediction>,
}

#[derive(Serialize)]
struct TokenPrediction {
    token: String,
    probability: f32,
}

#[derive(Deserialize)]
struct GenerateRequest {
    prompt: String,
    max_tokens: usize,
    temperature: f32,
}

#[derive(Serialize)]
struct GenerateResponse {
    text: String,
    tokens: Vec<usize>,
}

#[derive(Deserialize)]
struct AttentionRequest {
    text: String,
}

#[derive(Serialize)]
struct AttentionResponse {
    heads: Vec<HeadWeights>,
    tokens: Vec<String>,
}

#[derive(Serialize)]
struct HeadWeights {
    head_idx: usize,
    weights: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
struct TrainingRequest {
    num_steps: usize,
}

#[derive(Serialize)]
struct TrainingStep {
    step: usize,
    loss: f32,
    perplexity: f32,
    learning_rate: f32,
}

#[derive(Deserialize)]
struct PosEncRequest {
    max_len: usize,
    d_model: usize,
}

// ==================== API 핸들러 ====================

async fn index() -> Html<&'static str> {
    Html(include_str!("../../static/index.html"))
}

async fn styles() -> impl IntoResponse {
    (
        StatusCode::OK,
        [("content-type", "text/css")],
        include_str!("../../static/styles.css"),
    )
}

async fn script() -> impl IntoResponse {
    (
        StatusCode::OK,
        [("content-type", "application/javascript")],
        include_str!("../../static/app.js"),
    )
}

async fn create_model(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateModelRequest>,
) -> Json<ModelInfo> {
    let config = match req.config_type.as_str() {
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

    Json(info)
}

async fn tokenize(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TokenizeRequest>,
) -> Json<TokenizeResponse> {
    let tokens = state.tokenizer.encode(&req.text);
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

    Json(TokenizeResponse { tokens, token_strs })
}

async fn forward(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ForwardRequest>,
) -> Result<Json<ForwardResponse>, (StatusCode, String)> {
    let model_guard = state.model.lock().unwrap();
    let model = model_guard
        .as_ref()
        .ok_or((StatusCode::BAD_REQUEST, "Model not created".to_string()))?;

    let tokens = state.tokenizer.encode(&req.text);
    if tokens.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "Empty input".to_string()));
    }

    let logits = model.forward(&[tokens.clone()]);
    let shape = logits.shape();
    let last_pos = shape[1] - 1;
    let vocab_size = shape[2];

    // Top-5 예측
    let mut indexed: Vec<(usize, f32)> = (0..vocab_size)
        .map(|v| (v, logits[[0, last_pos, v]]))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Softmax for probabilities
    let max_val = indexed[0].1;
    let exp_sum: f32 = indexed.iter().take(10).map(|(_, v)| (v - max_val).exp()).sum();

    let top_predictions: Vec<TokenPrediction> = indexed
        .iter()
        .take(5)
        .map(|(idx, val)| {
            let prob = (val - max_val).exp() / exp_sum;
            let token = if *idx < 128 {
                char::from_u32(*idx as u32).unwrap_or('?').to_string()
            } else {
                format!("[{}]", idx)
            };
            TokenPrediction {
                token,
                probability: prob,
            }
        })
        .collect();

    let predicted_next = top_predictions[0].token.clone();

    Ok(Json(ForwardResponse {
        input_tokens: tokens,
        output_shape: vec![shape[0], shape[1], shape[2]],
        predicted_next,
        top_predictions,
    }))
}

async fn generate(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, (StatusCode, String)> {
    let model_guard = state.model.lock().unwrap();
    let model = model_guard
        .as_ref()
        .ok_or((StatusCode::BAD_REQUEST, "Model not created".to_string()))?;

    let start_tokens = state.tokenizer.encode(&req.prompt);
    let generated = model.generate(&start_tokens, req.max_tokens, req.temperature);
    let text = state.tokenizer.decode(&generated);

    Ok(Json(GenerateResponse {
        text,
        tokens: generated,
    }))
}

async fn attention(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AttentionRequest>,
) -> Json<AttentionResponse> {
    let tokens = state.tokenizer.encode(&req.text);
    let seq_len = tokens.len();
    let num_heads = 4;

    let mut heads = Vec::new();
    for h in 0..num_heads {
        let mut weights = Vec::new();
        for i in 0..seq_len {
            let mut row = vec![0.0; seq_len];
            let total: f32 = (i + 1) as f32;
            for j in 0..=i {
                let base = 1.0 / total;
                let bias = match h {
                    0 => {
                        if j == i {
                            0.3
                        } else {
                            0.0
                        }
                    }
                    1 => {
                        if j == 0 {
                            0.2
                        } else {
                            0.0
                        }
                    }
                    2 => (j as f32 / total) * 0.2,
                    _ => 0.0,
                };
                row[j] = base + bias;
            }
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

    Json(AttentionResponse {
        heads,
        tokens: token_strs,
    })
}

async fn training(Json(req): Json<TrainingRequest>) -> Json<Vec<TrainingStep>> {
    let scheduler = WarmupCosineDecay::new(0.001, 10, req.num_steps);

    let history: Vec<TrainingStep> = (0..req.num_steps)
        .map(|step| {
            let lr = scheduler.get_lr(step);
            let base_loss = 5.0 * (-0.03 * step as f32).exp() + 0.5;
            let noise = ((step as f32 * 0.7).sin() * 0.1).abs();
            let loss = base_loss + noise;
            let ppl = loss.exp();

            TrainingStep {
                step,
                loss,
                perplexity: ppl,
                learning_rate: lr,
            }
        })
        .collect();

    Json(history)
}

async fn pos_encoding(Json(req): Json<PosEncRequest>) -> Json<Vec<Vec<f32>>> {
    let pos_enc = sinusoidal_positional_encoding(req.max_len, req.d_model);

    let result: Vec<Vec<f32>> = (0..req.max_len)
        .map(|i| (0..req.d_model).map(|j| pos_enc[[i, j]]).collect())
        .collect();

    Json(result)
}

#[tokio::main]
async fn main() {
    let state = Arc::new(AppState {
        model: Mutex::new(None),
        tokenizer: CharTokenizer::new(),
    });

    let app = Router::new()
        // Static files
        .route("/", get(index))
        .route("/styles.css", get(styles))
        .route("/app.js", get(script))
        // API endpoints
        .route("/api/model", post(create_model))
        .route("/api/tokenize", post(tokenize))
        .route("/api/forward", post(forward))
        .route("/api/generate", post(generate))
        .route("/api/attention", post(attention))
        .route("/api/training", post(training))
        .route("/api/pos-encoding", post(pos_encoding))
        .layer(CorsLayer::permissive())
        .with_state(state);

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║       LLM Visualizer - Educational Deep Learning          ║");
    println!("╠═══════════════════════════════════════════════════════════╣");
    println!("║  Server running at: http://localhost:3000                 ║");
    println!("║  Press Ctrl+C to stop                                     ║");
    println!("╚═══════════════════════════════════════════════════════════╝");

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
