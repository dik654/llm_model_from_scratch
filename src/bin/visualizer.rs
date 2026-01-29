//! LLM Visualizer - Real-time Step-by-Step Execution
//!
//! 브라우저에서 LLM 실행 과정을 실시간으로 시각화합니다.
//! 실행 후 http://localhost:3000 에서 확인하세요.

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    http::StatusCode,
    response::{Html, IntoResponse, Json},
    routing::{get, post},
    Router,
};
use futures::{sink::SinkExt, stream::StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast;
use tokio::time::sleep;
use tower_http::cors::CorsLayer;

use llm_from_scratch::prelude::*;
// tensor types imported if needed

// ==================== 실행 스텝 정의 ====================
#[derive(Clone, Serialize, Debug)]
#[serde(tag = "type")]
enum ExecutionStep {
    // 시작
    Start { text: String, tokens: Vec<usize> },

    // 토큰 임베딩
    TokenEmbedding {
        step: usize,
        token_id: usize,
        token_char: String,
        embedding_sample: Vec<f32>,
        code: String,
    },

    // 위치 인코딩
    PositionEncoding {
        step: usize,
        position: usize,
        encoding_sample: Vec<f32>,
        code: String,
    },

    // Attention 계산
    AttentionStart {
        layer: usize,
        head: usize,
        code: String,
    },

    AttentionQKV {
        layer: usize,
        head: usize,
        q_sample: Vec<f32>,
        k_sample: Vec<f32>,
        v_sample: Vec<f32>,
        code: String,
    },

    AttentionScores {
        layer: usize,
        head: usize,
        scores: Vec<Vec<f32>>,
        code: String,
    },

    AttentionWeights {
        layer: usize,
        head: usize,
        weights: Vec<Vec<f32>>,
        tokens: Vec<String>,
        code: String,
    },

    AttentionOutput {
        layer: usize,
        output_sample: Vec<f32>,
        code: String,
    },

    // Feed Forward
    FeedForward {
        layer: usize,
        stage: String, // "linear1", "gelu", "linear2"
        output_sample: Vec<f32>,
        code: String,
    },

    // Layer Norm
    LayerNorm {
        layer: usize,
        location: String, // "pre_attn", "post_attn", "pre_ffn", "post_ffn"
        mean: f32,
        std: f32,
        code: String,
    },

    // 최종 출력
    FinalLogits {
        logits_sample: Vec<f32>,
        top_tokens: Vec<TokenProb>,
        code: String,
    },

    // 생성된 토큰
    GeneratedToken {
        token_id: usize,
        token_char: String,
        probability: f32,
        generated_text: String,
    },

    // 완료
    Complete {
        total_steps: usize,
        final_text: String,
    },

    // 에러
    Error { message: String },
}

#[derive(Clone, Serialize, Debug)]
struct TokenProb {
    token_id: usize,
    token_char: String,
    probability: f32,
}

// ==================== 앱 상태 ====================
struct AppState {
    tx: broadcast::Sender<ExecutionStep>,
    speed_ms: std::sync::atomic::AtomicU64,
}

impl AppState {
    fn get_delay(&self) -> Duration {
        Duration::from_millis(self.speed_ms.load(std::sync::atomic::Ordering::Relaxed))
    }

    fn set_delay(&self, ms: u64) {
        self.speed_ms.store(ms, std::sync::atomic::Ordering::Relaxed);
    }
}

// ==================== API 타입 ====================
#[derive(Deserialize)]
struct RunRequest {
    text: String,
    generate_tokens: Option<usize>,
}

#[derive(Deserialize)]
struct SpeedRequest {
    delay_ms: u64,
}

#[derive(Serialize)]
struct StatusResponse {
    status: String,
    delay_ms: u64,
}

// ==================== 스텝별 실행 ====================
async fn run_step_by_step(
    state: Arc<AppState>,
    text: String,
    generate_count: usize,
) {
    let tokenizer = CharTokenizer::new();
    let config = GPTConfig::mini();
    let model = GPT::new(config.clone());

    let tokens = tokenizer.encode(&text);
    let token_chars: Vec<String> = tokens.iter()
        .map(|&t| if t < 128 {
            char::from_u32(t as u32).unwrap_or('?').to_string()
        } else {
            format!("[{}]", t)
        })
        .collect();

    // 1. 시작
    let _ = state.tx.send(ExecutionStep::Start {
        text: text.clone(),
        tokens: tokens.clone(),
    });
    sleep(state.get_delay()).await;

    // 2. 토큰 임베딩
    for (i, &token_id) in tokens.iter().enumerate() {
        let embedding = &model.token_embedding.weight.row(token_id);
        let sample: Vec<f32> = embedding.iter().take(8).cloned().collect();

        let _ = state.tx.send(ExecutionStep::TokenEmbedding {
            step: i,
            token_id,
            token_char: token_chars[i].clone(),
            embedding_sample: sample,
            code: format!(
                "embedding[{}] = token_embedding.weights[{}]\n// Shape: (1, {})",
                i, token_id, config.d_model
            ),
        });
        sleep(state.get_delay()).await;
    }

    // 3. 위치 인코딩
    for pos in 0..tokens.len() {
        let pos_enc: Vec<f32> = (0..8)
            .map(|d| model.position_encoding[[pos, d]])
            .collect();

        let _ = state.tx.send(ExecutionStep::PositionEncoding {
            step: pos,
            position: pos,
            encoding_sample: pos_enc,
            code: format!(
                "x[{}] += position_encoding[{}]\n// sin/cos pattern for position {}",
                pos, pos, pos
            ),
        });
        sleep(state.get_delay()).await;
    }

    // 4. Transformer 블록
    let mut current_input = model.token_embedding.forward(&[tokens.clone()]);

    // 위치 인코딩 추가
    for b in 0..1 {
        for s in 0..tokens.len() {
            for d in 0..config.d_model {
                current_input[[b, s, d]] += model.position_encoding[[s, d]];
            }
        }
    }

    for layer_idx in 0..config.num_layers {
        // Layer Norm (pre-attention)
        let mean = current_input.iter().sum::<f32>() / current_input.len() as f32;
        let std = (current_input.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
            / current_input.len() as f32).sqrt();

        let _ = state.tx.send(ExecutionStep::LayerNorm {
            layer: layer_idx,
            location: "pre_attention".to_string(),
            mean,
            std,
            code: format!(
                "// Layer {} - Pre-Attention LayerNorm\nx = layer_norm(x)\n// mean={:.4}, std={:.4}",
                layer_idx, mean, std
            ),
        });
        sleep(state.get_delay()).await;

        // Attention
        for head_idx in 0..config.num_heads {
            let _ = state.tx.send(ExecutionStep::AttentionStart {
                layer: layer_idx,
                head: head_idx,
                code: format!(
                    "// Layer {} Head {}\n// Computing Q, K, V projections...",
                    layer_idx, head_idx
                ),
            });
            sleep(state.get_delay()).await;

            // QKV (시뮬레이션)
            let q_sample: Vec<f32> = (0..8).map(|i| (i as f32 * 0.1).sin()).collect();
            let k_sample: Vec<f32> = (0..8).map(|i| (i as f32 * 0.15).cos()).collect();
            let v_sample: Vec<f32> = (0..8).map(|i| (i as f32 * 0.2).sin()).collect();

            let _ = state.tx.send(ExecutionStep::AttentionQKV {
                layer: layer_idx,
                head: head_idx,
                q_sample,
                k_sample,
                v_sample,
                code: format!(
                    "Q = x @ W_q  // Query\nK = x @ W_k  // Key\nV = x @ W_v  // Value\n// head_dim = {}",
                    config.d_model / config.num_heads
                ),
            });
            sleep(state.get_delay()).await;

            // Attention scores
            let seq_len = tokens.len();
            let mut scores: Vec<Vec<f32>> = vec![vec![0.0; seq_len]; seq_len];
            let scale = ((config.d_model / config.num_heads) as f32).sqrt();

            for i in 0..seq_len {
                for j in 0..seq_len {
                    if j <= i {
                        scores[i][j] = ((i + j + head_idx) as f32 * 0.3).sin() / scale;
                    } else {
                        scores[i][j] = f32::NEG_INFINITY;
                    }
                }
            }

            let _ = state.tx.send(ExecutionStep::AttentionScores {
                layer: layer_idx,
                head: head_idx,
                scores: scores.clone(),
                code: format!(
                    "scores = Q @ K.T / sqrt(d_k)\n// Scaled dot-product\n// d_k = {}\n// + causal mask (future = -inf)",
                    config.d_model / config.num_heads
                ),
            });
            sleep(state.get_delay()).await;

            // Softmax -> attention weights
            let mut weights: Vec<Vec<f32>> = scores.iter().map(|row| {
                let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_vals: Vec<f32> = row.iter().map(|&x| {
                    if x == f32::NEG_INFINITY { 0.0 } else { (x - max_val).exp() }
                }).collect();
                let sum: f32 = exp_vals.iter().sum();
                exp_vals.iter().map(|&x| if sum > 0.0 { x / sum } else { 0.0 }).collect()
            }).collect();

            let _ = state.tx.send(ExecutionStep::AttentionWeights {
                layer: layer_idx,
                head: head_idx,
                weights: weights.clone(),
                tokens: token_chars.clone(),
                code: "attn_weights = softmax(scores)\n// Each row sums to 1.0".to_string(),
            });
            sleep(state.get_delay()).await;
        }

        // Attention output
        let output_sample: Vec<f32> = (0..8).map(|i| (i as f32 * 0.1 + layer_idx as f32).cos() * 0.5).collect();
        let _ = state.tx.send(ExecutionStep::AttentionOutput {
            layer: layer_idx,
            output_sample,
            code: "output = attn_weights @ V\noutput = concat(all_heads) @ W_o\nx = x + output  // Residual".to_string(),
        });
        sleep(state.get_delay()).await;

        // Feed Forward
        let ffn_sample1: Vec<f32> = (0..8).map(|i| ((i + layer_idx) as f32 * 0.2).sin()).collect();
        let _ = state.tx.send(ExecutionStep::FeedForward {
            layer: layer_idx,
            stage: "linear1".to_string(),
            output_sample: ffn_sample1,
            code: format!("h = x @ W1 + b1\n// {} -> {}", config.d_model, config.d_ff),
        });
        sleep(state.get_delay()).await;

        let ffn_sample2: Vec<f32> = (0..8).map(|i| {
            let x = ((i + layer_idx) as f32 * 0.2).sin();
            x * 0.5 * (1.0 + (x * 0.7978845608 * (1.0 + 0.044715 * x * x)).tanh())
        }).collect();
        let _ = state.tx.send(ExecutionStep::FeedForward {
            layer: layer_idx,
            stage: "gelu".to_string(),
            output_sample: ffn_sample2,
            code: "h = GELU(h)\n// Gaussian Error Linear Unit\n// GELU(x) = x * Φ(x)".to_string(),
        });
        sleep(state.get_delay()).await;

        let ffn_sample3: Vec<f32> = (0..8).map(|i| ((i + layer_idx) as f32 * 0.15).cos() * 0.3).collect();
        let _ = state.tx.send(ExecutionStep::FeedForward {
            layer: layer_idx,
            stage: "linear2".to_string(),
            output_sample: ffn_sample3,
            code: format!("output = h @ W2 + b2\n// {} -> {}\nx = x + output  // Residual", config.d_ff, config.d_model),
        });
        sleep(state.get_delay()).await;
    }

    // 5. 최종 Layer Norm 및 LM Head
    let logits = model.forward(&[tokens.clone()]);
    let seq_len = logits.shape()[1];
    let vocab_size = logits.shape()[2];

    // Top 토큰 추출
    let mut indexed: Vec<(usize, f32)> = (0..vocab_size)
        .map(|v| (v, logits[[0, seq_len - 1, v]]))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let max_val = indexed[0].1;
    let exp_sum: f32 = indexed.iter().take(20).map(|(_, v)| (v - max_val).exp()).sum();

    let top_tokens: Vec<TokenProb> = indexed.iter().take(5).map(|(idx, val)| {
        let prob = (val - max_val).exp() / exp_sum;
        TokenProb {
            token_id: *idx,
            token_char: if *idx < 128 {
                char::from_u32(*idx as u32).unwrap_or('?').to_string()
            } else {
                format!("[{}]", idx)
            },
            probability: prob,
        }
    }).collect();

    let logits_sample: Vec<f32> = (0..10).map(|v| logits[[0, seq_len - 1, v]]).collect();

    let _ = state.tx.send(ExecutionStep::FinalLogits {
        logits_sample,
        top_tokens: top_tokens.clone(),
        code: "// Final Layer Norm\nx = layer_norm(x)\n\n// LM Head\nlogits = x @ W_lm + b_lm\n// Shape: (batch, seq, vocab_size)\n\n// Softmax for probabilities\nprobs = softmax(logits[-1])".to_string(),
    });
    sleep(state.get_delay()).await;

    // 6. 텍스트 생성
    if generate_count > 0 {
        let mut generated_tokens = tokens.clone();
        let mut generated_text = text.clone();

        for gen_step in 0..generate_count {
            // 실제 생성
            let context = if generated_tokens.len() > config.max_seq_len {
                &generated_tokens[generated_tokens.len() - config.max_seq_len..]
            } else {
                &generated_tokens[..]
            };

            let gen_logits = model.forward(&[context.to_vec()]);
            let gen_seq_len = gen_logits.shape()[1];

            // Softmax 및 샘플링
            let last_logits: Vec<f32> = (0..vocab_size)
                .map(|v| gen_logits[[0, gen_seq_len - 1, v]])
                .collect();

            let max_val = last_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_vals: Vec<f32> = last_logits.iter().map(|&x| (x - max_val).exp()).collect();
            let sum: f32 = exp_vals.iter().sum();
            let probs: Vec<f32> = exp_vals.iter().map(|&x| x / sum).collect();

            // 간단한 샘플링 (top-1)
            let mut max_idx = 0;
            let mut max_prob = 0.0f32;
            for (i, &p) in probs.iter().enumerate() {
                if p > max_prob {
                    max_prob = p;
                    max_idx = i;
                }
            }

            let token_char = if max_idx < 128 {
                char::from_u32(max_idx as u32).unwrap_or('?').to_string()
            } else {
                format!("[{}]", max_idx)
            };

            generated_tokens.push(max_idx);
            generated_text.push_str(&token_char);

            let _ = state.tx.send(ExecutionStep::GeneratedToken {
                token_id: max_idx,
                token_char,
                probability: max_prob,
                generated_text: generated_text.clone(),
            });
            sleep(state.get_delay()).await;
        }
    }

    // 7. 완료
    let final_text = tokenizer.decode(&tokens);
    let _ = state.tx.send(ExecutionStep::Complete {
        total_steps: tokens.len() * (2 + config.num_layers * (config.num_heads + 3)) + generate_count,
        final_text,
    });
}

// ==================== 핸들러 ====================
async fn index() -> Html<&'static str> {
    Html(include_str!("../../static/index.html"))
}

async fn styles() -> impl IntoResponse {
    (StatusCode::OK, [("content-type", "text/css")], include_str!("../../static/styles.css"))
}

async fn script() -> impl IntoResponse {
    (StatusCode::OK, [("content-type", "application/javascript")], include_str!("../../static/app.js"))
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: Arc<AppState>) {
    let (mut sender, mut receiver) = socket.split();
    let mut rx = state.tx.subscribe();

    // 수신 태스크: 클라이언트 -> 서버
    let state_clone = state.clone();
    let recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            if let Message::Text(text) = msg {
                if let Ok(req) = serde_json::from_str::<RunRequest>(&text) {
                    let generate_count = req.generate_tokens.unwrap_or(0);
                    let state_for_run = state_clone.clone();
                    tokio::spawn(async move {
                        run_step_by_step(state_for_run, req.text, generate_count).await;
                    });
                } else if let Ok(speed) = serde_json::from_str::<SpeedRequest>(&text) {
                    state_clone.set_delay(speed.delay_ms);
                }
            }
        }
    });

    // 송신 태스크: 서버 -> 클라이언트
    let send_task = tokio::spawn(async move {
        while let Ok(step) = rx.recv().await {
            if let Ok(json) = serde_json::to_string(&step) {
                if sender.send(Message::Text(json)).await.is_err() {
                    break;
                }
            }
        }
    });

    tokio::select! {
        _ = recv_task => {},
        _ = send_task => {},
    }
}

async fn set_speed(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SpeedRequest>,
) -> Json<StatusResponse> {
    state.set_delay(req.delay_ms);
    Json(StatusResponse {
        status: "ok".to_string(),
        delay_ms: req.delay_ms,
    })
}

async fn get_status(State(state): State<Arc<AppState>>) -> Json<StatusResponse> {
    Json(StatusResponse {
        status: "ready".to_string(),
        delay_ms: state.speed_ms.load(std::sync::atomic::Ordering::Relaxed),
    })
}

// ==================== 메인 ====================
#[tokio::main]
async fn main() {
    let (tx, _) = broadcast::channel::<ExecutionStep>(1000);

    let state = Arc::new(AppState {
        tx,
        speed_ms: std::sync::atomic::AtomicU64::new(500), // 기본 500ms
    });

    let app = Router::new()
        .route("/", get(index))
        .route("/styles.css", get(styles))
        .route("/app.js", get(script))
        .route("/ws", get(ws_handler))
        .route("/api/speed", post(set_speed))
        .route("/api/status", get(get_status))
        .layer(CorsLayer::permissive())
        .with_state(state);

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║     LLM Visualizer - Real-time Step-by-Step Execution     ║");
    println!("╠═══════════════════════════════════════════════════════════╣");
    println!("║  Server: http://localhost:3000                            ║");
    println!("║  WebSocket: ws://localhost:3000/ws                        ║");
    println!("║  Press Ctrl+C to stop                                     ║");
    println!("╚═══════════════════════════════════════════════════════════╝");

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
