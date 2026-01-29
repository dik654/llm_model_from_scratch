// ==================== State ====================
let ws = null;
let isRunning = false;
let isPaused = false;
let isManualMode = false;
let totalSteps = 0;
let currentStep = 0;
let tokens = [];
let tokenChars = [];
let currentLayer = 0;
let currentHead = 0;

// ==================== 교육적 설명 ====================
const EXPLANATIONS = {
    start: {
        title: "🚀 Transformer 시작",
        text: `텍스트를 AI가 이해할 수 있는 형태로 변환합니다.

• 문자열 → 토큰(숫자): 컴퓨터는 숫자만 처리할 수 있어서 각 글자를 고유한 숫자(토큰 ID)로 변환합니다.
• 예: "Hello" → [72, 101, 108, 108, 111] (각 글자의 ASCII 코드)
• 이 숫자들이 모델의 입력이 됩니다.`
    },

    tokenEmbedding: {
        title: "📊 토큰 임베딩 (Token Embedding)",
        text: `토큰 ID를 고차원 벡터로 변환합니다.

• 왜 필요한가?: 단순한 숫자(72)는 의미 정보가 없습니다.
  임베딩은 각 토큰을 수백 개의 숫자로 된 벡터로 표현해서 의미를 담습니다.

• 작동 방식: 거대한 룩업 테이블에서 토큰 ID에 해당하는 행을 가져옵니다.
  예: token_id=72 → embedding[72] = [0.12, -0.34, 0.56, ...]

• 임베딩 벡터의 각 숫자는 학습을 통해 의미 있는 특징을 갖게 됩니다.`
    },

    positionEncoding: {
        title: "📍 위치 인코딩 (Positional Encoding)",
        text: `토큰의 순서 정보를 추가합니다.

• 왜 필요한가?: Transformer는 순서를 모릅니다!
  "고양이가 쥐를 쫓는다"와 "쥐가 고양이를 쫓는다"가 같아 보입니다.

• 해결책: 각 위치에 고유한 sin/cos 패턴을 더합니다.
  - PE(pos, 2i) = sin(pos / 10000^(2i/d))
  - PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

• 결과: 임베딩 + 위치인코딩 = 의미 + 순서 정보를 모두 담은 벡터`
    },

    layerNorm: {
        title: "⚖️ 레이어 정규화 (Layer Normalization)",
        text: `값들을 안정적인 범위로 조정합니다.

• 왜 필요한가?: 신경망을 거치면서 값이 너무 커지거나 작아지면 학습이 불안정해집니다.

• 작동 방식:
  1. 평균(μ)을 빼서 중심을 0으로 맞춤
  2. 표준편차(σ)로 나눠서 스케일을 1로 맞춤
  3. x_norm = (x - μ) / σ

• 결과: 값들이 대략 -2 ~ +2 범위에 분포하게 됩니다.`
    },

    attentionStart: {
        title: "🔍 멀티헤드 어텐션 시작",
        text: `Self-Attention: "이 단어가 어떤 단어들과 관련 있는지" 파악합니다.

• 핵심 아이디어: 문장 내 모든 단어 쌍의 관련성을 계산합니다.
  예: "The cat sat on the mat"에서 "sat"은 "cat"과 강하게 연결

• Multi-Head: 여러 개의 어텐션을 병렬로 실행합니다.
  - 각 Head는 다른 관점에서 관계를 봅니다
  - Head 1: 문법적 관계, Head 2: 의미적 관계, ...

• 다음 단계에서 Q, K, V를 계산합니다.`
    },

    attentionQKV: {
        title: "🔑 Q, K, V 계산",
        text: `Query, Key, Value 벡터를 생성합니다.

• Q (Query): "내가 찾고 싶은 것" - 질문하는 역할
• K (Key): "내가 가진 정보의 라벨" - 검색 키 역할
• V (Value): "실제 정보" - 전달할 내용

• 비유: 도서관 검색
  - Query: "고양이에 관한 책 찾기"
  - Key: 각 책의 제목/키워드
  - Value: 책의 실제 내용

• 계산: 입력 x에 학습된 가중치 행렬을 곱합니다.
  Q = x × W_q,  K = x × W_k,  V = x × W_v`
    },

    attentionScores: {
        title: "📈 어텐션 스코어 계산",
        text: `Q와 K의 유사도를 계산합니다.

• Score = Q × K^T (내적)
  - 두 벡터가 비슷할수록 높은 점수
  - 결과: (시퀀스 길이 × 시퀀스 길이) 행렬

• 스케일링: Score / √d_k
  - d_k는 Key의 차원
  - 값이 너무 커지면 softmax가 극단적이 됨
  - √d_k로 나눠서 안정화

• Causal Mask (미래 숨기기):
  - GPT 같은 모델은 미래 토큰을 볼 수 없음
  - 미래 위치에 -∞를 넣어서 어텐션이 0이 되게 함`
    },

    attentionWeights: {
        title: "📊 어텐션 가중치 (Softmax)",
        text: `스코어를 확률로 변환합니다.

• Softmax 함수:
  - 모든 점수를 0~1 사이로 변환
  - 합이 1이 되도록 정규화
  - 높은 점수 → 높은 가중치

• 히트맵 해석:
  - 밝은 색: 강한 어텐션 (이 단어에 집중)
  - 어두운 색: 약한 어텐션 (무시)
  - 대각선 아래만 값이 있음 (Causal Mask 때문)

• 예: "The cat sat"에서 "sat"의 어텐션
  - "cat": 0.6 (주어와 강하게 연결)
  - "The": 0.3
  - "sat": 0.1`
    },

    attentionOutput: {
        title: "✨ 어텐션 출력",
        text: `가중치를 사용해 Value를 조합합니다.

• Output = Attention_Weights × V
  - 관련 있는 토큰의 정보를 더 많이 가져옴
  - 관련 없는 토큰은 무시됨

• Residual Connection (잔차 연결):
  - Output = x + Attention(x)
  - 원래 입력에 어텐션 결과를 더함
  - 정보 손실 방지, 학습 안정화

• 결과: 각 토큰이 문맥을 파악한 새로운 표현을 얻음`
    },

    ffn: {
        title: "🧠 피드포워드 네트워크 (FFN)",
        text: `각 위치에서 독립적으로 비선형 변환을 수행합니다.

• 구조: Linear → GELU → Linear
  1. 첫 번째 Linear: 차원 확장 (d_model → d_ff, 보통 4배)
  2. GELU 활성화: 비선형성 추가
  3. 두 번째 Linear: 차원 축소 (d_ff → d_model)

• GELU (Gaussian Error Linear Unit):
  - ReLU보다 부드러운 활성화 함수
  - 음수도 약간 통과시킴

• 역할: 어텐션이 "무엇에 집중할지" 결정했다면,
  FFN은 "그 정보를 어떻게 해석할지" 처리합니다.`
    },

    finalLogits: {
        title: "🎯 최종 출력 (LM Head)",
        text: `다음 토큰 확률을 계산합니다.

• LM Head: 마지막 Linear 레이어
  - 입력: (batch, seq, d_model)
  - 출력: (batch, seq, vocab_size)
  - 각 토큰 위치에서 모든 가능한 다음 토큰의 점수

• Softmax → 확률 분포:
  - 모든 점수를 확률로 변환
  - 가장 높은 확률의 토큰 = 예측 결과

• Top-K 예측:
  - 가장 확률 높은 K개 토큰을 보여줌
  - 모델이 얼마나 확신하는지 알 수 있음`
    },

    generated: {
        title: "✍️ 토큰 생성",
        text: `예측한 토큰을 선택하고 출력합니다.

• 샘플링 방법:
  - Greedy: 항상 가장 확률 높은 토큰 선택
  - Temperature: 확률 분포를 조절 (높으면 다양, 낮으면 보수적)
  - Top-K: 상위 K개 중에서 샘플링

• Autoregressive 생성:
  1. 입력 처리 → 다음 토큰 예측
  2. 예측된 토큰을 입력에 추가
  3. 다시 1번부터 반복

• 이 과정이 반복되며 텍스트가 생성됩니다!`
    },

    complete: {
        title: "🎉 완료!",
        text: `Transformer의 전체 Forward Pass가 완료되었습니다.

• 요약:
  1. 토큰화: 텍스트 → 숫자
  2. 임베딩: 숫자 → 의미 벡터
  3. 위치 인코딩: 순서 정보 추가
  4. Transformer 블록 × N:
     - Self-Attention: 단어 간 관계 파악
     - FFN: 정보 해석
  5. LM Head: 다음 토큰 예측

• 다른 텍스트로 다시 실험해보세요!`
    }
};

// ==================== WebSocket ====================
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        setConnectionStatus(true);
        setStatus('Connected - Ready');
    };

    ws.onclose = () => {
        setConnectionStatus(false);
        setStatus('Disconnected', true);
        setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setStatus('Connection error', true);
    };

    ws.onmessage = (event) => {
        try {
            const step = JSON.parse(event.data);
            handleExecutionStep(step);
        } catch (e) {
            console.error('Failed to parse message:', e);
        }
    };
}

function setConnectionStatus(connected) {
    const statusEl = document.getElementById('connection-status');
    const dot = statusEl.querySelector('.status-dot');
    const text = statusEl.querySelector('span:last-child');

    if (connected) {
        dot.classList.add('connected');
        text.textContent = 'Connected';
    } else {
        dot.classList.remove('connected');
        text.textContent = 'Disconnected';
    }
}

// ==================== Execution Step Handler ====================
function handleExecutionStep(step) {
    currentStep++;
    updateProgress();

    switch (step.type) {
        case 'Start':
            handleStart(step);
            break;
        case 'TokenEmbedding':
            handleTokenEmbedding(step);
            break;
        case 'PositionEncoding':
            handlePositionEncoding(step);
            break;
        case 'LayerNorm':
            handleLayerNorm(step);
            break;
        case 'AttentionStart':
            handleAttentionStart(step);
            break;
        case 'AttentionQKV':
            handleAttentionQKV(step);
            break;
        case 'AttentionScores':
            handleAttentionScores(step);
            break;
        case 'AttentionWeights':
            handleAttentionWeights(step);
            break;
        case 'AttentionOutput':
            handleAttentionOutput(step);
            break;
        case 'FeedForward':
            handleFeedForward(step);
            break;
        case 'FinalLogits':
            handleFinalLogits(step);
            break;
        case 'GeneratedToken':
            handleGeneratedToken(step);
            break;
        case 'Complete':
            handleComplete(step);
            break;
        case 'Error':
            handleError(step);
            break;
    }
}

// ==================== Step Handlers ====================
function handleStart(step) {
    isRunning = true;
    currentStep = 0;
    currentLayer = 0;
    currentHead = 0;
    tokens = step.tokens;
    tokenChars = tokens.map(t => t < 128 ? String.fromCharCode(t) : `[${t}]`);

    const numLayers = 4;
    const numHeads = 4;
    totalSteps = tokens.length * 2 + numLayers * (1 + numHeads * 4 + 4) + 5;

    clearAllHighlights();
    highlightArchBlock('arch-input');

    displayInputTokens(tokenChars);
    displayTokenSequence(tokenChars);

    setStepInfo('Start', `입력: "${step.text}" → ${tokens.length}개 토큰으로 변환`);
    setExplanation(EXPLANATIONS.start);
    setCode(`// Transformer Forward Pass 시작
// 입력 텍스트: "${step.text}"
// 토큰 ID: [${tokens.join(', ')}]
// 문자: [${tokenChars.map(c => `'${c}'`).join(', ')}]

// 각 글자가 숫자(토큰 ID)로 변환되었습니다.
// 이 숫자들이 모델의 입력이 됩니다.`);

    addLog('start', `시작: "${step.text}"`);
    setStatus('Running...');

    document.getElementById('btn-run').disabled = true;
    document.getElementById('btn-stop').disabled = false;
    updateManualButtons(true);

    // 수동 모드면 첫 스텝 후 자동 일시정지
    if (isManualMode) {
        updatePauseButton(true);
    }
}

function handleTokenEmbedding(step) {
    highlightArchBlock('arch-embed');

    updateTokenState(step.step, 'processing');
    if (step.step > 0) updateTokenState(step.step - 1, 'embedded');
    updateInputTokenState(step.step, 'processing');

    setStepInfo('Token Embedding', `토큰 ${step.step}: "${step.token_char}" (ID: ${step.token_id})`);
    setExplanation(EXPLANATIONS.tokenEmbedding);
    setCode(`// 토큰 임베딩
// "${step.token_char}" (ID: ${step.token_id})를 벡터로 변환

embedding = embedding_table[${step.token_id}]
// 결과: ${step.embedding_sample.length}차원 벡터

// 이 벡터의 각 숫자는 해당 토큰의 "의미"를 표현합니다.
// 비슷한 의미의 단어는 비슷한 벡터를 가집니다.`);

    showValueCard('embed-value-card', 'embed-value',
        `[${step.embedding_sample.map(v => v.toFixed(3)).join(', ')}...]`);

    showTensorViz('임베딩 벡터', `토큰 "${step.token_char}" → ${step.embedding_sample.length}차원`, step.embedding_sample);

    addLog('embed', `"${step.token_char}" → 임베딩[${step.token_id}]`);
}

function handlePositionEncoding(step) {
    highlightArchBlock('arch-pos');

    updateInputTokenState(step.position, 'done');

    setStepInfo('Positional Encoding', `위치 ${step.position}: sin/cos 패턴 추가`);
    setExplanation(EXPLANATIONS.positionEncoding);
    setCode(`// 위치 인코딩 추가
// 위치 ${step.position}에 고유한 패턴을 더합니다

for i in 0..d_model:
    if i % 2 == 0:
        PE[${step.position}, i] = sin(${step.position} / 10000^(i/d))
    else:
        PE[${step.position}, i] = cos(${step.position} / 10000^(i/d))

x[${step.position}] = embedding[${step.position}] + PE[${step.position}]

// 이제 모델이 이 토큰이 ${step.position}번째임을 알 수 있습니다.`);

    showValueCard('embed-value-card', 'embed-value',
        `위치 ${step.position}:\n[${step.encoding_sample.map(v => v.toFixed(3)).join(', ')}...]`);

    showTensorViz('위치 인코딩', `위치 ${step.position}의 sin/cos 패턴`, step.encoding_sample);

    addLog('embed', `위치 ${step.position}: sin/cos 인코딩 추가`);
}

function handleLayerNorm(step) {
    currentLayer = step.layer;

    if (step.location === 'pre_attention') {
        highlightArchBlock('arch-ln1');
    } else {
        highlightArchBlock('arch-ln2');
    }

    setStepInfo('Layer Normalization', `레이어 ${step.layer} - ${step.location === 'pre_attention' ? 'Attention 전' : 'FFN 전'}`);
    setExplanation(EXPLANATIONS.layerNorm);
    setCode(`// 레이어 정규화
// 값들을 안정적인 범위로 조정합니다

평균(μ) = ${step.mean.toFixed(6)}
표준편차(σ) = ${step.std.toFixed(6)}

x_normalized = (x - μ) / σ

// 정규화 후 값들은 대략 -2 ~ +2 범위에 분포합니다.
// 이렇게 하면 학습이 안정적으로 진행됩니다.`);

    showValueCard('norm-value-card', 'norm-value',
        `평균(μ) = ${step.mean.toFixed(4)}\n표준편차(σ) = ${step.std.toFixed(4)}`);

    addLog('norm', `LayerNorm L${step.layer}: μ=${step.mean.toFixed(3)}, σ=${step.std.toFixed(3)}`);
}

function handleAttentionStart(step) {
    currentLayer = step.layer;
    currentHead = step.head;

    highlightArchBlock('arch-attention');
    clearFormulaHighlights();

    document.querySelectorAll('.qkv-block').forEach(b => b.classList.remove('active'));

    setStepInfo('Multi-Head Attention', `레이어 ${step.layer}, 헤드 ${step.head}`);
    setExplanation(EXPLANATIONS.attentionStart);
    setCode(`// Multi-Head Self-Attention 시작
// 레이어 ${step.layer}, 헤드 ${step.head}

// Self-Attention은 "이 단어가 다른 어떤 단어들과
// 관련이 있는지"를 계산합니다.

// 각 헤드는 서로 다른 관점에서 관계를 파악합니다.
// 예: 헤드1은 문법적 관계, 헤드2는 의미적 관계...`);

    addLog('attention', `레이어 ${step.layer} 헤드 ${step.head}: Attention 시작`);
}

function handleAttentionQKV(step) {
    highlightArchBlock('arch-attention');

    document.getElementById('q-block').classList.add('active');
    document.getElementById('k-block').classList.add('active');
    document.getElementById('v-block').classList.add('active');

    document.getElementById('q-matrix').textContent = `[${step.q_sample.slice(0, 3).map(v => v.toFixed(2)).join(', ')}...]`;
    document.getElementById('k-matrix').textContent = `[${step.k_sample.slice(0, 3).map(v => v.toFixed(2)).join(', ')}...]`;
    document.getElementById('v-matrix').textContent = `[${step.v_sample.slice(0, 3).map(v => v.toFixed(2)).join(', ')}...]`;

    setStepInfo('Q, K, V 계산', `레이어 ${step.layer}, 헤드 ${step.head}`);
    setExplanation(EXPLANATIONS.attentionQKV);
    setCode(`// Q, K, V 벡터 생성
Q = x @ W_q   // Query: "무엇을 찾을까?"
K = x @ W_k   // Key: "내가 가진 정보의 라벨"
V = x @ W_v   // Value: "실제 전달할 정보"

// Q, K, V 샘플 값:
Q = [${step.q_sample.slice(0, 4).map(v => v.toFixed(3)).join(', ')}...]
K = [${step.k_sample.slice(0, 4).map(v => v.toFixed(3)).join(', ')}...]
V = [${step.v_sample.slice(0, 4).map(v => v.toFixed(3)).join(', ')}...]`);

    showValueCard('qkv-value-card', 'qkv-value',
        `Q: [${step.q_sample.slice(0, 5).map(v => v.toFixed(3)).join(', ')}...]\n` +
        `K: [${step.k_sample.slice(0, 5).map(v => v.toFixed(3)).join(', ')}...]\n` +
        `V: [${step.v_sample.slice(0, 5).map(v => v.toFixed(3)).join(', ')}...]`);

    showMatrixOperation(step.q_sample, step.k_sample, null, 'Q', 'K<sup>T</sup>', 'Scores');
    highlightFormulaPart('formula-qk');

    addLog('attention', `Q/K/V 계산 완료 (헤드 ${step.head})`);
}

function handleAttentionScores(step) {
    highlightFormulaPart('formula-scale');
    highlightFormulaPart('formula-mask');

    setStepInfo('Attention Score 계산', `레이어 ${step.layer}, 헤드 ${step.head}`);
    setExplanation(EXPLANATIONS.attentionScores);
    setCode(`// Attention Score 계산
scores = Q @ K.transpose()  // Q와 K의 내적
scores = scores / sqrt(d_k) // 스케일링 (값 안정화)

// Causal Mask 적용 (GPT 스타일)
// 미래 토큰은 볼 수 없으므로 -∞로 마스킹
for i, j in scores:
    if j > i:  // 미래 위치
        scores[i][j] = -infinity

// -∞는 softmax 후 0이 됩니다.`);

    if (step.scores && step.scores.length > 0) {
        updateMatrixResult(step.scores, 'Scores');
    }

    addLog('attention', `Score 계산: (Q × K^T) / √d_k + Mask`);
}

function handleAttentionWeights(step) {
    highlightFormulaPart('formula-softmax');

    setStepInfo('Softmax → Attention Weights', `레이어 ${step.layer}, 헤드 ${step.head}`);
    setExplanation(EXPLANATIONS.attentionWeights);
    setCode(`// Softmax로 확률 변환
attention_weights = softmax(scores)

// 각 행의 합 = 1.0
// 높은 점수 → 높은 가중치 (더 많이 주목)
// 낮은 점수 → 낮은 가중치 (덜 주목)

// 히트맵에서:
// 밝은 색 = 강한 어텐션 (집중!)
// 어두운 색 = 약한 어텐션 (무시)`);

    document.getElementById('attention-heatmap-section').style.display = 'block';
    document.getElementById('attn-layer-head').textContent = `레이어 ${step.layer}, 헤드 ${step.head}`;
    renderAttentionHeatmap(step.weights, step.tokens, step.layer, step.head);

    addLog('attention', `Softmax 적용: Attention 가중치 계산`);
}

function handleAttentionOutput(step) {
    highlightFormulaPart('formula-v');

    document.querySelectorAll('.qkv-block').forEach(b => b.classList.remove('active'));

    setStepInfo('Attention Output', `레이어 ${step.layer}`);
    setExplanation(EXPLANATIONS.attentionOutput);
    setCode(`// Attention 출력 계산
output = attention_weights @ V

// 가중치가 높은 토큰의 Value를 더 많이 가져옴
// = 관련 있는 정보를 더 많이 수집

// Residual Connection (잔차 연결)
x = x + output

// 원래 입력에 어텐션 결과를 더합니다.
// 이렇게 하면 정보가 손실되지 않습니다.`);

    showValueCard('attn-value-card', 'attn-value',
        `[${step.output_sample.map(v => v.toFixed(3)).join(', ')}...]`);

    document.getElementById('matrix-operation').style.display = 'none';

    addLog('attention', `Attention 출력: weights × V + residual`);
}

function handleFeedForward(step) {
    highlightArchBlock('arch-ffn');

    clearFFNHighlights();
    const stageId = {
        'linear1': 'ffn-linear1',
        'gelu': 'ffn-gelu',
        'linear2': 'ffn-linear2'
    }[step.stage];
    if (stageId) document.getElementById(stageId).classList.add('active');

    const stageNames = {
        'linear1': 'FFN: Linear 1 (차원 확장)',
        'gelu': 'FFN: GELU 활성화',
        'linear2': 'FFN: Linear 2 (차원 축소)'
    };

    setStepInfo(stageNames[step.stage] || `FFN: ${step.stage}`, `레이어 ${step.layer}`);
    setExplanation(EXPLANATIONS.ffn);

    const stageCode = {
        'linear1': `// FFN: 첫 번째 Linear 레이어
h = x @ W1 + b1
// 차원: ${64} → ${256} (4배 확장)

// 더 높은 차원에서 복잡한 패턴을 학습합니다.`,
        'gelu': `// FFN: GELU 활성화 함수
h = GELU(h)
// GELU(x) = x * Φ(x)  (Φ는 정규분포 CDF)

// 비선형성을 추가합니다.
// ReLU보다 부드럽고, 음수도 약간 통과시킵니다.`,
        'linear2': `// FFN: 두 번째 Linear 레이어
output = h @ W2 + b2
// 차원: ${256} → ${64} (원래 크기로)

// Residual Connection
x = x + output`
    };

    setCode(stageCode[step.stage] || step.code);

    showValueCard('ffn-value-card', 'ffn-value',
        `[${step.output_sample.map(v => v.toFixed(3)).join(', ')}...]`);

    showTensorViz(`FFN ${step.stage}`, `레이어 ${step.layer}`, step.output_sample);

    addLog('ffn', `레이어 ${step.layer} ${step.stage}`);
}

function handleFinalLogits(step) {
    highlightArchBlock('arch-lm-head');
    highlightArchBlock('arch-output');

    tokenChars.forEach((_, i) => updateTokenState(i, 'complete'));

    setStepInfo('Final Output', '다음 토큰 확률 계산');
    setExplanation(EXPLANATIONS.finalLogits);
    setCode(`// LM Head: 최종 출력 레이어
logits = x @ W_lm + b_lm
// 출력 shape: (batch, seq_len, vocab_size)

// Softmax로 확률 변환
probs = softmax(logits[-1])  // 마지막 위치만

// Top-5 예측:
${step.top_tokens.map((t, i) =>
    `// ${i+1}. "${t.token_char}" - ${(t.probability * 100).toFixed(1)}%`
).join('\n')}`);

    document.getElementById('predictions-section').style.display = 'block';
    showPredictions(step.top_tokens);

    addLog('output', `Top: "${step.top_tokens[0]?.token_char}" (${(step.top_tokens[0]?.probability * 100).toFixed(1)}%)`);
}

function handleGeneratedToken(step) {
    setStepInfo('토큰 생성', `"${step.token_char}" 생성 (확률: ${(step.probability * 100).toFixed(1)}%)`);
    setExplanation(EXPLANATIONS.generated);
    setCode(`// 토큰 생성
선택된 토큰: "${step.token_char}" (ID: ${step.token_id})
확률: ${(step.probability * 100).toFixed(2)}%

// 현재까지 생성된 텍스트:
"${step.generated_text}"

// 이 토큰이 입력에 추가되고,
// 다시 Forward Pass가 실행됩니다.`);

    document.getElementById('generation-section').style.display = 'block';
    document.getElementById('generated-text').textContent = step.generated_text;

    addGeneratedToken(step.token_char);

    addLog('output', `생성: "${step.token_char}"`);
}

function handleComplete(step) {
    isRunning = false;
    clearAllHighlights();

    setStepInfo('완료!', `총 ${step.total_steps} 스텝`);
    setExplanation(EXPLANATIONS.complete);
    setCode(`// Transformer Forward Pass 완료!

총 스텝 수: ${step.total_steps}
최종 출력: "${step.final_text}"

// 다른 텍스트로 다시 실험해보세요!
// 속도를 조절하면서 각 단계를 자세히 살펴볼 수 있습니다.`);

    setStatus('완료');
    addLog('output', `완료: ${step.total_steps} 스텝`);

    document.getElementById('btn-run').disabled = false;
    document.getElementById('btn-stop').disabled = true;
    updateManualButtons(false);
    updatePauseButton(false);
}

function handleError(step) {
    isRunning = false;
    setStepInfo('오류', step.message);
    setStatus('오류: ' + step.message, true);
    addLog('output', `오류: ${step.message}`);

    document.getElementById('btn-run').disabled = false;
    document.getElementById('btn-stop').disabled = true;
}

// ==================== UI Updates ====================
function setStepInfo(name, description) {
    document.getElementById('step-number').textContent = `Step ${currentStep}`;
    document.getElementById('step-name').textContent = name;
    document.getElementById('step-description').textContent = description;
}

function setExplanation(explanation) {
    document.getElementById('explanation-title').textContent = explanation.title;
    document.getElementById('explanation-text').textContent = explanation.text;
}

function setCode(code) {
    document.getElementById('code-content').textContent = code;
}

function setStatus(text, isError = false) {
    const el = document.getElementById('status-text');
    el.textContent = text;
    el.style.color = isError ? 'var(--color-output)' : 'var(--color-input)';
    document.getElementById('status-time').textContent = new Date().toLocaleTimeString();
}

function updateProgress() {
    const percent = totalSteps > 0 ? Math.min((currentStep / totalSteps) * 100, 100) : 0;
    document.getElementById('progress-fill').style.width = `${percent}%`;
    document.getElementById('progress-text').textContent = `${Math.round(percent)}%`;
}

// ==================== Architecture Diagram ====================
function highlightArchBlock(blockId) {
    document.querySelectorAll('.arch-block').forEach(b => b.classList.remove('active'));
    const block = document.getElementById(blockId);
    if (block) {
        block.classList.add('active');
        block.classList.add('animating');
        setTimeout(() => block.classList.remove('animating'), 500);
    }
}

function clearAllHighlights() {
    document.querySelectorAll('.arch-block').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.qkv-block').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.formula-part').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.ffn-step').forEach(s => s.classList.remove('active'));
}

function highlightFormulaPart(partId) {
    const part = document.getElementById(partId);
    if (part) part.classList.add('active');
}

function clearFormulaHighlights() {
    document.querySelectorAll('.formula-part').forEach(p => p.classList.remove('active'));
}

function clearFFNHighlights() {
    document.querySelectorAll('.ffn-step').forEach(s => s.classList.remove('active'));
}

// ==================== Token Display ====================
function displayInputTokens(chars) {
    const container = document.getElementById('input-tokens');
    container.innerHTML = chars.map((c, i) =>
        `<span class="input-token" data-idx="${i}">${c}</span>`
    ).join('');
}

function updateInputTokenState(idx, state) {
    const token = document.querySelector(`.input-token[data-idx="${idx}"]`);
    if (token) {
        token.classList.remove('processing', 'done');
        token.classList.add(state);
    }
}

function displayTokenSequence(chars) {
    const container = document.getElementById('token-sequence');
    container.innerHTML = chars.map((c, i) =>
        `<span class="token-item" data-idx="${i}">${c}</span>`
    ).join('');
}

function updateTokenState(idx, state) {
    const token = document.querySelector(`.token-item[data-idx="${idx}"]`);
    if (token) {
        token.classList.remove('processing', 'embedded', 'complete');
        token.classList.add(state);
    }
}

function addGeneratedToken(char) {
    const container = document.getElementById('token-sequence');
    const span = document.createElement('span');
    span.className = 'token-item generated';
    span.textContent = char;
    container.appendChild(span);
}

// ==================== Value Cards ====================
function showValueCard(cardId, valueId, text) {
    document.getElementById(cardId).style.display = 'block';
    document.getElementById(valueId).textContent = text;
}

function hideAllValueCards() {
    ['embed-value-card', 'qkv-value-card', 'attn-value-card', 'ffn-value-card', 'norm-value-card']
        .forEach(id => document.getElementById(id).style.display = 'none');
}

// ==================== Tensor Visualization ====================
function showTensorViz(label, shape, values) {
    document.getElementById('tensor-label').textContent = label;
    document.getElementById('tensor-shape').textContent = shape;

    const grid = document.getElementById('tensor-grid');
    grid.innerHTML = '';

    values.slice(0, 16).forEach(v => {
        const cell = document.createElement('div');
        cell.className = 'tensor-cell';
        cell.textContent = v.toFixed(2);
        const intensity = Math.min(Math.abs(v) * 100, 100);
        const hue = v >= 0 ? 200 : 0;
        cell.style.background = `hsla(${hue}, 70%, 50%, ${intensity / 100})`;
        cell.style.color = intensity > 50 ? 'white' : 'var(--text-primary)';
        grid.appendChild(cell);
    });
}

// ==================== Matrix Operation ====================
function showMatrixOperation(a, b, result, titleA, titleB, titleResult) {
    const container = document.getElementById('matrix-operation');
    container.style.display = 'block';

    document.querySelector('#matrix-a .matrix-title').innerHTML = titleA;
    document.querySelector('#matrix-b .matrix-title').innerHTML = titleB;
    document.querySelector('#matrix-result .matrix-title').textContent = titleResult;

    renderSmallMatrix('matrix-a-grid', a.slice(0, 4), 1, 4);
    renderSmallMatrix('matrix-b-grid', b.slice(0, 4), 4, 1);
    document.getElementById('matrix-result-grid').innerHTML = '<div class="matrix-cell">...</div>';
}

function renderSmallMatrix(gridId, values, rows, cols) {
    const grid = document.getElementById(gridId);
    grid.innerHTML = '';
    grid.style.gridTemplateColumns = `repeat(${cols}, 36px)`;

    values.forEach(v => {
        const cell = document.createElement('div');
        cell.className = 'matrix-cell';
        cell.textContent = v.toFixed(1);
        const intensity = Math.min(Math.abs(v) * 50, 100);
        cell.style.background = `rgba(88, 166, 255, ${intensity / 100})`;
        grid.appendChild(cell);
    });
}

function updateMatrixResult(scores, title) {
    const grid = document.getElementById('matrix-result-grid');
    grid.innerHTML = '';

    const size = Math.min(scores.length, 4);
    grid.style.gridTemplateColumns = `repeat(${size}, 36px)`;

    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            const cell = document.createElement('div');
            cell.className = 'matrix-cell';
            const val = scores[i][j];
            if (val === -Infinity || val < -1000) {
                cell.textContent = '-∞';
                cell.style.background = 'rgba(248, 81, 73, 0.3)';
            } else {
                cell.textContent = val.toFixed(1);
                const intensity = Math.min(Math.abs(val) * 30, 100);
                cell.style.background = `rgba(88, 166, 255, ${intensity / 100})`;
            }
            grid.appendChild(cell);
        }
    }
}

// ==================== Attention Heatmap ====================
function renderAttentionHeatmap(weights, tokens, layer, head) {
    const data = [{
        z: weights,
        x: tokens,
        y: tokens,
        type: 'heatmap',
        colorscale: [
            [0, '#0d1117'],
            [0.2, '#21262d'],
            [0.4, '#58a6ff'],
            [0.7, '#a371f7'],
            [1, '#f0883e']
        ],
        showscale: true,
        colorbar: {
            title: { text: '가중치', font: { color: '#8b949e', size: 12 } },
            tickfont: { color: '#8b949e', size: 11 }
        }
    }];

    const layout = {
        paper_bgcolor: '#21262d',
        plot_bgcolor: '#21262d',
        font: { color: '#e6edf3', size: 12 },
        xaxis: {
            title: { text: 'Key (정보 제공)', font: { size: 12 } },
            tickangle: -45,
            tickfont: { size: 11 }
        },
        yaxis: {
            title: { text: 'Query (질문)', font: { size: 12 } },
            autorange: 'reversed',
            tickfont: { size: 11 }
        },
        margin: { l: 60, r: 50, t: 30, b: 60 }
    };

    Plotly.newPlot('attention-heatmap', data, layout, { responsive: true });
}

// ==================== Predictions ====================
function showPredictions(topTokens) {
    const container = document.getElementById('predictions-list');
    container.innerHTML = '';

    topTokens.forEach((pred, i) => {
        const percent = (pred.probability * 100).toFixed(1);
        const item = document.createElement('div');
        item.className = 'prediction-item';
        item.innerHTML = `
            <span class="prediction-rank">${i + 1}.</span>
            <span class="prediction-token">"${pred.token_char}"</span>
            <div class="prediction-bar">
                <div class="prediction-fill" style="width: ${percent}%"></div>
            </div>
            <span class="prediction-prob">${percent}%</span>
        `;
        container.appendChild(item);
    });
}

// ==================== Log ====================
function addLog(type, message) {
    const container = document.getElementById('log-container');
    const entry = document.createElement('div');
    entry.className = 'log-entry';

    const time = new Date().toLocaleTimeString('en-US', {
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });

    entry.innerHTML = `
        <span class="log-time">${time}</span>
        <span class="log-type ${type}">${type.toUpperCase()}</span>
        <span class="log-message">${message}</span>
    `;

    container.appendChild(entry);
    container.scrollTop = container.scrollHeight;
}

// ==================== Utilities ====================
function resetUI() {
    currentStep = 0;
    totalSteps = 0;
    isPaused = false;

    document.getElementById('progress-fill').style.width = '0%';
    document.getElementById('progress-text').textContent = '0%';
    document.getElementById('token-sequence').innerHTML = '';
    document.getElementById('input-tokens').innerHTML = '';
    document.getElementById('generation-section').style.display = 'none';
    document.getElementById('predictions-section').style.display = 'none';
    document.getElementById('attention-heatmap-section').style.display = 'none';
    document.getElementById('matrix-operation').style.display = 'none';

    hideAllValueCards();
    clearAllHighlights();
    updateManualButtons(false);
    updatePauseButton(false);

    document.getElementById('q-matrix').textContent = '';
    document.getElementById('k-matrix').textContent = '';
    document.getElementById('v-matrix').textContent = '';

    setStepInfo('준비', '"Run" 버튼을 눌러 시작하세요');
    setExplanation({
        title: "🎓 Transformer 시각화 도구",
        text: `이 도구는 Transformer 모델의 동작을 단계별로 시각화합니다.

• 왼쪽: Transformer 아키텍처 다이어그램
• 가운데: 현재 단계 설명 및 데이터 시각화
• 오른쪽: 토큰 상태 및 실행 로그

"Run" 버튼을 눌러 텍스트가 어떻게 처리되는지 확인해보세요!`
    });
    setCode(`// Transformer Forward Pass
// "Run" 버튼을 눌러 시작하세요

// 전체 흐름:
// 1. Token Embedding: 글자 → 벡터
// 2. Position Encoding: 위치 정보 추가
// 3. Transformer Block × N:
//    - Self-Attention: 단어 간 관계 파악
//    - FFN: 정보 해석
// 4. LM Head: 다음 토큰 예측`);
}

// ==================== Manual Control Functions ====================
function setManualMode(enabled) {
    isManualMode = enabled;

    // Update button states
    document.getElementById('mode-auto').classList.toggle('active', !enabled);
    document.getElementById('mode-manual').classList.toggle('active', enabled);

    // Show/hide manual controls
    document.getElementById('manual-controls').style.display = enabled ? 'block' : 'none';

    // Send to server
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ command: enabled ? 'manual_on' : 'manual_off' }));
    }

    addLog('control', enabled ? '수동 모드 활성화' : '자동 모드로 전환');
}

function sendControlCommand(command) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ command: command }));
    }
}

function updateManualButtons(enabled) {
    document.getElementById('btn-prev').disabled = !enabled;
    document.getElementById('btn-pause').disabled = !enabled;
    document.getElementById('btn-next').disabled = !enabled;
}

function updatePauseButton(paused) {
    const btn = document.getElementById('btn-pause');
    btn.textContent = paused ? '▶ 재개' : '⏸ 일시정지';
    isPaused = paused;
}

// ==================== Event Listeners ====================
document.getElementById('btn-run').addEventListener('click', () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        setStatus('연결되지 않음', true);
        return;
    }

    const text = document.getElementById('input-text').value.trim();
    if (!text) {
        setStatus('텍스트를 입력해주세요', true);
        return;
    }

    const generateTokens = parseInt(document.getElementById('gen-count').value);

    resetUI();
    updateManualButtons(true);

    ws.send(JSON.stringify({
        text: text,
        generate_tokens: generateTokens
    }));
});

document.getElementById('btn-stop').addEventListener('click', () => {
    setStatus('중지 요청...');
    updateManualButtons(false);
});

document.getElementById('btn-clear-log').addEventListener('click', () => {
    document.getElementById('log-container').innerHTML = '';
});

document.getElementById('gen-count').addEventListener('input', (e) => {
    document.getElementById('gen-count-value').textContent = e.target.value;
});

document.getElementById('speed').addEventListener('input', (e) => {
    const value = e.target.value;
    document.getElementById('speed-value').textContent = value;

    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ delay_ms: parseInt(value) }));
    }
});

// Mode toggle
document.getElementById('mode-auto').addEventListener('click', () => {
    setManualMode(false);
});

document.getElementById('mode-manual').addEventListener('click', () => {
    setManualMode(true);
});

// Manual control buttons
document.getElementById('btn-prev').addEventListener('click', () => {
    sendControlCommand('prev');
    addLog('control', '이전 스텝으로 이동');
});

document.getElementById('btn-pause').addEventListener('click', () => {
    if (isPaused) {
        sendControlCommand('resume');
        updatePauseButton(false);
        addLog('control', '재개');
    } else {
        sendControlCommand('pause');
        updatePauseButton(true);
        addLog('control', '일시정지');
    }
});

document.getElementById('btn-next').addEventListener('click', () => {
    sendControlCommand('next');
    sendControlCommand('resume');  // Resume to advance one step
    addLog('control', '다음 스텝으로 이동');
});

// ==================== Init ====================
window.addEventListener('DOMContentLoaded', () => {
    connectWebSocket();
    resetUI();
    setStatus('연결 중...');
});

window.addEventListener('beforeunload', () => {
    if (ws) ws.close();
});
