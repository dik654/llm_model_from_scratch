// ==================== State ====================
let ws = null;
let isRunning = false;
let totalSteps = 0;
let currentStep = 0;
let tokens = [];
let tokenChars = [];
let currentLayer = 0;
let currentHead = 0;

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

    // 예상 스텝 수
    const numLayers = 4;
    const numHeads = 4;
    totalSteps = tokens.length * 2 + numLayers * (1 + numHeads * 4 + 4) + 5;

    // UI 초기화
    clearAllHighlights();
    highlightArchBlock('arch-input');

    // 입력 토큰 표시
    displayInputTokens(tokenChars);
    displayTokenSequence(tokenChars);

    setStepInfo('Start', `Input: "${step.text}" → ${tokens.length} tokens`);
    setCode(`// Starting Transformer Forward Pass
// Input: "${step.text}"
// Tokens: [${tokens.join(', ')}]
// Characters: [${tokenChars.map(c => `'${c}'`).join(', ')}]`);

    addLog('start', `Started: "${step.text}"`);
    setStatus('Running...');

    document.getElementById('btn-run').disabled = true;
    document.getElementById('btn-stop').disabled = false;
}

function handleTokenEmbedding(step) {
    highlightArchBlock('arch-embed');

    // 토큰 상태 업데이트
    updateTokenState(step.step, 'processing');
    if (step.step > 0) updateTokenState(step.step - 1, 'embedded');
    updateInputTokenState(step.step, 'processing');

    setStepInfo('Token Embedding', `Token ${step.step}: "${step.token_char}" (ID: ${step.token_id})`);
    setCode(step.code);

    // 임베딩 값 표시
    showValueCard('embed-value-card', 'embed-value', formatVector(step.embedding_sample));

    // 텐서 시각화
    showTensorViz('Embedding Vector', `Shape: (1, ${step.embedding_sample.length}...)`, step.embedding_sample);

    addLog('embed', `Token "${step.token_char}" → embedding[${step.token_id}]`);
}

function handlePositionEncoding(step) {
    highlightArchBlock('arch-pos');

    updateInputTokenState(step.position, 'done');

    setStepInfo('Positional Encoding', `Position ${step.position}: Adding sin/cos pattern`);
    setCode(step.code);

    showValueCard('embed-value-card', 'embed-value',
        `Position ${step.position}:\n${formatVector(step.encoding_sample)}`);

    showTensorViz('Position Encoding', `PE(${step.position})`, step.encoding_sample);

    addLog('embed', `Position ${step.position}: sin/cos encoding added`);
}

function handleLayerNorm(step) {
    currentLayer = step.layer;

    if (step.location === 'pre_attention') {
        highlightArchBlock('arch-ln1');
    } else {
        highlightArchBlock('arch-ln2');
    }

    setStepInfo('Layer Normalization', `Layer ${step.layer} - ${step.location}`);
    setCode(step.code);

    showValueCard('norm-value-card', 'norm-value',
        `μ = ${step.mean.toFixed(6)}\nσ = ${step.std.toFixed(6)}`);

    addLog('norm', `LayerNorm L${step.layer}: μ=${step.mean.toFixed(4)}, σ=${step.std.toFixed(4)}`);
}

function handleAttentionStart(step) {
    currentLayer = step.layer;
    currentHead = step.head;

    highlightArchBlock('arch-attention');
    clearFormulaHighlights();

    // Q, K, V 블록 초기화
    document.querySelectorAll('.qkv-block').forEach(b => b.classList.remove('active'));

    setStepInfo('Multi-Head Attention', `Layer ${step.layer}, Head ${step.head}`);
    setCode(step.code);

    addLog('attention', `Layer ${step.layer} Head ${step.head}: Starting attention`);
}

function handleAttentionQKV(step) {
    highlightArchBlock('arch-attention');

    // Q, K, V 블록 활성화
    document.getElementById('q-block').classList.add('active');
    document.getElementById('k-block').classList.add('active');
    document.getElementById('v-block').classList.add('active');

    // Q, K, V 값 표시
    document.getElementById('q-matrix').textContent = `[${step.q_sample.slice(0, 3).map(v => v.toFixed(2)).join(', ')}...]`;
    document.getElementById('k-matrix').textContent = `[${step.k_sample.slice(0, 3).map(v => v.toFixed(2)).join(', ')}...]`;
    document.getElementById('v-matrix').textContent = `[${step.v_sample.slice(0, 3).map(v => v.toFixed(2)).join(', ')}...]`;

    setStepInfo('Attention: Q, K, V Projection', `Layer ${step.layer}, Head ${step.head}`);
    setCode(step.code);

    showValueCard('qkv-value-card', 'qkv-value',
        `Q: [${formatVectorShort(step.q_sample)}]\n` +
        `K: [${formatVectorShort(step.k_sample)}]\n` +
        `V: [${formatVectorShort(step.v_sample)}]`);

    // 행렬 시각화
    showMatrixOperation(step.q_sample, step.k_sample, null, 'Q', 'K^T', 'Scores');
    highlightFormulaPart('formula-qk');

    addLog('attention', `Q/K/V projected for head ${step.head}`);
}

function handleAttentionScores(step) {
    highlightFormulaPart('formula-scale');
    highlightFormulaPart('formula-mask');

    setStepInfo('Attention: Scaled Scores + Mask', `Layer ${step.layer}, Head ${step.head}`);
    setCode(step.code);

    // 점수 행렬 시각화
    if (step.scores && step.scores.length > 0) {
        updateMatrixResult(step.scores, 'Scores');
    }

    addLog('attention', `Scores computed: (Q × K^T) / √d_k + mask`);
}

function handleAttentionWeights(step) {
    highlightFormulaPart('formula-softmax');

    setStepInfo('Attention: Softmax Weights', `Layer ${step.layer}, Head ${step.head}`);
    setCode(step.code);

    // Attention 히트맵 표시
    document.getElementById('attention-heatmap-section').style.display = 'block';
    document.getElementById('attn-layer-head').textContent = `Layer ${step.layer}, Head ${step.head}`;
    renderAttentionHeatmap(step.weights, step.tokens, step.layer, step.head);

    addLog('attention', `Softmax applied: attention weights computed`);
}

function handleAttentionOutput(step) {
    highlightFormulaPart('formula-v');

    // Q, K, V 블록 비활성화
    document.querySelectorAll('.qkv-block').forEach(b => b.classList.remove('active'));

    setStepInfo('Attention: Output + Residual', `Layer ${step.layer}`);
    setCode(step.code);

    showValueCard('attn-value-card', 'attn-value', formatVector(step.output_sample));

    document.getElementById('matrix-operation').style.display = 'none';

    addLog('attention', `Attention output: weights × V + residual`);
}

function handleFeedForward(step) {
    highlightArchBlock('arch-ffn');

    // FFN 단계 하이라이트
    clearFFNHighlights();
    if (step.stage === 'linear1') {
        document.getElementById('ffn-linear1').classList.add('active');
    } else if (step.stage === 'gelu') {
        document.getElementById('ffn-gelu').classList.add('active');
    } else if (step.stage === 'linear2') {
        document.getElementById('ffn-linear2').classList.add('active');
    }

    const stageNames = {
        'linear1': 'FFN: First Linear Layer',
        'gelu': 'FFN: GELU Activation',
        'linear2': 'FFN: Second Linear Layer'
    };

    setStepInfo(stageNames[step.stage] || `FFN: ${step.stage}`, `Layer ${step.layer}`);
    setCode(step.code);

    showValueCard('ffn-value-card', 'ffn-value', formatVector(step.output_sample));
    showTensorViz(`FFN ${step.stage}`, `Layer ${step.layer}`, step.output_sample);

    addLog('ffn', `Layer ${step.layer} ${step.stage}`);
}

function handleFinalLogits(step) {
    highlightArchBlock('arch-lm-head');
    highlightArchBlock('arch-output');

    // 모든 토큰 완료
    tokenChars.forEach((_, i) => updateTokenState(i, 'complete'));

    setStepInfo('Final Output', 'Computing logits and predictions');
    setCode(step.code);

    // 예측 표시
    document.getElementById('predictions-section').style.display = 'block';
    showPredictions(step.top_tokens);

    addLog('output', `Top: "${step.top_tokens[0]?.token_char}" (${(step.top_tokens[0]?.probability * 100).toFixed(1)}%)`);
}

function handleGeneratedToken(step) {
    setStepInfo('Token Generated', `"${step.token_char}" (prob: ${(step.probability * 100).toFixed(1)}%)`);
    setCode(`// Generated Token
// Token: "${step.token_char}" (ID: ${step.token_id})
// Probability: ${(step.probability * 100).toFixed(2)}%
//
// Text so far: "${step.generated_text}"`);

    // 생성된 텍스트 표시
    document.getElementById('generation-section').style.display = 'block';
    document.getElementById('generated-text').textContent = step.generated_text;

    // 토큰 시퀀스에 추가
    addGeneratedToken(step.token_char);

    addLog('output', `Generated: "${step.token_char}"`);
}

function handleComplete(step) {
    isRunning = false;
    clearAllHighlights();

    setStepInfo('Complete', `Total ${step.total_steps} steps`);
    setCode(`// Transformer Forward Pass Complete!
//
// Total steps: ${step.total_steps}
// Final output: "${step.final_text}"
//
// Ready for next input.`);

    setStatus('Complete');
    addLog('output', `Complete: ${step.total_steps} steps`);

    document.getElementById('btn-run').disabled = false;
    document.getElementById('btn-stop').disabled = true;
}

function handleError(step) {
    isRunning = false;
    setStepInfo('Error', step.message);
    setStatus('Error: ' + step.message, true);
    addLog('output', `Error: ${step.message}`);

    document.getElementById('btn-run').disabled = false;
    document.getElementById('btn-stop').disabled = true;
}

// ==================== UI Updates ====================
function setStepInfo(name, description) {
    document.getElementById('step-number').textContent = `Step ${currentStep}`;
    document.getElementById('step-name').textContent = name;
    document.getElementById('step-description').textContent = description;
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
    // 모든 블록 비활성화
    document.querySelectorAll('.arch-block').forEach(b => b.classList.remove('active'));
    // 해당 블록 활성화
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
        // 값에 따른 색상
        const intensity = Math.min(Math.abs(v) * 100, 100);
        const hue = v >= 0 ? 200 : 0; // 파랑 vs 빨강
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

    // 행렬 그리드 생성 (4x4 미리보기)
    renderSmallMatrix('matrix-a-grid', a.slice(0, 4), 1, 4);
    renderSmallMatrix('matrix-b-grid', b.slice(0, 4), 4, 1);
    document.getElementById('matrix-result-grid').innerHTML = '<div class="matrix-cell">...</div>';
}

function renderSmallMatrix(gridId, values, rows, cols) {
    const grid = document.getElementById(gridId);
    grid.innerHTML = '';
    grid.style.gridTemplateColumns = `repeat(${cols}, 24px)`;

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
    grid.style.gridTemplateColumns = `repeat(${size}, 24px)`;

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
            title: 'Weight',
            titlefont: { color: '#8b949e', size: 10 },
            tickfont: { color: '#8b949e', size: 9 }
        }
    }];

    const layout = {
        paper_bgcolor: '#21262d',
        plot_bgcolor: '#21262d',
        font: { color: '#e6edf3', size: 10 },
        xaxis: {
            title: 'Key',
            tickangle: -45,
            tickfont: { size: 9 }
        },
        yaxis: {
            title: 'Query',
            autorange: 'reversed',
            tickfont: { size: 9 }
        },
        margin: { l: 50, r: 40, t: 20, b: 50 }
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
function formatVector(arr) {
    return `[${arr.map(v => v.toFixed(4)).join(', ')}...]`;
}

function formatVectorShort(arr) {
    return arr.slice(0, 4).map(v => v.toFixed(3)).join(', ') + '...';
}

function resetUI() {
    currentStep = 0;
    totalSteps = 0;

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

    // Q, K, V 행렬 초기화
    document.getElementById('q-matrix').textContent = '';
    document.getElementById('k-matrix').textContent = '';
    document.getElementById('v-matrix').textContent = '';

    setStepInfo('Ready', 'Press "Run" to start the visualization');
    setCode(`// Transformer Forward Pass
// Click "Run" to start

// Architecture:
// 1. Token Embedding → lookup table
// 2. Position Encoding → sin/cos patterns
// 3. Transformer Block × N:
//    - Layer Norm
//    - Multi-Head Attention
//      • Q = x @ W_q
//      • K = x @ W_k
//      • V = x @ W_v
//      • Attention = softmax(Q @ K^T / √d_k) @ V
//    - Residual Connection
//    - Layer Norm
//    - Feed Forward Network
//      • Linear → GELU → Linear
//    - Residual Connection
// 4. LM Head → Logits
// 5. Softmax → Probabilities`);
}

// ==================== Event Listeners ====================
document.getElementById('btn-run').addEventListener('click', () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        setStatus('Not connected', true);
        return;
    }

    const text = document.getElementById('input-text').value.trim();
    if (!text) {
        setStatus('Please enter text', true);
        return;
    }

    const generateTokens = parseInt(document.getElementById('gen-count').value);

    resetUI();

    ws.send(JSON.stringify({
        text: text,
        generate_tokens: generateTokens
    }));
});

document.getElementById('btn-stop').addEventListener('click', () => {
    setStatus('Stop requested...');
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

// ==================== Init ====================
window.addEventListener('DOMContentLoaded', () => {
    connectWebSocket();
    resetUI();
    setStatus('Connecting...');
});

window.addEventListener('beforeunload', () => {
    if (ws) ws.close();
});
