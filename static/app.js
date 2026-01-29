// ==================== WebSocket 연결 ====================
let ws = null;
let isRunning = false;
let totalSteps = 0;
let currentStep = 0;
let tokens = [];
let tokenChars = [];

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
        // 자동 재연결
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
        dot.classList.remove('disconnected');
        text.textContent = 'Connected';
    } else {
        dot.classList.remove('connected');
        dot.classList.add('disconnected');
        text.textContent = 'Disconnected';
    }
}

// ==================== 실행 스텝 처리 ====================
function handleExecutionStep(step) {
    currentStep++;

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
        case 'LayerNorm':
            handleLayerNorm(step);
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

    updateProgress();
}

// ==================== 스텝 핸들러 ====================
function handleStart(step) {
    isRunning = true;
    currentStep = 0;
    tokens = step.tokens;
    tokenChars = tokens.map(t => t < 128 ? String.fromCharCode(t) : `[${t}]`);

    // 예상 스텝 수 계산 (대략적)
    const numLayers = 4; // mini config
    const numHeads = 4;
    totalSteps = tokens.length * 2 + numLayers * (numHeads * 4 + 3) + 5;

    setStage('Start', `Input: "${step.text}" (${tokens.length} tokens)`);
    setCode(`// Starting LLM execution
// Input text: "${step.text}"
// Tokens: [${tokens.join(', ')}]
// Token chars: [${tokenChars.map(t => `'${t}'`).join(', ')}]

// Forward pass begins...`);

    // 토큰 표시
    displayTokens(tokenChars);

    addLog('start', `Starting execution: "${step.text}"`);
    setStatus('Running...');

    document.getElementById('btn-run').disabled = true;
    document.getElementById('btn-stop').disabled = false;
}

function handleTokenEmbedding(step) {
    setStage('Token Embedding', `Token ${step.step}: "${step.token_char}" (ID: ${step.token_id})`);
    setCode(step.code);

    highlightToken(step.step, 'active');
    if (step.step > 0) highlightToken(step.step - 1, 'completed');

    showEmbeddingValues(step.embedding_sample);
    addLog('embedding', `Token ${step.step}: "${step.token_char}" -> [${formatFloats(step.embedding_sample)}...]`);
}

function handlePositionEncoding(step) {
    setStage('Position Encoding', `Position ${step.position}`);
    setCode(step.code);

    showEmbeddingValues(step.encoding_sample);
    addLog('embedding', `Position ${step.position}: [${formatFloats(step.encoding_sample)}...]`);
}

function handleAttentionStart(step) {
    setStage('Attention', `Layer ${step.layer}, Head ${step.head}`);
    setCode(step.code);

    document.getElementById('attention-section').style.display = 'block';
    document.getElementById('attention-info').textContent = `Layer ${step.layer}, Head ${step.head}`;

    addLog('attention', `Layer ${step.layer} Head ${step.head}: Computing attention...`);
}

function handleAttentionQKV(step) {
    setStage('Attention - Q/K/V', `Layer ${step.layer}, Head ${step.head}`);
    setCode(step.code);

    showQKVValues(step.q_sample, step.k_sample, step.v_sample);
    addLog('attention', `Q/K/V computed for head ${step.head}`);
}

function handleAttentionScores(step) {
    setStage('Attention - Scores', `Layer ${step.layer}, Head ${step.head}`);
    setCode(step.code);

    addLog('attention', `Attention scores computed`);
}

function handleAttentionWeights(step) {
    setStage('Attention - Softmax', `Layer ${step.layer}, Head ${step.head}`);
    setCode(step.code);

    renderAttentionHeatmap(step.weights, step.tokens, step.layer, step.head);
    addLog('attention', `Attention weights: softmax applied`);
}

function handleAttentionOutput(step) {
    setStage('Attention - Output', `Layer ${step.layer}`);
    setCode(step.code);

    showOutputValues(step.output_sample);
    addLog('attention', `Attention output + residual connection`);
}

function handleFeedForward(step) {
    const stageName = {
        'linear1': 'FFN - Linear 1',
        'gelu': 'FFN - GELU',
        'linear2': 'FFN - Linear 2'
    }[step.stage] || `FFN - ${step.stage}`;

    setStage(stageName, `Layer ${step.layer}`);
    setCode(step.code);

    showOutputValues(step.output_sample);
    addLog('ffn', `Layer ${step.layer} ${step.stage}`);
}

function handleLayerNorm(step) {
    setStage('Layer Norm', `Layer ${step.layer} - ${step.location}`);
    setCode(step.code);

    addLog('token', `LayerNorm: mean=${step.mean.toFixed(4)}, std=${step.std.toFixed(4)}`);
}

function handleFinalLogits(step) {
    setStage('Final Output', 'Computing logits and predictions');
    setCode(step.code);

    // 모든 토큰 완료 표시
    tokenChars.forEach((_, i) => highlightToken(i, 'completed'));

    // 예측 표시
    showPredictions(step.top_tokens);

    addLog('output', `Top prediction: "${step.top_tokens[0]?.token_char}" (${(step.top_tokens[0]?.probability * 100).toFixed(1)}%)`);
}

function handleGeneratedToken(step) {
    setStage('Token Generation', `Generated: "${step.token_char}"`);
    setCode(`// Token Generation Step
// Selected token: "${step.token_char}" (ID: ${step.token_id})
// Probability: ${(step.probability * 100).toFixed(2)}%
//
// Generated text so far:
// "${step.generated_text}"`);

    // 생성된 텍스트 표시
    document.getElementById('generated-section').style.display = 'block';
    document.getElementById('generated-text').textContent = step.generated_text;

    // 토큰 리스트에 추가
    addTokenToDisplay(step.token_char, true);

    addLog('output', `Generated: "${step.token_char}" (${(step.probability * 100).toFixed(1)}%)`);
}

function handleComplete(step) {
    isRunning = false;
    setStage('Complete', `Total ${step.total_steps} steps`);
    setCode(`// Execution Complete!
//
// Total steps: ${step.total_steps}
// Final output: "${step.final_text}"
//
// The LLM has finished processing.
// Try a different input to see more!`);

    setStatus('Complete');
    addLog('output', `Execution complete: ${step.total_steps} steps`);

    document.getElementById('btn-run').disabled = false;
    document.getElementById('btn-stop').disabled = true;
}

function handleError(step) {
    isRunning = false;
    setStage('Error', step.message);
    setCode(`// Error occurred:
// ${step.message}`);

    setStatus('Error: ' + step.message, true);
    addLog('output', `Error: ${step.message}`);

    document.getElementById('btn-run').disabled = false;
    document.getElementById('btn-stop').disabled = true;
}

// ==================== UI 업데이트 함수 ====================
function setStage(name, detail) {
    document.getElementById('stage-name').textContent = name;
    document.getElementById('stage-detail').textContent = detail;
    document.getElementById('step-indicator').textContent = `Step ${currentStep}`;
}

function setCode(code) {
    document.getElementById('code-block').querySelector('code').textContent = code;
}

function setStatus(text, isError = false) {
    const statusEl = document.getElementById('status-text');
    statusEl.textContent = text;
    statusEl.style.color = isError ? 'var(--accent-red)' : 'var(--accent-green)';

    document.getElementById('status-time').textContent = new Date().toLocaleTimeString();
}

function updateProgress() {
    const percent = totalSteps > 0 ? Math.min((currentStep / totalSteps) * 100, 100) : 0;
    document.getElementById('progress-fill').style.width = `${percent}%`;
    document.getElementById('progress-text').textContent = `${Math.round(percent)}%`;
}

function displayTokens(chars) {
    const container = document.getElementById('token-list');
    container.innerHTML = '';

    chars.forEach((char, i) => {
        const el = document.createElement('span');
        el.className = 'token-item';
        el.textContent = char;
        el.dataset.index = i;
        container.appendChild(el);
    });
}

function highlightToken(index, state) {
    const container = document.getElementById('token-list');
    const token = container.querySelector(`[data-index="${index}"]`);
    if (token) {
        token.classList.remove('active', 'completed');
        if (state) token.classList.add(state);
    }
}

function addTokenToDisplay(char, isGenerated = false) {
    const container = document.getElementById('token-list');
    const el = document.createElement('span');
    el.className = 'token-item';
    if (isGenerated) el.classList.add('active');
    el.textContent = char;
    el.dataset.index = container.children.length;
    container.appendChild(el);

    // 이전 생성 토큰 완료 표시
    const prevGenerated = container.querySelectorAll('.token-item.active');
    prevGenerated.forEach((t, i) => {
        if (i < prevGenerated.length - 1) {
            t.classList.remove('active');
            t.classList.add('completed');
        }
    });
}

function showEmbeddingValues(values) {
    document.getElementById('embedding-card').style.display = 'block';
    document.getElementById('embedding-values').textContent = `[${formatFloats(values)}...]`;
}

function showQKVValues(q, k, v) {
    document.getElementById('qkv-card').style.display = 'block';
    document.getElementById('qkv-values').textContent =
        `Q: [${formatFloats(q.slice(0, 4))}...]\n` +
        `K: [${formatFloats(k.slice(0, 4))}...]\n` +
        `V: [${formatFloats(v.slice(0, 4))}...]`;
}

function showOutputValues(values) {
    document.getElementById('output-card').style.display = 'block';
    document.getElementById('output-values').textContent = `[${formatFloats(values)}...]`;
}

function showPredictions(topTokens) {
    document.getElementById('predictions-section').style.display = 'block';
    const container = document.getElementById('predictions-list');
    container.innerHTML = '';

    topTokens.forEach((pred, i) => {
        const item = document.createElement('div');
        item.className = 'prediction-item';

        const percent = (pred.probability * 100).toFixed(1);
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

function renderAttentionHeatmap(weights, tokens, layer, head) {
    const data = [{
        z: weights,
        x: tokens,
        y: tokens,
        type: 'heatmap',
        colorscale: [
            [0, '#1a1b26'],
            [0.25, '#414868'],
            [0.5, '#7aa2f7'],
            [0.75, '#bb9af7'],
            [1, '#ff9e64']
        ],
        showscale: true,
        colorbar: { title: 'Weight' }
    }];

    const layout = {
        title: {
            text: `Attention Layer ${layer} Head ${head}`,
            font: { color: '#c0caf5', size: 14 }
        },
        paper_bgcolor: '#24283b',
        plot_bgcolor: '#24283b',
        font: { color: '#c0caf5' },
        xaxis: { title: 'Key', tickangle: -45 },
        yaxis: { title: 'Query', autorange: 'reversed' },
        margin: { l: 60, r: 60, t: 40, b: 60 }
    };

    Plotly.newPlot('attention-chart', data, layout, { responsive: true });
}

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

function formatFloats(arr) {
    return arr.map(v => v.toFixed(4)).join(', ');
}

function resetUI() {
    currentStep = 0;
    totalSteps = 0;

    document.getElementById('progress-fill').style.width = '0%';
    document.getElementById('progress-text').textContent = '0%';
    document.getElementById('token-list').innerHTML = '';
    document.getElementById('generated-section').style.display = 'none';
    document.getElementById('attention-section').style.display = 'none';
    document.getElementById('predictions-section').style.display = 'none';
    document.getElementById('embedding-card').style.display = 'none';
    document.getElementById('qkv-card').style.display = 'none';
    document.getElementById('output-card').style.display = 'none';

    setStage('Ready', 'Click Run to start');
    setCode(`// LLM execution process will be shown here
// Click "Run" button to start

// Forward Pass steps:
// 1. Token Embedding
// 2. Position Encoding
// 3. Transformer Blocks (x N)
//    - Layer Norm
//    - Multi-Head Attention
//    - Feed Forward Network
// 4. Final Layer Norm
// 5. LM Head -> Logits`);
}

// ==================== 이벤트 리스너 ====================
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
    // 현재는 중지 기능 미구현 (서버에서 취소 로직 필요)
    setStatus('Stop requested...');
});

document.getElementById('btn-clear-log').addEventListener('click', () => {
    document.getElementById('log-container').innerHTML = '';
});

// 슬라이더 값 표시
document.getElementById('gen-count').addEventListener('input', (e) => {
    document.getElementById('gen-count-value').textContent = e.target.value;
});

document.getElementById('speed').addEventListener('input', (e) => {
    const value = e.target.value;
    document.getElementById('speed-value').textContent = value;

    // 속도 변경을 서버에 전송
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ delay_ms: parseInt(value) }));
    }
});

// ==================== 초기화 ====================
window.addEventListener('DOMContentLoaded', () => {
    connectWebSocket();
    resetUI();
    setStatus('Connecting...');
});

// 페이지 종료 시 WebSocket 정리
window.addEventListener('beforeunload', () => {
    if (ws) {
        ws.close();
    }
});
