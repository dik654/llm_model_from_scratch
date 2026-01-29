// ==================== API 호출 ====================
async function apiCall(endpoint, data) {
    const response = await fetch(`/api/${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
    if (!response.ok) {
        throw new Error(await response.text());
    }
    return response.json();
}

// ==================== 상태 관리 ====================
let modelCreated = false;
let currentAttentionData = null;

// ==================== 유틸리티 함수 ====================
function setStatus(text, isError = false) {
    const statusEl = document.getElementById('status-text');
    statusEl.textContent = text;
    statusEl.style.color = isError ? 'var(--accent-red)' : 'var(--accent-green)';

    const timeEl = document.getElementById('status-time');
    timeEl.textContent = new Date().toLocaleTimeString();
}

function showInfo(elementId, data) {
    const el = document.getElementById(elementId);
    if (typeof data === 'object') {
        el.textContent = JSON.stringify(data, null, 2);
    } else {
        el.textContent = data;
    }
}

// ==================== 탭 네비게이션 ====================
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        const tabId = btn.dataset.tab;
        document.getElementById(`tab-${tabId}`).classList.add('active');
    });
});

// ==================== 슬라이더 값 표시 ====================
document.getElementById('gen-length').addEventListener('input', (e) => {
    document.getElementById('gen-length-value').textContent = e.target.value;
});

document.getElementById('temperature').addEventListener('input', (e) => {
    document.getElementById('temp-value').textContent = e.target.value;
});

document.getElementById('train-steps').addEventListener('input', (e) => {
    document.getElementById('train-steps-value').textContent = e.target.value;
});

// ==================== 모델 생성 ====================
document.getElementById('btn-create-model').addEventListener('click', async () => {
    const modelSize = document.getElementById('model-size').value;
    setStatus('모델 생성 중...');

    try {
        const info = await apiCall('model', { config_type: modelSize });
        modelCreated = true;

        showInfo('model-info',
            `✓ 모델 생성 완료\n` +
            `Vocab Size: ${info.vocab_size}\n` +
            `Max Seq Len: ${info.max_seq_len}\n` +
            `d_model: ${info.d_model}\n` +
            `Heads: ${info.num_heads}\n` +
            `Layers: ${info.num_layers}\n` +
            `Parameters: ${info.total_params.toLocaleString()}\n` +
            `Memory: ${info.memory_mb.toFixed(2)} MB`
        );

        setStatus('모델 생성 완료');
    } catch (error) {
        setStatus('모델 생성 실패: ' + error.message, true);
    }
});

// ==================== 토큰화 ====================
document.getElementById('btn-tokenize').addEventListener('click', async () => {
    const text = document.getElementById('input-text').value;

    try {
        const result = await apiCall('tokenize', { text });
        showInfo('token-info',
            `입력: "${text}"\n` +
            `토큰 수: ${result.tokens.length}\n` +
            `토큰 ID: [${result.tokens.join(', ')}]\n` +
            `토큰 문자: [${result.token_strs.join(', ')}]`
        );

        updateAttention(text);
        setStatus('토큰화 완료');
    } catch (error) {
        setStatus('토큰화 실패: ' + error.message, true);
    }
});

// ==================== 순전파 ====================
document.getElementById('btn-forward').addEventListener('click', async () => {
    if (!modelCreated) {
        setStatus('먼저 모델을 생성하세요', true);
        return;
    }

    const text = document.getElementById('input-text').value;
    setStatus('순전파 실행 중...');

    try {
        const result = await apiCall('forward', { text });

        let output = `입력: [${result.input_tokens.join(', ')}]\n`;
        output += `출력 Shape: [${result.output_shape.join(', ')}]\n\n`;
        output += `예측된 다음 토큰: "${result.predicted_next}"\n\n`;
        output += `Top-5 예측:\n`;
        result.top_predictions.forEach((p, i) => {
            output += `  ${i + 1}. "${p.token}" (${(p.probability * 100).toFixed(1)}%)\n`;
        });

        showInfo('token-info', output);
        updateAttention(text);
        setStatus('순전파 완료');
    } catch (error) {
        setStatus('순전파 실패: ' + error.message, true);
    }
});

// ==================== 텍스트 생성 ====================
document.getElementById('btn-generate').addEventListener('click', async () => {
    if (!modelCreated) {
        setStatus('먼저 모델을 생성하세요', true);
        return;
    }

    const prompt = document.getElementById('gen-prompt').value;
    const maxTokens = parseInt(document.getElementById('gen-length').value);
    const temperature = parseFloat(document.getElementById('temperature').value);

    setStatus('텍스트 생성 중...');

    try {
        const result = await apiCall('generate', {
            prompt,
            max_tokens: maxTokens,
            temperature
        });

        let output = `생성된 텍스트:\n"${result.text}"\n\n`;
        output += `토큰 수: ${result.tokens.length}`;

        showInfo('generation-result', output);
        setStatus('텍스트 생성 완료');
    } catch (error) {
        setStatus('텍스트 생성 실패: ' + error.message, true);
    }
});

// ==================== Attention 시각화 ====================
async function updateAttention(text) {
    try {
        currentAttentionData = await apiCall('attention', { text });
        renderAttentionHeatmap(0);
    } catch (error) {
        console.error('Attention 가져오기 실패:', error);
    }
}

function renderAttentionHeatmap(headIdx) {
    if (!currentAttentionData) return;

    const head = currentAttentionData.heads[headIdx];
    const tokens = currentAttentionData.tokens;

    const data = [{
        z: head.weights,
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
        colorbar: {
            title: 'Weight',
            titleside: 'right'
        }
    }];

    const layout = {
        title: {
            text: `Attention Head ${headIdx}`,
            font: { color: '#c0caf5' }
        },
        paper_bgcolor: '#24283b',
        plot_bgcolor: '#24283b',
        font: { color: '#c0caf5' },
        xaxis: {
            title: 'Key (attended to)',
            tickangle: -45
        },
        yaxis: {
            title: 'Query (attending)',
            autorange: 'reversed'
        },
        margin: { l: 80, r: 80, t: 60, b: 80 }
    };

    Plotly.newPlot('attention-heatmap', data, layout, { responsive: true });
}

document.getElementById('attention-head').addEventListener('change', (e) => {
    renderAttentionHeatmap(parseInt(e.target.value));
});

// ==================== 학습 시각화 ====================
document.getElementById('btn-train').addEventListener('click', async () => {
    const numSteps = parseInt(document.getElementById('train-steps').value);
    setStatus('학습 시뮬레이션 중...');

    try {
        const history = await apiCall('training', { num_steps: numSteps });
        renderTrainingCharts(history);
        setStatus('학습 시뮬레이션 완료');
        document.querySelector('[data-tab="training"]').click();
    } catch (error) {
        setStatus('학습 시뮬레이션 실패: ' + error.message, true);
    }
});

function renderTrainingCharts(history) {
    const steps = history.map(h => h.step);
    const losses = history.map(h => h.loss);
    const ppls = history.map(h => h.perplexity);
    const lrs = history.map(h => h.learning_rate);

    const commonLayout = {
        paper_bgcolor: '#24283b',
        plot_bgcolor: '#1a1b26',
        font: { color: '#c0caf5' },
        margin: { l: 50, r: 30, t: 40, b: 40 }
    };

    Plotly.newPlot('loss-chart', [{
        x: steps,
        y: losses,
        type: 'scatter',
        mode: 'lines',
        line: { color: '#f7768e', width: 2 },
        name: 'Loss'
    }], {
        ...commonLayout,
        title: { text: 'Loss', font: { color: '#c0caf5', size: 14 } },
        xaxis: { title: 'Step', gridcolor: '#414868' },
        yaxis: { title: 'Loss', gridcolor: '#414868' }
    }, { responsive: true });

    Plotly.newPlot('lr-chart', [{
        x: steps,
        y: lrs,
        type: 'scatter',
        mode: 'lines',
        line: { color: '#9ece6a', width: 2 },
        name: 'Learning Rate'
    }], {
        ...commonLayout,
        title: { text: 'Learning Rate', font: { color: '#c0caf5', size: 14 } },
        xaxis: { title: 'Step', gridcolor: '#414868' },
        yaxis: { title: 'LR', gridcolor: '#414868' }
    }, { responsive: true });

    Plotly.newPlot('perplexity-chart', [{
        x: steps,
        y: ppls,
        type: 'scatter',
        mode: 'lines',
        line: { color: '#bb9af7', width: 2 },
        fill: 'tozeroy',
        fillcolor: 'rgba(187, 154, 247, 0.1)',
        name: 'Perplexity'
    }], {
        ...commonLayout,
        title: { text: 'Perplexity', font: { color: '#c0caf5', size: 14 } },
        xaxis: { title: 'Step', gridcolor: '#414868' },
        yaxis: { title: 'Perplexity', gridcolor: '#414868', type: 'log' }
    }, { responsive: true });
}

// ==================== Positional Encoding 시각화 ====================
document.getElementById('btn-pos-enc').addEventListener('click', async () => {
    const maxLen = parseInt(document.getElementById('pos-length').value);
    const dModel = 64;

    setStatus('Positional Encoding 계산 중...');

    try {
        const posEnc = await apiCall('pos-encoding', { max_len: maxLen, d_model: dModel });
        renderPositionalEncoding(posEnc);
        setStatus('Positional Encoding 완료');
    } catch (error) {
        setStatus('Positional Encoding 실패: ' + error.message, true);
    }
});

function renderPositionalEncoding(posEnc) {
    const data = [{
        z: posEnc,
        type: 'heatmap',
        colorscale: [
            [0, '#1a1b26'],
            [0.25, '#414868'],
            [0.5, '#7dcfff'],
            [0.75, '#9ece6a'],
            [1, '#ff9e64']
        ],
        showscale: true,
        colorbar: {
            title: 'Value',
            titleside: 'right'
        }
    }];

    const layout = {
        title: {
            text: 'Sinusoidal Positional Encoding',
            font: { color: '#c0caf5' }
        },
        paper_bgcolor: '#24283b',
        plot_bgcolor: '#24283b',
        font: { color: '#c0caf5' },
        xaxis: { title: 'Dimension' },
        yaxis: { title: 'Position', autorange: 'reversed' },
        margin: { l: 60, r: 80, t: 60, b: 60 }
    };

    Plotly.newPlot('pos-encoding-chart', data, layout, { responsive: true });
}

// ==================== 초기화 ====================
window.addEventListener('DOMContentLoaded', async () => {
    setStatus('Ready');

    // 초기 Positional Encoding 표시
    try {
        const posEnc = await apiCall('pos-encoding', { max_len: 50, d_model: 64 });
        renderPositionalEncoding(posEnc);
    } catch (error) {
        console.log('초기 Positional Encoding 로드 실패');
    }

    // 초기 Attention 표시
    try {
        const attData = await apiCall('attention', { text: 'Hello' });
        currentAttentionData = attData;
        renderAttentionHeatmap(0);
    } catch (error) {
        console.log('초기 Attention 로드 실패');
    }
});
