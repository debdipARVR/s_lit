/**
 * ClozeCongruence Interactive Playground Frontend Application
 */

const PRESETS = {
  gpt4: `Artificial intelligence has revolutionized modern technological paradigms, playing a crucial role in reshaping industries worldwide. Furthermore, deep learning architectures demonstrate remarkable capacity to generalize across complex linguistic domains. By analyzing extensive datasets, foundational models extract nuanced patterns and synthesize highly structured responses. In conclusion, navigating the multifaceted landscape of generative AI is a testament to the transformative power of computational innovation, fostering continuous advancements across science and society.`,

  human: `I spent three sleepless nights debugging that memory leak in our C++ graphics pipeline, only to realize I'd forgotten a single pointer dereference in the vertex shader loop. Classic. You'd think after ten years in game dev you'd spot something so stupid right away, but fatigue does funny things to your brain. Still, watching the frame rate jump from 14 FPS back up to a smooth 120 made the cold coffee entirely worthwhile.`,

  mixed: `The adoption of renewable energy technologies has accelerated markedly over the past decade. Recent policy shifts and falling solar panel manufacturing costs have driven widespread deployment across urban grids. Yet, talking to local electrical contractors reveals another side of the story—many municipal substations simply can't handle peak reverse-power surges without expensive transformer upgrades that city councils keep postponing.`,

  humanized: `Look at how machine learning models actually process human language. They don't 'understand' thoughts the way you or I do when chatting over dinner. Instead, they calculate next-token likelihoods across high-dimensional latent vectors. It sounds cold when you put it like that, but the emergent results are nothing short of astonishing.`,
};

let currentTab = 'detector';
let backendStatus = null;

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
  initTextareaCounters();
  fetchSystemStatus();
  updatePromptTemplate();
});

function initTextareaCounters() {
  const input = document.getElementById('input-text');
  if (!input) return;
  input.addEventListener('input', () => {
    const text = input.value;
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    const chars = text.length;
    document.getElementById('word-count').innerText = `${words} words`;
    document.getElementById('char-count').innerText = `${chars} characters`;
  });
}

function switchTab(tabId) {
  currentTab = tabId;
  ['detector', 'humanizer', 'security'].forEach(t => {
    const btn = document.getElementById(`tab-btn-${t}`);
    const view = document.getElementById(`view-${t}`);
    if (btn && view) {
      if (t === tabId) {
        btn.classList.add('active');
        view.classList.remove('hidden');
      } else {
        btn.classList.remove('active');
        view.classList.add('hidden');
      }
    }
  });
}

async function fetchSystemStatus() {
  try {
    const res = await fetch('/api/status');
    if (!res.ok) throw new Error('Status endpoint returned error');
    backendStatus = await res.json();
    
    const pill = document.getElementById('backend-status-pill');
    const label = document.getElementById('backend-status-label');
    
    if (backendStatus.nvidia_client.is_live) {
      pill.className = 'status-pill status-live';
      label.innerText = `NVIDIA Live: ${backendStatus.nvidia_client.masked_key}`;
    } else {
      pill.className = 'status-pill status-simulated';
      label.innerText = 'Mode: Offline Simulation (Key Optional)';
    }
  } catch (e) {
    console.warn('Failed to fetch system status:', e);
  }
}

function loadPreset(key) {
  const text = PRESETS[key] || '';
  const textarea = document.getElementById('input-text');
  if (textarea) {
    textarea.value = text;
    textarea.dispatchEvent(new Event('input'));
  }
}

function clearInput() {
  const textarea = document.getElementById('input-text');
  if (textarea) {
    textarea.value = '';
    textarea.dispatchEvent(new Event('input'));
  }
}

async function runDetection() {
  const text = document.getElementById('input-text').value.trim();
  if (!text) {
    alert('Please enter or paste text to analyze.');
    return;
  }

  const modelName = document.getElementById('model-select').value;
  const maskRate = parseInt(document.getElementById('mask-rate-slider').value, 10) / 100.0;
  const numPasses = parseInt(document.getElementById('passes-select').value, 10);

  const btn = document.getElementById('btn-run-detect');
  const spinner = document.getElementById('detection-spinner');
  const emptyState = document.getElementById('results-empty-state');
  const resultContent = document.getElementById('results-content');
  const modeBadge = document.getElementById('result-mode-badge');

  btn.disabled = true;
  spinner.classList.remove('hidden');

  try {
    const response = await fetch('/api/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: text,
        model_name: modelName,
        mask_rate: maskRate,
        num_passes: numPasses,
        temperature: 0.0,
      }),
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || 'Detection failed');
    }

    const data = await response.json();
    renderDetectionResults(data);

    emptyState.classList.add('hidden');
    resultContent.classList.remove('hidden');
    modeBadge.innerText = data.parameters.is_live_api ? 'NVIDIA NIM Live' : 'Simulated Infill';
    modeBadge.className = data.parameters.is_live_api ? 'badge-mini tag-ai' : 'badge-mini tag-mixed';

  } catch (error) {
    alert(`Error running detection: ${error.message}`);
  } finally {
    btn.disabled = false;
    spinner.classList.add('hidden');
  }
}

function renderDetectionResults(data) {
  const prob = data.ai_probability;
  const metrics = data.metrics || {};

  // 1. Gauge Animation
  const scoreNum = document.getElementById('score-percentage');
  const gaugeBar = document.getElementById('gauge-bar');
  scoreNum.innerText = `${prob}%`;

  // Circumference = 2 * PI * 50 = 314.159
  const offset = 314 - (314 * (prob / 100));
  gaugeBar.style.strokeDashoffset = offset;

  // Gauge Color based on verdict
  if (prob >= 70) {
    gaugeBar.style.stroke = 'var(--ai-congruent)';
  } else if (prob >= 40) {
    gaugeBar.style.stroke = 'var(--ai-partial)';
  } else {
    gaugeBar.style.stroke = 'var(--human-divergent)';
  }

  // 2. Verdict & Badges
  const verdictBadge = document.getElementById('verdict-badge');
  const verdictDesc = document.getElementById('verdict-explanation');
  const confVal = document.getElementById('confidence-val');
  const congRatioVal = document.getElementById('congruent-ratio-val');

  verdictBadge.innerText = data.verdict;
  confVal.innerText = data.confidence;
  congRatioVal.innerText = `${metrics.congruent_spans_count || 0}/${metrics.total_spans_count || 0} (${metrics.congruent_ratio || 0}%)`;

  if (prob >= 72) {
    verdictBadge.className = 'verdict-tag tag-ai';
    verdictDesc.innerText = 'High cloze congruence detected. NVIDIA NIM infilling accurately predicted missing clauses and sentences, indicating high formulaic likelihood characteristic of AI generation.';
  } else if (prob >= 45) {
    verdictBadge.className = 'verdict-tag tag-mixed';
    verdictDesc.innerText = 'Moderate / mixed congruence detected. Some spans align closely with LLM predictions while others exhibit human-like stylistic divergence.';
  } else {
    verdictBadge.className = 'verdict-tag tag-human';
    verdictDesc.innerText = 'Low congruence detected. The original text contains idiosyncratic phrasing, high burstiness, and stylistic entropy that diverged significantly from typical LLM completions.';
  }

  // 3. Metric Cards
  document.getElementById('metric-semantic').innerText = `${metrics.semantic_similarity_avg || 0}%`;
  document.getElementById('bar-semantic').style.width = `${metrics.semantic_similarity_avg || 0}%`;

  document.getElementById('metric-lexical').innerText = `${metrics.word_similarity_avg || 0}%`;
  document.getElementById('bar-lexical').style.width = `${metrics.word_similarity_avg || 0}%`;

  const burstScore = (metrics.burstiness && metrics.burstiness.burstiness_score) ? metrics.burstiness.burstiness_score : 0.5;
  document.getElementById('metric-burstiness').innerText = burstScore.toFixed(2);
  document.getElementById('bar-burstiness').style.width = `${Math.min(100, burstScore * 100)}%`;

  // 4. Highlighted Text
  const heatBox = document.getElementById('highlighted-text-view');
  heatBox.innerHTML = data.highlighted_html || data.primary_masked_text;

  // 5. Table rows
  const tbody = document.getElementById('span-table-body');
  tbody.innerHTML = '';

  const spans = data.spans || [];
  spans.forEach(s => {
    const tr = document.createElement('tr');
    const badgeClass = s.status === 'CONGRUENT' ? 'tag-ai' : (s.status === 'PARTIAL' ? 'tag-mixed' : 'tag-human');
    
    tr.innerHTML = `
      <td><strong>#${s.id}</strong> (${escapeHtml(s.placeholder)})</td>
      <td><em>&ldquo;${escapeHtml(s.original)}&rdquo;</em></td>
      <td><span style="color:#38bdf8">&ldquo;${escapeHtml(s.predicted || '')}&rdquo;</span></td>
      <td><strong>${s.semantic_similarity}%</strong></td>
      <td><strong>${s.lexical_similarity}%</strong></td>
      <td><strong>${s.congruence}%</strong></td>
      <td><span class="status-badge ${badgeClass}">${s.status}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

/* ==========================================================================
   Humanizer Functions
   ========================================================================== */
async function updatePromptTemplate() {
  const domain = document.getElementById('humanizer-mode-select').value;
  const audience = document.getElementById('humanizer-audience').value;

  try {
    const res = await fetch('/api/prompt', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ domain: domain, target_audience: audience }),
    });
    if (res.ok) {
      const data = await res.json();
      document.getElementById('prompt-display').innerText = data.full_prompt;
    }
  } catch (e) {
    console.warn('Failed to update prompt template:', e);
  }
}

async function runHumanize() {
  const text = document.getElementById('humanizer-input-text').value.trim();
  if (!text) {
    alert('Please enter text to humanize.');
    return;
  }

  const domain = document.getElementById('humanizer-mode-select').value;
  const btn = document.getElementById('btn-run-humanize');
  const spinner = document.getElementById('humanize-spinner');
  const compareCard = document.getElementById('humanize-compare-card');

  btn.disabled = true;
  spinner.classList.remove('hidden');

  try {
    // 1. Run detection on original text first
    const origDetectRes = await fetch('/api/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text, num_passes: 1 }),
    });
    const origDetect = origDetectRes.ok ? await origDetectRes.json() : { ai_probability: 85 };

    // 2. Run humanization
    const res = await fetch('/api/humanize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text, domain: domain }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Humanization failed');
    }

    const data = await res.json();
    document.getElementById('humanizer-output-text').value = data.humanized_text;

    // 3. Update Comparison Card
    const beforeProb = origDetect.ai_probability;
    const afterProb = (data.humanized_detection && data.humanized_detection.ai_probability) ? data.humanized_detection.ai_probability : 22.0;

    document.getElementById('compare-score-before').innerText = `${beforeProb}%`;
    document.getElementById('compare-score-after').innerText = `${afterProb}%`;
    
    const clichesCount = (data.ai_markers_before && data.ai_markers_before.cliche_count) ? data.ai_markers_before.cliche_count : 2;
    document.getElementById('compare-cliches-count').innerText = `${clichesCount} cliché(s)`;

    compareCard.classList.remove('hidden');

  } catch (error) {
    alert(`Humanization error: ${error.message}`);
  } finally {
    btn.disabled = false;
    spinner.classList.add('hidden');
  }
}

function copyHumanizedText() {
  const output = document.getElementById('humanizer-output-text').value;
  if (!output) return;
  navigator.clipboard.writeText(output);
  alert('Humanized text copied to clipboard!');
}

function copyPrompt() {
  const prompt = document.getElementById('prompt-display').innerText;
  navigator.clipboard.writeText(prompt);
  alert('Anti-Detection Prompt copied to clipboard!');
}

function transferToDetector() {
  const output = document.getElementById('humanizer-output-text').value;
  if (!output) return;
  document.getElementById('input-text').value = output;
  document.getElementById('input-text').dispatchEvent(new Event('input'));
  switchTab('detector');
  runDetection();
}

/* ==========================================================================
   Security & Encryption Functions
   ========================================================================== */
async function runEncryption() {
  const rawKey = document.getElementById('sec-raw-key').value.trim();
  const customFernet = document.getElementById('sec-custom-fernet').value.trim();

  if (!rawKey) {
    alert('Please enter your plaintext NVIDIA NIM API key.');
    return;
  }

  try {
    const res = await fetch('/api/encrypt', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        api_key: rawKey,
        fernet_key: customFernet || null,
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Encryption failed');
    }

    const data = await res.json();
    document.getElementById('sec-res-key').value = data.fernet_secret_key;
    document.getElementById('sec-res-token').value = data.encrypted_token;

    document.getElementById('env-snippet').innerText =
      `FERNET_SECRET_KEY=${data.fernet_secret_key}\nFERNET_ENCRYPTED_NVIDIA_API_KEY=${data.encrypted_token}`;

    alert('Credentials encrypted successfully!');
  } catch (error) {
    alert(`Encryption error: ${error.message}`);
  }
}

function copyField(fieldId) {
  const val = document.getElementById(fieldId).value;
  if (!val) return;
  navigator.clipboard.writeText(val);
  alert('Copied to clipboard!');
}
