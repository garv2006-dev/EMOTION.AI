/**
 * Emotion AI - Frontend Application Logic & Responsive Interactivity
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const textInput = document.getElementById('textInput');
    const charCounter = document.getElementById('charCounter');
    const clearBtn = document.getElementById('clearBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const samplesContainer = document.getElementById('samplesContainer');
    const ambientGlow = document.getElementById('ambientGlow');
    
    // Status Pill
    const statusPill = document.getElementById('statusPill');
    const statusText = document.getElementById('statusText');

    // States
    const emptyState = document.getElementById('emptyState');
    const loadingState = document.getElementById('loadingState');
    const loadingMsg = document.getElementById('loadingMsg');
    const errorState = document.getElementById('errorState');
    const errorTitle = document.getElementById('errorTitle');
    const errorMessage = document.getElementById('errorMessage');
    const retryBtn = document.getElementById('retryBtn');
    const resultsContent = document.getElementById('resultsContent');
    const resultsCard = document.getElementById('resultsCard');

    // Hero Top Display
    const heroEmoji = document.getElementById('heroEmoji');
    const heroEmotionName = document.getElementById('heroEmotionName');
    const heroEmotionDesc = document.getElementById('heroEmotionDesc');
    const heroConfidenceVal = document.getElementById('heroConfidenceVal');
    const topEmotionBanner = document.getElementById('topEmotionBanner');
    
    // Bars Container
    const barsContainer = document.getElementById('barsContainer');

    // Preprocessing Token Insights Accordion
    const preprocHeader = document.getElementById('preprocHeader');
    const preprocBody = document.getElementById('preprocBody');
    const preprocToggleIcon = document.getElementById('preprocToggleIcon');
    const tokenOriginalCount = document.getElementById('tokenOriginalCount');
    const tokenOriginalText = document.getElementById('tokenOriginalText');
    const tokenCleanedText = document.getElementById('tokenCleanedText');

    // History Log
    const historyTableBody = document.getElementById('historyTableBody');
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');
    let predictionHistory = JSON.parse(localStorage.getItem('emotion_ai_history') || '[]');

    // Initial Setup
    checkBackendHealth();
    loadSamplePrompts();

    // 1. Backend Health Monitoring
    statusPill.addEventListener('click', checkBackendHealth);

    async function checkBackendHealth() {
        updateStatusPill('connecting', 'Connecting...');
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 8000);

            const res = await fetch('/api/health', { signal: controller.signal });
            clearTimeout(timeoutId);

            if (res.ok) {
                const data = await res.json();
                if (data.model_loaded) {
                    const isMobile = window.innerWidth < 480;
                    const label = data.model_source === 'pickle' 
                        ? (isMobile ? 'Connected' : 'Model Connected') 
                        : (isMobile ? 'Fallback' : 'Model (Fallback)');
                    updateStatusPill('connected', label);
                } else {
                    updateStatusPill('offline', 'Model Error');
                }
            } else {
                updateStatusPill('offline', 'Degraded');
            }
        } catch (err) {
            if (err.name === 'AbortError') {
                updateStatusPill('waking', 'Waking Server...');
                setTimeout(async () => {
                    try {
                        const res2 = await fetch('/api/health');
                        if (res2.ok) updateStatusPill('connected', window.innerWidth < 480 ? 'Connected' : 'Model Connected');
                        else updateStatusPill('offline', 'Offline');
                    } catch (e) {
                        updateStatusPill('offline', 'Offline');
                    }
                }, 4000);
            } else {
                updateStatusPill('offline', 'Offline');
            }
        }
    }

    function updateStatusPill(state, text) {
        statusPill.className = `status-pill ${state}`;
        statusText.textContent = text;
    }

    // 2. Character Counter
    textInput.addEventListener('input', () => {
        const len = textInput.value.length;
        charCounter.textContent = `${len} / 500`;
    });

    // 3. Clear Button
    clearBtn.addEventListener('click', () => {
        textInput.value = '';
        charCounter.textContent = '0 / 500';
        textInput.focus();
    });

    // 4. Retry Button
    retryBtn.addEventListener('click', () => {
        performAnalysis();
    });

    // 5. Preprocessing Drawer Accordion Toggle
    if (preprocHeader) {
        preprocHeader.addEventListener('click', () => {
            const isHidden = preprocBody.classList.contains('hidden');
            if (isHidden) {
                preprocBody.classList.remove('hidden');
                preprocToggleIcon.textContent = '▲';
            } else {
                preprocBody.classList.add('hidden');
                preprocToggleIcon.textContent = '▼';
            }
        });
    }

    // 6. Submit Emotion Analysis
    analyzeBtn.addEventListener('click', performAnalysis);
    textInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) {
            performAnalysis();
        }
    });

    function showState(activeState) {
        emptyState.classList.add('hidden');
        loadingState.classList.add('hidden');
        errorState.classList.add('hidden');
        resultsContent.classList.add('hidden');

        if (activeState) {
            activeState.classList.remove('hidden');
        }
    }

    function showError(title, message) {
        errorTitle.textContent = title;
        errorMessage.textContent = message;
        showState(errorState);
    }

    async function performAnalysis() {
        const text = textInput.value.trim();
        if (!text) {
            showError('Empty Input', 'Please enter a sentence or select a sample prompt chip to analyze!');
            return;
        }

        // Show Loading
        loadingMsg.textContent = 'Processing text through TF-IDF Vectorizer & Logistic Model...';
        showState(loadingState);

        // Smooth scroll to results on mobile/small screens if stacked
        if (window.innerWidth < 890) {
            resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

        const maxRetries = 2;
        let attempt = 0;

        while (attempt <= maxRetries) {
            try {
                attempt++;
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 18000);

                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text }),
                    signal: controller.signal
                });
                clearTimeout(timeoutId);

                const contentType = response.headers.get('content-type') || '';
                if (!contentType.includes('application/json')) {
                    throw new Error(`Server returned HTML error (HTTP ${response.status}). Render instance may be spinning up.`);
                }

                const data = await response.json();

                if (response.ok && data.status === 'success') {
                    displayResults(data);
                    updateStatusPill('connected', 'Model Connected');
                    return;
                } else {
                    showError('Prediction Error', data.message || 'An error occurred during analysis.');
                    return;
                }
            } catch (error) {
                console.error(`API Error (Attempt ${attempt}):`, error);

                if (attempt <= maxRetries) {
                    loadingMsg.textContent = 'Backend is waking up on Render... Retrying analysis...';
                    updateStatusPill('waking', 'Waking Server...');
                    await new Promise(r => setTimeout(r, 3000));
                } else {
                    updateStatusPill('offline', 'Backend Offline');
                    showError(
                        'Connection Error',
                        'Failed to connect to Flask backend. If hosting on Render free tier, the server may be spinning up from sleep. Please click retry in a few seconds.'
                    );
                }
            }
        }
    }

    // Render Analysis Output
    function displayResults(data) {
        showState(resultsContent);

        const top = data.top_emotion;

        // Ambient glow background change
        if (top.bg_glow && ambientGlow) {
            ambientGlow.style.background = `radial-gradient(circle, ${top.bg_glow} 0%, rgba(30, 27, 75, 0) 70%)`;
        }

        // Hero banner update
        heroEmoji.textContent = top.emoji;
        heroEmotionName.textContent = top.name;
        heroEmotionName.style.color = top.color;
        heroEmotionDesc.textContent = top.description;
        heroConfidenceVal.textContent = `${top.confidence}%`;
        topEmotionBanner.style.borderColor = top.color;

        // Render Bars
        barsContainer.innerHTML = '';
        data.ranked_emotions.forEach((emo, idx) => {
            const barItem = document.createElement('div');
            barItem.className = 'bar-item';
            barItem.innerHTML = `
                <div class="bar-label-row">
                    <span class="bar-emotion-name">
                        <span>${emo.emoji}</span>
                        <span style="color: ${emo.color}; font-weight: 600;">${emo.name}</span>
                    </span>
                    <span class="bar-val" style="font-weight: 600;">${emo.percentage}%</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill" id="barFill_${idx}" style="background: ${emo.color}; box-shadow: 0 0 10px ${emo.color};"></div>
                </div>
            `;
            barsContainer.appendChild(barItem);

            // Animate bar width with slight stagger delay
            setTimeout(() => {
                const fillEl = document.getElementById(`barFill_${idx}`);
                if (fillEl) {
                    fillEl.style.width = `${emo.percentage}%`;
                }
            }, 50 + idx * 80);
        });

        // NLP Token Insights
        if (data.word_tokens) {
            tokenOriginalCount.textContent = data.word_count || 0;
            tokenOriginalText.textContent = data.input_text || '-';
            tokenCleanedText.textContent = (data.word_tokens.cleaned || []).join(' ') || '(no valid word tokens remaining)';
        }
    }

    // 7. Sample Prompts Fetch
    async function loadSamplePrompts() {
        try {
            const res = await fetch('/api/examples');
            const data = await res.json();
            if (data.status === 'success') {
                samplesContainer.innerHTML = '';
                data.examples.forEach(item => {
                    const chip = document.createElement('button');
                    chip.className = 'chip';
                    chip.innerHTML = `<span>💬</span> ${item.text.slice(0, 32)}...`;
                    chip.title = item.text;
                    chip.addEventListener('click', () => {
                        textInput.value = item.text;
                        charCounter.textContent = `${item.text.length} / 500`;
                        performAnalysis();
                    });
                    samplesContainer.appendChild(chip);
                });
            }
        } catch (e) {
            console.log('Sample load error:', e);
        }
    }
});


