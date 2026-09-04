/**
 * Emotion AI - Frontend Application Logic
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const textInput = document.getElementById('textInput');
    const charCounter = document.getElementById('charCounter');
    const clearBtn = document.getElementById('clearBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const samplesContainer = document.getElementById('samplesContainer');
    const ambientGlow = document.getElementById('ambientGlow');
    
    // States
    const emptyState = document.getElementById('emptyState');
    const loadingState = document.getElementById('loadingState');
    const resultsContent = document.getElementById('resultsContent');

    // Hero Top Display
    const heroEmoji = document.getElementById('heroEmoji');
    const heroEmotionName = document.getElementById('heroEmotionName');
    const heroEmotionDesc = document.getElementById('heroEmotionDesc');
    const heroConfidenceVal = document.getElementById('heroConfidenceVal');
    const topEmotionBanner = document.getElementById('topEmotionBanner');
    
    // Bars
    const barsContainer = document.getElementById('barsContainer');

    // Initial Setup
    loadSamplePrompts();

    // 1. Character Counter
    textInput.addEventListener('input', () => {
        const len = textInput.value.length;
        charCounter.textContent = `${len} / 500`;
    });

    // 2. Clear Button
    clearBtn.addEventListener('click', () => {
        textInput.value = '';
        charCounter.textContent = '0 / 500';
        textInput.focus();
    });

    // 3. Submit Emotion Analysis
    analyzeBtn.addEventListener('click', performAnalysis);
    textInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) {
            performAnalysis();
        }
    });

    async function performAnalysis() {
        const text = textInput.value.trim();
        if (!text) {
            alert('Please enter a sentence or select a sample prompt to analyze!');
            return;
        }

        // Show Loading
        emptyState.classList.add('hidden');
        resultsContent.classList.add('hidden');
        loadingState.classList.remove('hidden');

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });

            const data = await response.json();
            loadingState.classList.add('hidden');

            if (data.status === 'success') {
                displayResults(data);
            } else {
                alert(data.message || 'An error occurred during analysis.');
                emptyState.classList.remove('hidden');
            }
        } catch (error) {
            loadingState.classList.add('hidden');
            emptyState.classList.remove('hidden');
            alert('Failed to connect to the Flask server. Please check backend status.');
            console.error('API Error:', error);
        }
    }

    // Render Analysis Output
    function displayResults(data) {
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
                    <span class="bar-val">${emo.percentage}%</span>
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

        resultsContent.classList.remove('hidden');
    }

    // 4. Sample Prompts Fetch
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
