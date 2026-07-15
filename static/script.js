/* ═══════════════════════════════════════════════════════════
   INSAF-GUIDE — Chat Frontend Logic
   ═══════════════════════════════════════════════════════════ */

const chatArea = document.getElementById('chatArea');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const welcomeCard = document.getElementById('welcomeCard');

let isProcessing = false;
let chatHistory = [];   // ← conversation memory

// ── Category Config ──
const categoryConfig = {
    'Criminal': { icon: '🚔', class: 'cat-criminal' },
    'Civil': { icon: '📋', class: 'cat-civil' },
    'Family': { icon: '👨‍👩‍👧', class: 'cat-family' },
    'Corporate': { icon: '🏢', class: 'cat-corporate' },
    'Constitutional': { icon: '📜', class: 'cat-constitutional' },
    'Tax': { icon: '💰', class: 'cat-constitutional' },
    'Labour': { icon: '👷', class: 'cat-civil' },
    'Shariah': { icon: '🕌', class: 'cat-family' },
    'General': { icon: '⚖️', class: 'cat-constitutional' },
};

// ── Urdu / Arabic Script Detection ──
function isUrdu(text) {
    return /[\u0600-\u06FF]/.test(text);
}

// ── Auto-resize Textarea ──
userInput.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});

// ── Enter to Send ──
userInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// ── Topic Shortcut ──
function askTopic(question) {
    userInput.value = question;
    userInput.style.height = 'auto';
    userInput.style.height = Math.min(userInput.scrollHeight, 120) + 'px';
    sendMessage();
}

// ── Send Message ──
async function sendMessage() {
    const message = userInput.value.trim();
    if (!message || isProcessing) return;

    isProcessing = true;
    sendBtn.disabled = true;

    // Hide welcome card
    if (welcomeCard) welcomeCard.style.display = 'none';

    // Add user message to UI
    appendMessage('user', message);
    userInput.value = '';
    userInput.style.height = 'auto';

    // Show typing indicator
    const typingEl = showTypingIndicator();

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                chat_history: chatHistory,   // ← send history to backend
            }),
        });

        typingEl.remove();

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || `Server error: ${response.status}`);
        }

        const data = await response.json();

        // Append bot reply
        appendMessage('bot', data.response, data.category, data.is_vague, data.confidence);

        // Update conversation history
        chatHistory.push(`User: ${message}`);
        chatHistory.push(`Assistant: ${data.response.slice(0, 300)}`);

        // Keep last 20 turns to avoid huge payloads
        if (chatHistory.length > 20) chatHistory = chatHistory.slice(-20);

    } catch (error) {
        typingEl.remove();
        appendMessage('bot',
            `⚠️ **Connection Error**\n\nCould not reach the server. Please ensure:\n1. The server is running (\`python src/main.py\`)\n2. Your GROQ_API_KEY is set\n\n*Error: ${error.message}*`,
            '', false, ''
        );
    }

    isProcessing = false;
    sendBtn.disabled = false;
    userInput.focus();
}

// ── Append Message ──
function appendMessage(role, content, category = '', isVague = false, confidence = '') {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message message-${role}`;

    // Apply Urdu RTL class if needed
    if (isUrdu(content)) {
        msgDiv.classList.add('urdu');
    }

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? '👤' : '⚖️';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    if (role === 'bot') {
        // ── Category badge ──
        if (category && !isVague) {
            const config = categoryConfig[category] || categoryConfig['General'];
            const badge = document.createElement('div');
            badge.className = `category-badge ${config.class}`;
            badge.innerHTML = `${config.icon} ${category}`;
            contentDiv.appendChild(badge);
        }

        // ── Confidence badge ──
        if (confidence && !isVague) {
            const confBadge = document.createElement('span');
            confBadge.className = `confidence-badge confidence-${confidence.toLowerCase()}`;
            confBadge.textContent = `${confidence} confidence`;
            contentDiv.appendChild(confBadge);
        }

        // ── Markdown render ──
        const textDiv = document.createElement('div');
        textDiv.innerHTML = marked.parse(content, { breaks: true, gfm: true });
        contentDiv.appendChild(textDiv);

    } else {
        contentDiv.textContent = content;
    }

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(contentDiv);
    chatArea.appendChild(msgDiv);

    requestAnimationFrame(() => {
        chatArea.scrollTop = chatArea.scrollHeight;
    });
}

// ── Typing Indicator ──
function showTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'typing-indicator';
    indicator.id = 'typingIndicator';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.style.background = 'linear-gradient(135deg, var(--accent-emerald), #047857)';
    avatar.style.boxShadow = '0 4px 15px rgba(16, 185, 129, 0.3)';
    avatar.textContent = '⚖️';

    const dotsContainer = document.createElement('div');
    dotsContainer.className = 'typing-dots';
    for (let i = 0; i < 3; i++) {
        const dot = document.createElement('div');
        dot.className = 'typing-dot';
        dotsContainer.appendChild(dot);
    }

    const text = document.createElement('span');
    text.className = 'typing-text';
    text.textContent = 'Analyzing legal provisions...';

    const statusTexts = [
        'Analyzing legal provisions...',
        'Searching relevant sections...',
        'Verifying legal accuracy...',
        'Preparing cited response...',
    ];
    let textIdx = 0;
    const textInterval = setInterval(() => {
        textIdx = (textIdx + 1) % statusTexts.length;
        text.textContent = statusTexts[textIdx];
    }, 3000);

    indicator._interval = textInterval;
    const originalRemove = indicator.remove.bind(indicator);
    indicator.remove = function () {
        clearInterval(textInterval);
        originalRemove();
    };

    indicator.appendChild(avatar);
    indicator.appendChild(dotsContainer);
    indicator.appendChild(text);
    chatArea.appendChild(indicator);

    chatArea.scrollTop = chatArea.scrollHeight;
    return indicator;
}

// ── Initial Focus ──
userInput.focus();