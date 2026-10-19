// Auto-migrate stale Render manual override so Vercel's native serverless functions take over
try {
  const existingBackend = localStorage.getItem('TRADINGAGENTS_BACKEND_URL');
  if (existingBackend && existingBackend.includes('onrender.com')) {
    localStorage.removeItem('TRADINGAGENTS_BACKEND_URL');
  }
} catch (e) {}

function getBackendEndpoint(url) {
  if (typeof url === 'string' && (url.startsWith('/api') || url.startsWith('/health'))) {
    const custom = localStorage.getItem('TRADINGAGENTS_BACKEND_URL');
    if (custom) {
      return `${custom.replace(/\/$/, '')}${url}`;
    }
  }
  return url;
}

const originalFetch = window.fetch;
window.fetch = function(resource, init) {
  if (typeof resource === 'string') {
    resource = getBackendEndpoint(resource);
  }
  return originalFetch.call(this, resource, init);
};

function updateBackendBadge() {
  const custom = localStorage.getItem('TRADINGAGENTS_BACKEND_URL');
  const badge = document.getElementById('active-backend-badge');
  const dot = document.getElementById('backend-status-dot');
  if (badge) {
    if (custom) {
      const cleanHost = custom.replace(/^https?:\/\//, '').split('/')[0];
      badge.innerText = `🔗 ${cleanHost.slice(0, 18)}`;
      badge.title = `Conectado a: ${custom} (Click para cambiar URL)`;
    } else {
      badge.innerText = `⚡ Vercel Cloud`;
      badge.title = `Vercel Serverless conectado a Alpaca en tiempo real (Click para cambiar URL)`;
    }
  }
  if (dot) dot.className = 'status-dot online';
}


function promptBackendUrl() {
  const current = localStorage.getItem('TRADINGAGENTS_BACKEND_URL') || '';
  const newUrl = prompt(
    'Configurar URL del Servidor Backend (Render Cloud o Túnel):\n\n' +
    'Ejemplo: https://bot-trader-dear.onrender.com\n' +
    'Deja vacío para usar el backend automático / Vercel proxy:',
    current
  );
  if (newUrl !== null) {
    if (newUrl.trim() === '' || newUrl.trim().toLowerCase() === 'default') {
      localStorage.removeItem('TRADINGAGENTS_BACKEND_URL');
      alert('Configuración restaurada al backend por defecto.');
    } else {
      let cleaned = newUrl.trim().replace(/\/$/, '');
      if (!cleaned.startsWith('http://') && !cleaned.startsWith('https://')) {
        cleaned = 'https://' + cleaned;
      }
      localStorage.setItem('TRADINGAGENTS_BACKEND_URL', cleaned);
      alert(`Servidor backend configurado a: ${cleaned}`);
    }
    window.location.reload();
  }
}

let equityChartInstance = null;
let currentTab = 'all';
let equityHistory = [];

document.addEventListener('DOMContentLoaded', () => {
  updateBackendBadge();
  initEquityChart();
  loadEquityHistory();
  fetchTelemetryData();
  fetchWatchlist();
  fetchLogs();
  fetchMarketRegime();

  // Polling every 5 seconds for telemetry, regime, logs, and notifications
  setInterval(fetchTelemetryData, 5000);
  setInterval(fetchMarketRegime, 10000);
  setInterval(fetchLogs, 3000);
  setInterval(fetchNotifications, 4000);
  fetchNotifications();

  // Check initial notification push permission
  checkNotificationPermissionUI();
});


// Fetch Market Regime & Volatility Specialist
async function fetchMarketRegime() {
  try {
    const res = await fetch('/api/market/regime');
    if (!res.ok) return;
    const data = await res.json();
    const badge = document.getElementById('active-regime-badge');
    const dot = document.getElementById('regime-status-dot');
    if (badge) {
      badge.innerText = `Régimen: ${data.regime_display} (Vol: ${data.realized_volatility_20d}%)`;
      badge.title = `${data.rationale} | TP: ${data.tp_multiplier}x ATR, SL: ${data.sl_multiplier}x ATR, BE: +${data.break_even_pct}%`;
    }
    if (dot) {
      dot.className = `status-dot ${data.regime === 'DEFENSIVE_BEAR_BREAKDOWN' ? 'offline' : 'online'}`;
    }

    if (data.web_sentiment) {
      const sBadge = document.getElementById('active-sentiment-badge');
      const sDot = document.getElementById('sentiment-status-dot');
      if (sBadge) {
        sBadge.innerText = `🌐 Web: ${data.web_sentiment.sentiment_label} (FGI: ${data.web_sentiment.fear_greed_value})`;
        const catalysts = (data.web_sentiment.macro_catalysts || []).slice(0, 3).join(' \n• ');
        sBadge.title = `Índice Miedo/Codicia: ${data.web_sentiment.fear_greed_value}/100 (${data.web_sentiment.fear_greed_classification})\nScore Sentimiento: ${data.web_sentiment.sentiment_score}\nTitulares Macro:\n• ${catalysts}`;
      }
      if (sDot) {
        sDot.className = `status-dot ${data.web_sentiment.sentiment_score < -0.3 ? 'offline' : 'online'}`;
      }
    }
  } catch (err) {
    console.warn('Regime fetch error:', err);
  }
}

// Load historical equity curve from Neon PostgreSQL
async function loadEquityHistory() {
  try {
    const res = await fetch('/api/equity/history?limit=30');
    if (!res.ok) return;
    const history = await res.json();
    if (history && history.length > 0 && equityChartInstance) {
      equityChartInstance.data.labels = history.map(h => {
        const d = new Date(h.recorded_at);
        return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      });
      equityChartInstance.data.datasets[0].data = history.map(h => h.equity);
      equityChartInstance.update('none');
    }
  } catch (err) {
    console.debug('Equity history note:', err);
  }
}

// Chart Initialization
function initEquityChart() {
  const ctx = document.getElementById('equityChart').getContext('2d');
  const gradient = ctx.createLinearGradient(0, 0, 0, 300);
  gradient.addColorStop(0, 'rgba(99, 102, 241, 0.4)');
  gradient.addColorStop(1, 'rgba(99, 102, 241, 0.0)');

  equityChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Equity ($ USD)',
        data: [],
        borderColor: '#6366F1',
        borderWidth: 2.5,
        backgroundColor: gradient,
        fill: true,
        tension: 0.35,
        pointRadius: 4,
        pointHoverRadius: 6,
        pointBackgroundColor: '#06B6D4',
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#111827',
          borderColor: 'rgba(255, 255, 255, 0.1)',
          borderWidth: 1,
          titleColor: '#F8FAFC',
          bodyColor: '#10B981',
          displayColors: false,
          callbacks: {
            label: (ctx) => `$${ctx.parsed.y.toLocaleString('en-US', { minimumFractionDigits: 2 })} USD`
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: { color: '#64748B', font: { family: 'Plus Jakarta Sans', size: 11 } }
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: {
            color: '#64748B',
            font: { family: 'Plus Jakarta Sans', size: 11 },
            callback: (v) => `$${v.toLocaleString()}`
          }
        }
      }
    }
  });
}

// Fetch Account Telemetry & Positions
async function fetchTelemetryData() {
  try {
    const res = await fetch('/api/balance');
    if (!res.ok) return;
    const data = await res.json();

    if (data.error || typeof data.equity !== 'number') {
      const badge = document.getElementById('active-backend-badge');
      const dot = document.getElementById('backend-status-dot');
      if (badge) {
        badge.innerText = '⚠️ Conectar Backend';
        badge.title = `El backend actual no responde correctamente (${data.error || 'sin datos'}). Haz clic aquí para ingresar la URL de tu servicio Render.`;
        badge.style.color = 'var(--accent-warning)';
      }
      if (dot) dot.className = 'status-dot offline';
      return;
    }

    
    const badge = document.getElementById('active-broker-badge');
    if (badge) {
      if (data.is_simulation) {
        badge.innerText = 'Modo: Simulación (Sintética)';
        badge.style.color = 'var(--accent-warning)';
      } else {
        badge.innerText = 'Broker: Alpaca (Datos Reales)';
        badge.style.color = 'var(--accent-success)';
      }
    }

    const pnl = data.daily_pnl || 0;
    const pnlEl = document.getElementById('kpi-pnl');
    const formattedPnl = Math.abs(pnl).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    pnlEl.innerText = `${pnl < 0 ? '-' : (pnl > 0 ? '+' : '')}$${formattedPnl}`;
    pnlEl.style.color = pnl > 0 ? 'var(--accent-success)' : (pnl < 0 ? 'var(--accent-danger)' : 'var(--text-primary)');

    if (document.getElementById('kpi-pnl-realized')) {
      if (data.is_simulation) {
        const realPnl = data.realized_pnl || 0;
        const tradesCount = data.closed_trades || 0;
        document.getElementById('kpi-pnl-realized').innerText = `Realizado: ${realPnl >= 0 ? '+' : ''}$${realPnl.toFixed(2)} (${tradesCount} trades)`;
        document.getElementById('kpi-pnl-realized').style.color = realPnl >= 0 ? 'var(--accent-success)' : 'var(--accent-danger)';
      } else {
        document.getElementById('kpi-pnl-realized').innerText = 'Conectado a Alpaca Matching Engine';
        document.getElementById('kpi-pnl-realized').style.color = 'var(--text-muted)';
      }
    }

    const dd = data.daily_drawdown_pct || 0;
    const ddEl = document.getElementById('kpi-drawdown');
    ddEl.innerText = `${dd.toFixed(2)}%`;
    ddEl.style.color = dd > 2.0 ? 'var(--accent-danger)' : 'var(--text-primary)';

    if (document.getElementById('kpi-winrate-sub')) {
      const winRate = data.win_rate || 0;
      document.getElementById('kpi-winrate-sub').innerText = `Win Rate: ${winRate.toFixed(1)}%`;
    }

    // Update Chart
    const nowTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    if (equityChartInstance) {
      if (equityChartInstance.data.labels.length === 0) {
        equityChartInstance.data.labels.push(nowTime);
        equityChartInstance.data.datasets[0].data.push(data.equity);
      } else {
        if (equityChartInstance.data.labels.length > 25) {
          equityChartInstance.data.labels.shift();
          equityChartInstance.data.datasets[0].data.shift();
        }
        equityChartInstance.data.labels.push(nowTime);
        equityChartInstance.data.datasets[0].data.push(data.equity);
      }
      equityChartInstance.update('none');
    }

    // Health & DB status
    try {
      const hRes = await fetch('/health');
      if (hRes.ok) {
        const hData = await hRes.json();
        const dbBadge = document.getElementById('active-db-badge');
        const dbDot = document.getElementById('db-status-dot');
        if (dbBadge && dbDot) {
          if (hData.neon_connected) {
            dbDot.className = 'status-dot online';
            dbBadge.innerText = 'DB: Neon PostgreSQL (Nube)';
            dbBadge.style.color = 'var(--accent-success)';
          } else {
            dbDot.className = 'status-dot busy';
            dbBadge.innerText = 'DB: Modo Offline';
            dbBadge.style.color = 'var(--accent-warning)';
          }
        }
      }
    } catch (_) {}

    // Positions & Orders
    fetchPositions();
    fetchOrders();
  } catch (err) {
    console.warn('Telemetry error:', err);
  }
}

// Fetch Positions
async function fetchPositions() {
  try {
    const res = await fetch('/api/positions');
    if (!res.ok) return;
    const positions = await res.json();

    const tbody = document.getElementById('positions-table-body');
    const badge = document.getElementById('positions-count-badge');
    badge.innerText = `${positions.length} Activas`;

    if (positions.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-muted); padding: 32px;">No hay posiciones abiertas actualmente.</td></tr>`;
      return;
    }

    tbody.innerHTML = positions.map(p => {
      const pnlColor = p.unrealized_pnl >= 0 ? 'var(--accent-success)' : 'var(--accent-danger)';
      return `
        <tr>
          <td data-label="Activo"><span class="symbol-badge">${p.symbol}</span></td>
          <td data-label="Tipo"><span class="badge ${p.asset_class === 'CRYPTO' ? 'badge-info' : 'badge-warning'}">${p.asset_class}</span></td>
          <td data-label="Cantidad">${p.quantity.toFixed(4)}</td>
          <td data-label="Entrada">$${p.entry_price.toLocaleString('en-US', { minimumFractionDigits: 2 })}</td>
          <td data-label="Precio Actual">$${p.current_price.toLocaleString('en-US', { minimumFractionDigits: 2 })}</td>
          <td data-label="P&L Flotante" style="color: ${pnlColor}; font-weight: 700;">
            ${p.unrealized_pnl >= 0 ? '+' : ''}$${p.unrealized_pnl.toFixed(2)} (${p.unrealized_pnl_pct.toFixed(2)}%)
          </td>
        </tr>
      `;
    }).join('');
  } catch (err) {
    console.warn('Positions error:', err);
  }
}

// Fetch Orders
async function fetchOrders() {
  try {
    const res = await fetch('/api/orders');
    if (!res.ok) return;
    const orders = await res.json();
    const tbody = document.getElementById('orders-table-body');

    if (orders.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted); padding: 32px;">No hay órdenes ejecutadas en esta sesión.</td></tr>`;
      return;
    }

    tbody.innerHTML = orders.slice(0, 10).map(o => {
      const isBuy = o.side === 'BUY';
      return `
        <tr>
          <td data-label="ID"><code>${o.order_id}</code></td>
          <td data-label="Símbolo"><strong>${o.symbol}</strong></td>
          <td data-label="Operación"><span class="badge ${isBuy ? 'badge-success' : 'badge-danger'}">${o.side}</span></td>
          <td data-label="Cantidad">${o.quantity}</td>
          <td data-label="Precio">$${(o.filled_price || o.limit_price || 0).toLocaleString('en-US', { minimumFractionDigits: 2 })}</td>
          <td data-label="Estado"><span class="status-dot ${o.status === 'FILLED' ? 'online' : 'busy'}" style="display:inline-block; margin-right:4px;"></span>${o.status}</td>
          <td data-label="Hora">${o.created_at ? new Date(o.created_at).toLocaleTimeString() : '-'}</td>
        </tr>
      `;
    }).join('');
  } catch (err) {
    console.warn('Orders error:', err);
  }
}

// Fetch Watchlist
async function fetchWatchlist() {
  try {
    const res = await fetch('/api/watchlist');
    if (!res.ok) return;
    const data = await res.json();
    const container = document.getElementById('watchlist-container');
    const all = [...(data.equities || []), ...(data.crypto || [])];
    container.innerHTML = all.map(sym => `
      <span class="tag-item">● ${sym}</span>
    `).join('');
  } catch (err) {
    console.warn('Watchlist error:', err);
  }
}

async function addWatchlist() {
  const input = document.getElementById('new-ticker-input');
  const symbol = input.value.trim().toUpperCase();
  if (!symbol) return;
  await fetch('/api/watchlist', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'add', symbol }),
  });
  input.value = '';
  fetchWatchlist();
}

async function removeWatchlist(symbol) {
  await fetch('/api/watchlist', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'remove', symbol }),
  });
  fetchWatchlist();
}

// Fetch Logs for Terminal
async function fetchLogs() {
  try {
    const res = await fetch(`/api/logs?filter=${currentTab}`);
    if (!res.ok) return;
    const logs = await res.json();
    const container = document.getElementById('terminal-logs');
    if (logs && logs.length > 0) {
      container.innerHTML = logs.map(l => formatLogLine(l)).join('');
      container.scrollTop = container.scrollHeight;
    }
  } catch (err) {
    console.warn('Logs error:', err);
  }
}

function formatLogLine(log) {
  let contentClass = 'log-info';
  if (log.includes('Bull')) contentClass = 'log-bull';
  else if (log.includes('Bear')) contentClass = 'log-bear';
  else if (log.includes('FILLED') || log.includes('BUY')) contentClass = 'log-action';
  else if (log.includes('REJECTED')) contentClass = 'log-bear';

  return `<div class="log-entry"><span class="log-time">[${new Date().toLocaleTimeString()}]</span> <span class="${contentClass}">${log}</span></div>`;
}

function switchTerminalTab(tab) {
  currentTab = tab;
  document.querySelectorAll('.terminal-tab').forEach(t => t.classList.remove('active'));
  if (event && event.target) event.target.classList.add('active');
  fetchLogs();
}

function clearTerminal() {
  document.getElementById('terminal-logs').innerHTML = '';
}

// Actions
async function triggerManualTrade() {
  const symbol = document.getElementById('manual-symbol').value.trim().toUpperCase();
  if (!symbol) {
    alert('Introduce un símbolo válido (ej: NVDA)');
    return;
  }
  document.getElementById('bot-status-text').innerText = `Analizando ${symbol}...`;
  document.getElementById('bot-status-dot').className = 'status-dot busy';

  const res = await fetch('/api/trade', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ symbol })
  });
  const data = await res.json();
  alert(data.message || 'Ciclo de trading iniciado.');
  fetchTelemetryData();
}

async function triggerFullScan() {
  if (!confirm('¿Deseas iniciar un escaneo completo de todos los activos de la cartera?')) return;
  document.getElementById('bot-status-text').innerText = 'Escaneando Cartera...';
  document.getElementById('bot-status-dot').className = 'status-dot busy';

  const res = await fetch('/api/scan', { method: 'POST' });
  const data = await res.json();
  alert(data.message || 'Escaneo iniciado en segundo plano.');
}

async function triggerKillSwitch() {
  if (!confirm('🚨 ATENCIÓN: El Kill Switch cancelará todas las órdenes pendientes y detendrá las compras. ¿Confirmar?')) return;
  const res = await fetch('/api/kill-switch', { method: 'POST' });
  const data = await res.json();
  alert(data.message || 'Kill switch activado.');
  fetchTelemetryData();
}

// Settings Modal
function openSettingsModal() {
  document.getElementById('settings-modal').classList.add('open');
}

function closeSettingsModal() {
  document.getElementById('settings-modal').classList.remove('open');
}

async function saveSettings() {
  const geminiKey = document.getElementById('cfg-gemini-key').value.trim();
  const openrouterKey = document.getElementById('cfg-openrouter-key').value.trim();
  const groqKey = document.getElementById('cfg-groq-key').value.trim();
  const mistralKey = document.getElementById('cfg-mistral-key').value.trim();
  const provider = document.getElementById('cfg-provider-select').value;
  const riskTrade = parseFloat(document.getElementById('cfg-risk-trade').value);
  const maxDrawdown = parseFloat(document.getElementById('cfg-max-drawdown').value);

  const payload = {
    llm_provider: provider,
    max_risk_per_trade_pct: riskTrade,
    max_daily_drawdown_pct: maxDrawdown,
  };
  if (geminiKey) payload.google_api_key = geminiKey;
  if (openrouterKey) payload.openrouter_api_key = openrouterKey;
  if (groqKey) payload.groq_api_key = groqKey;
  if (mistralKey) payload.mistral_api_key = mistralKey;

  const res = await fetch('/api/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  alert(data.message || 'Configuración actualizada.');
  closeSettingsModal();
  document.getElementById('active-model-badge').innerText = `LLM: ${provider === 'rotator' ? 'Pool Rotativo' : (provider === 'google' ? 'Google Gemini' : 'OpenRouter')}`;
}

// Live Simulation Toggle
async function toggleSimulation() {
  const btn = document.getElementById('btn-toggle-sim');
  const res = await fetch('/api/simulation/toggle', { method: 'POST' });
  const data = await res.json();
  updateSimulationUI(data.running);
}

// Reset Simulation Account
async function resetSimulation() {
  if (!confirm('¿Deseas reiniciar la cartera de simulación a $100,000.00 USD con saldo limpio?')) return;
  const res = await fetch('/api/simulation/reset', { method: 'POST' });
  const data = await res.json();
  alert(data.message || 'Simulación reiniciada.');
  fetchTelemetryData();
  fetchLogs();
}

function updateSimulationUI(isRunning) {
  const btn = document.getElementById('btn-toggle-sim');
  const badge = document.getElementById('active-broker-badge');
  const statusDot = document.getElementById('bot-status-dot');
  const statusText = document.getElementById('bot-status-text');

  if (isRunning) {
    btn.innerText = '⏸️ Pausar Simulación';
    btn.className = 'btn btn-danger';
    badge.innerText = 'Broker: Simulación Activa (Sin API)';
    statusDot.className = 'status-dot busy';
    statusText.innerText = 'Simulación Operando';
  } else {
    btn.innerText = '▶️ Simulación en Vivo';
    btn.className = 'btn btn-primary';
    badge.innerText = 'Broker: Alpaca (Paper)';
    statusDot.className = 'status-dot online';
    statusText.innerText = 'Server Activo';
  }
}

// ==============================================================================
// MOBILE NAVIGATION CONTROLLER & VIEW SWITCHER
// ==============================================================================

function switchMobileView(viewName) {
  // Update bottom navigation bar button states
  document.querySelectorAll('.mobile-bottom-nav .mobile-nav-btn').forEach(btn => {
    btn.classList.remove('active');
  });
  const targetNavBtn = document.getElementById(`mob-nav-${viewName}`);
  if (targetNavBtn) targetNavBtn.classList.add('active');

  // In mobile viewports (< 768px), toggle specific view section
  if (window.innerWidth <= 768) {
    const panes = ['portfolio', 'orders', 'agents'];
    panes.forEach(pane => {
      const el = document.getElementById(`view-${pane}`);
      if (el) {
        if (pane === viewName) {
          el.classList.add('active');
        } else {
          el.classList.remove('active');
        }
      }
    });
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
}

window.addEventListener('resize', () => {
  if (window.innerWidth > 768) {
    document.querySelectorAll('.mobile-view-pane').forEach(el => el.classList.add('active'));
  } else {
    const activeBtn = document.querySelector('.mobile-bottom-nav .mobile-nav-btn.active');
    const view = activeBtn ? activeBtn.id.replace('mob-nav-', '') : 'portfolio';
    switchMobileView(view);
  }
});

// ==============================================================================
// NOTIFICATIONS, TOASTS & AUDIO CHIME SYSTEM
// ==============================================================================

let lastSeenNotifId = null;
let unreadNotifCount = 0;
let audioAlertsEnabled = localStorage.getItem('audio_alerts') !== 'false';

function toggleAudioAlerts(enabled) {
  audioAlertsEnabled = enabled;
  localStorage.setItem('audio_alerts', enabled ? 'true' : 'false');
  if (enabled) playChime();
}

function playChime() {
  if (!audioAlertsEnabled) return;
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;
    const ctx = new AudioCtx();
    const now = ctx.currentTime;

    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.type = 'sine';
    osc.frequency.setValueAtTime(587.33, now); // D5
    osc.frequency.exponentialRampToValueAtTime(880.00, now + 0.12); // A5

    gain.gain.setValueAtTime(0.12, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.45);

    osc.connect(gain);
    gain.connect(ctx.destination);

    osc.start(now);
    osc.stop(now + 0.5);
  } catch (_) {}
}

function showToast(title, message, icon = '⚡', type = 'ORDER_BUY') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  playChime();

  const toast = document.createElement('div');
  toast.className = `toast-card toast-${type}`;
  toast.innerHTML = `
    <div class="toast-icon">${icon}</div>
    <div class="toast-body">
      <div class="toast-title">${title}</div>
      <div class="toast-message">${message}</div>
      <div class="toast-time">Ahora</div>
    </div>
    <button class="toast-close" onclick="this.parentElement.remove()">✕</button>
  `;

  container.appendChild(toast);
  requestAnimationFrame(() => toast.classList.add('show'));

  // Native browser push notification if permission granted
  if (typeof Notification !== 'undefined' && Notification.permission === 'granted') {
    try {
      new Notification(`TradingAgents: ${title}`, {
        body: message,
        icon: '/static/favicon.ico',
        tag: `ta-${Date.now()}`,
      });
    } catch (_) {}
  }

  setTimeout(() => {
    toast.classList.remove('show');
    toast.classList.add('hide');
    setTimeout(() => toast.remove(), 400);
  }, 4500);
}

function checkNotificationPermissionUI() {
  const btn = document.getElementById('btn-request-push');
  if (!btn) return;
  if (typeof Notification !== 'undefined' && Notification.permission === 'granted') {
    btn.innerText = '✅ Notificaciones Activas';
    btn.className = 'btn btn-secondary';
    btn.disabled = true;
  }
}

async function enablePushNotifications() {
  if (typeof Notification === 'undefined') {
    alert('Las notificaciones Web no están soportadas en este navegador.');
    return;
  }
  const perm = await Notification.requestPermission();
  checkNotificationPermissionUI();
  if (perm === 'granted') {
    showToast('Notificaciones Nativas Habilitadas', 'Recibirás avisos en tu pantalla de bloqueo y barra de estado.', '🔔', 'TEST_ALERT');
  } else {
    alert('Permiso de notificaciones no concedido. Puedes habilitarlo en los ajustes del navegador.');
  }
}

function toggleNotificationsModal() {
  const modal = document.getElementById('notifications-modal');
  if (!modal) return;
  const isHidden = modal.style.display === 'none' || !modal.style.display;
  modal.style.display = isHidden ? 'flex' : 'none';

  if (isHidden) {
    unreadNotifCount = 0;
    updateBadgeUI();
    fetchNotifications();
  }
}

function updateBadgeUI() {
  const badge = document.getElementById('notif-badge-count');
  const mobBadge = document.getElementById('mobile-badge-count');
  const txt = unreadNotifCount > 9 ? '9+' : `${unreadNotifCount}`;
  if (badge) {
    badge.innerText = txt;
    badge.style.display = unreadNotifCount > 0 ? 'flex' : 'none';
  }
  if (mobBadge) {
    mobBadge.innerText = txt;
    mobBadge.style.display = unreadNotifCount > 0 ? 'flex' : 'none';
  }
}

async function fetchNotifications() {
  try {
    const res = await fetch('/api/notifications');
    if (!res.ok) return;
    const events = await res.json();
    const feed = document.getElementById('notifications-feed');

    if (!events || events.length === 0) {
      if (feed) feed.innerHTML = '<div style="text-align: center; color: var(--text-muted); padding: 24px;">No hay eventos recientes.</div>';
      return;
    }

    // Check for new incoming notifications to trigger toast
    if (lastSeenNotifId && events.length > 0 && events[0].id !== lastSeenNotifId) {
      const newest = events[0];
      unreadNotifCount += 1;
      updateBadgeUI();
      showToast(newest.title, newest.message, newest.icon || '⚡', newest.type);
    }
    lastSeenNotifId = events[0].id;

    if (feed) {
      feed.innerHTML = events.map(ev => {
        const timeStr = ev.timestamp ? new Date(ev.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '-';
        return `
          <div class="notif-item">
            <span class="notif-item-icon">${ev.icon || '⚡'}</span>
            <div class="notif-item-body">
              <div class="notif-item-title">${ev.title}</div>
              <div class="notif-item-msg">${ev.message}</div>
              <div class="notif-item-time">${timeStr}</div>
            </div>
          </div>
        `;
      }).join('');
    }
  } catch (err) {
    console.debug('Notification fetch note:', err);
  }
}

async function triggerTestNotification() {
  try {
    const res = await fetch('/api/notifications/test', { method: 'POST' });
    const data = await res.json();
    if (data.event) {
      showToast(data.event.title, data.event.message, data.event.icon, data.event.type);
      fetchNotifications();
    }
  } catch (e) {
    showToast('Prueba de Alerta Local', 'Sistema de notificaciones verificado en pantalla.', '🔔', 'TEST_ALERT');
  }
}

async function saveTelegramSettings() {
  const token = document.getElementById('input-tg-token').value.trim();
  const chat = document.getElementById('input-tg-chat').value.trim();
  if (!token && !chat) {
    alert('Ingresa el token y chat ID de Telegram.');
    return;
  }
  const res = await fetch('/api/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      telegram_bot_token: token || undefined,
      telegram_chat_id: chat || undefined,
    }),
  });
  const data = await res.json();
  showToast('Telegram Configurado', 'Credenciales actualizadas para alertas 24/7.', '📱', 'TEST_ALERT');
  const badge = document.getElementById('telegram-status-badge');
  if (badge) {
    badge.innerText = 'Conectado';
    badge.className = 'badge badge-success';
  }
}

document.addEventListener('DOMContentLoaded', checkSimulationStatus);
