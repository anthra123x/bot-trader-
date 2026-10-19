// ==============================================================================
// TRADINGAGENTS OS - CLIENT CONTROLLER (MINIMALIST MONOCHROME EDITION)
// ==============================================================================

// Auto-migrate stale Render manual override so Vercel's native serverless functions take over
try {
  const existingBackend = localStorage.getItem('TRADINGAGENTS_BACKEND_URL');
  if (existingBackend && (existingBackend.includes('onrender.com') || existingBackend.includes('trycloudflare.com'))) {
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
      badge.innerText = `${cleanHost.slice(0, 16)}`;
      badge.title = `Conectado a: ${custom} (Click para cambiar URL)`;
    } else {
      badge.innerText = `Vercel Cloud`;
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
    'Deja vacío para usar el backend automático / Vercel Edge proxy:',
    current
  );
  if (newUrl !== null) {
    if (newUrl.trim() === '' || newUrl.trim().toLowerCase() === 'default') {
      localStorage.removeItem('TRADINGAGENTS_BACKEND_URL');
      alert('Configuración restaurada al backend por defecto (Vercel Serverless).');
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

document.addEventListener('DOMContentLoaded', () => {
  updateBackendBadge();
  initEquityChart();
  loadEquityHistory();
  fetchTelemetryData();
  fetchPositions();
  fetchOrders();
  fetchWatchlist();
  fetchLogs();
  fetchMarketRegime();

  // Periodic polling
  setInterval(fetchTelemetryData, 4000);
  setInterval(fetchPositions, 5000);
  setInterval(fetchOrders, 8000);
  setInterval(fetchMarketRegime, 10000);
  setInterval(fetchLogs, 3000);
  setInterval(fetchNotifications, 5000);
  fetchNotifications();

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
      badge.innerText = `Vol ${data.realized_volatility_20d}% • ${data.regime_display.split('/')[0].trim()}`;
      badge.title = `${data.rationale} | TP: ${data.tp_multiplier}x ATR, SL: ${data.sl_multiplier}x ATR`;
    }
    if (dot) {
      dot.className = `status-dot ${data.regime === 'DEFENSIVE_BEAR_BREAKDOWN' ? 'offline' : 'online'}`;
    }

    if (data.web_sentiment) {
      const sBadge = document.getElementById('active-sentiment-badge');
      const sDot = document.getElementById('sentiment-status-dot');
      if (sBadge) {
        sBadge.innerText = `FGI ${data.web_sentiment.fear_greed_value} • ${data.web_sentiment.sentiment_label}`;
      }
      if (sDot) {
        sDot.className = `status-dot ${data.web_sentiment.sentiment_score < -0.3 ? 'offline' : 'online'}`;
      }
    }
  } catch (err) {
    console.debug('Regime note:', err);
  }
}

// Load historical equity curve
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

// Minimalist Monochrome Chart Initialization
function initEquityChart() {
  const canvas = document.getElementById('equityChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const gradient = ctx.createLinearGradient(0, 0, 0, 240);
  gradient.addColorStop(0, 'rgba(255, 255, 255, 0.08)');
  gradient.addColorStop(1, 'rgba(255, 255, 255, 0.00)');

  equityChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Equity ($ USD)',
        data: [],
        borderColor: '#e4e4e7',
        borderWidth: 2,
        backgroundColor: gradient,
        fill: true,
        tension: 0.25,
        pointRadius: 3,
        pointHoverRadius: 5,
        pointBackgroundColor: '#ffffff',
        pointBorderColor: '#0c0c0e',
        pointBorderWidth: 1.5,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#161619',
          borderColor: '#27272a',
          borderWidth: 1,
          titleColor: '#a1a1aa',
          bodyColor: '#ffffff',
          titleFont: { family: 'Inter', size: 11 },
          bodyFont: { family: 'JetBrains Mono', size: 12, weight: '600' },
          padding: 8,
          displayColors: false,
          callbacks: {
            label: (ctx) => `$${ctx.parsed.y.toLocaleString('en-US', { minimumFractionDigits: 2 })} USD`
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.02)' },
          ticks: { color: '#71717a', font: { family: 'JetBrains Mono', size: 10 } }
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.03)' },
          ticks: {
            color: '#71717a',
            font: { family: 'JetBrains Mono', size: 10 },
            callback: (v) => `$${v.toLocaleString()}`
          }
        }
      }
    }
  });
}

// Fetch Account Telemetry
async function fetchTelemetryData() {
  try {
    const res = await fetch('/api/balance');
    if (!res.ok) return;
    const data = await res.json();

    if (data.error || typeof data.equity !== 'number') {
      const badge = document.getElementById('active-backend-badge');
      const dot = document.getElementById('backend-status-dot');
      if (badge) {
        badge.innerText = 'Error Conexión';
        badge.title = `Error: ${data.error || 'sin datos'}`;
      }
      if (dot) dot.className = 'status-dot offline';
      return;
    }

    // Set Primary KPIs
    const elEquity = document.getElementById('kpi-equity');
    if (elEquity) {
      elEquity.innerText = `$${data.equity.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    }
    const elBuyingPower = document.getElementById('kpi-buying-power');
    if (elBuyingPower) {
      elBuyingPower.innerText = `$${data.buying_power.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    }
    const elCash = document.getElementById('kpi-cash-sub');
    if (elCash) {
      elCash.innerText = `Cash: $${data.cash.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} USD`;
    }

    const badge = document.getElementById('active-broker-badge');
    if (badge) {
      badge.innerText = data.is_simulation ? 'Simulación Paper' : 'Alpaca Live Matching';
    }

    const pnl = data.daily_pnl || 0;
    const pnlEl = document.getElementById('kpi-pnl');
    if (pnlEl) {
      const formattedPnl = Math.abs(pnl).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
      pnlEl.innerText = `${pnl < 0 ? '-' : (pnl > 0 ? '+' : '')}$${formattedPnl}`;
      pnlEl.style.color = pnl > 0 ? 'var(--accent-success)' : (pnl < 0 ? 'var(--accent-danger)' : 'var(--text-primary)');
    }

    const dd = data.daily_drawdown_pct || 0;
    const ddEl = document.getElementById('kpi-drawdown');
    if (ddEl) {
      ddEl.innerText = `${dd.toFixed(2)}%`;
      ddEl.style.color = dd > 2.0 ? 'var(--accent-danger)' : 'var(--text-primary)';
    }

    // Update Chart with live point
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
          dbDot.className = hData.neon_connected ? 'status-dot online' : 'status-dot busy';
          dbBadge.innerText = hData.neon_connected ? 'Neon Cloud' : 'DB Offline';
        }
      }
    } catch (_) {}

  } catch (err) {
    console.debug('Telemetry error:', err);
  }
}

// Fetch Active Positions
async function fetchPositions() {
  try {
    const res = await fetch('/api/positions');
    if (!res.ok) return;
    const positions = await res.json();

    const tbody = document.getElementById('positions-table-body');
    const badge = document.getElementById('positions-count-badge');
    if (badge) badge.innerText = `${positions.length} Activas`;

    if (!tbody) return;
    if (!Array.isArray(positions) || positions.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-muted); padding: 32px;">No hay posiciones abiertas actualmente.</td></tr>`;
      return;
    }

    tbody.innerHTML = positions.map(p => {
      const pnlColor = p.unrealized_pnl >= 0 ? 'var(--accent-success)' : 'var(--accent-danger)';
      const qtyStr = p.quantity < 0.01 
        ? p.quantity.toFixed(6) 
        : p.quantity.toLocaleString('en-US', { maximumFractionDigits: 4 });
      const entryPrice = p.entry_price < 1 
        ? `$${p.entry_price.toFixed(4)}` 
        : `$${p.entry_price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
      const currentPrice = p.current_price < 1 
        ? `$${p.current_price.toFixed(4)}` 
        : `$${p.current_price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

      return `
        <tr>
          <td><span class="symbol-badge">${p.symbol}</span></td>
          <td><span class="badge ${p.asset_class === 'CRYPTO' ? 'badge-info' : 'badge-warning'}">${p.asset_class}</span></td>
          <td class="mono">${qtyStr}</td>
          <td class="mono">${entryPrice}</td>
          <td class="mono">${currentPrice}</td>
          <td class="mono" style="color: ${pnlColor}; font-weight: 600;">
            ${p.unrealized_pnl >= 0 ? '+' : ''}$${p.unrealized_pnl.toFixed(2)} (${p.unrealized_pnl_pct.toFixed(2)}%)
          </td>
        </tr>
      `;
    }).join('');
  } catch (err) {
    console.debug('Positions error:', err);
  }
}

// Fetch Executed Orders
async function fetchOrders() {
  try {
    const res = await fetch('/api/orders');
    if (!res.ok) return;
    const orders = await res.json();

    const tbody = document.getElementById('orders-table-body');
    if (!tbody) return;

    if (!Array.isArray(orders) || orders.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted); padding: 32px;">No hay órdenes registradas.</td></tr>`;
      return;
    }

    tbody.innerHTML = orders.slice(0, 20).map(o => {
      const isBuy = (o.side || '').toUpperCase() === 'BUY';
      const sideColor = isBuy ? 'var(--accent-success)' : 'var(--accent-danger)';
      const sideBadge = `<span class="badge" style="background: ${isBuy ? 'var(--accent-success-subtle)' : 'var(--accent-danger-subtle)'}; color: ${sideColor}; border-color: ${isBuy ? 'rgba(52, 211, 153, 0.25)' : 'rgba(248, 113, 113, 0.25)'};">${o.side}</span>`;
      const price = o.filled_price 
        ? `$${o.filled_price < 1 ? o.filled_price.toFixed(4) : o.filled_price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` 
        : (o.limit_price ? `$${o.limit_price}` : 'MKT');
      const timeStr = o.created_at ? new Date(o.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '-';
      const qtyStr = o.quantity < 0.01 
        ? o.quantity.toFixed(6) 
        : o.quantity.toLocaleString('en-US', { maximumFractionDigits: 4 });

      return `
        <tr>
          <td class="mono" style="color: var(--text-muted); font-size: 11px;">${(o.order_id || '').slice(0, 8)}</td>
          <td><span class="symbol-badge">${o.symbol}</span></td>
          <td>${sideBadge}</td>
          <td class="mono">${qtyStr}</td>
          <td class="mono">${price}</td>
          <td><span class="badge ${o.status === 'FILLED' ? 'badge-success' : 'badge-info'}">${o.status}</span></td>
          <td class="mono" style="color: var(--text-muted); font-size: 11px;">${timeStr}</td>
        </tr>
      `;
    }).join('');
  } catch (err) {
    console.debug('Orders error:', err);
  }
}

// Fetch Watchlist
async function fetchWatchlist() {
  try {
    const res = await fetch('/api/watchlist');
    if (!res.ok) return;
    const tickers = await res.json();
    const container = document.getElementById('watchlist-container');
    if (container && Array.isArray(tickers) && tickers.length > 0) {
      container.innerHTML = tickers.map(t => `<span class="tag-item">${t}</span>`).join('');
    }
  } catch (err) {
    console.debug('Watchlist note:', err);
  }
}

// Terminal Logging Stream
async function fetchLogs() {
  try {
    const res = await fetch(`/api/logs?filter=${currentTab}`);
    if (!res.ok) return;
    const logs = await res.json();

    const container = document.getElementById('terminal-logs');
    if (!container) return;

    if (!Array.isArray(logs) || logs.length === 0) {
      container.innerHTML = `<div class="log-entry"><span class="log-time">[${new Date().toLocaleTimeString()}]</span><span class="log-info">Esperando eventos del motor autónomo...</span></div>`;
      return;
    }

    container.innerHTML = logs.map(line => {
      let cssClass = 'log-info';
      if (line.includes('[ERROR]') || line.includes('❌') || line.includes('REJECTED')) {
        cssClass = 'log-error';
      } else if (line.includes('[WARNING]') || line.includes('⚠️')) {
        cssClass = 'log-warning';
      }

      return `<div class="log-entry"><span class="${cssClass}">${line}</span></div>`;
    }).join('');

    container.scrollTop = container.scrollHeight;
  } catch (err) {
    console.debug('Logs note:', err);
  }
}

function switchTerminalTab(tab) {
  currentTab = tab;
  document.querySelectorAll('.terminal-tab').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  fetchLogs();
}

function clearTerminal() {
  const container = document.getElementById('terminal-logs');
  if (container) container.innerHTML = '';
}

// Full Scan & Kill Switch Triggers
async function triggerFullScan() {
  try {
    const btn = document.querySelector('.header-btn-scan');
    if (btn) btn.innerText = 'Escaneando...';
    await fetch('/api/scan/trigger', { method: 'POST' });
    setTimeout(() => {
      if (btn) btn.innerText = 'Escanear';
      fetchLogs();
      fetchPositions();
    }, 2000);
  } catch (_) {}
}

async function triggerKillSwitch() {
  if (confirm('¿Detener el motor autónomo inmediatamente?')) {
    try {
      await fetch('/api/kill-switch', { method: 'POST' });
      alert('Motor detenido con éxito.');
      window.location.reload();
    } catch (_) {}
  }
}

// Notification Drawer & Modals
function toggleNotificationsModal() {
  const modal = document.getElementById('notifications-modal');
  if (modal) {
    modal.style.display = modal.style.display === 'none' ? 'flex' : 'none';
  }
}

async function fetchNotifications() {
  try {
    const res = await fetch('/api/notifications');
    if (!res.ok) return;
    const notifs = await res.json();
    const countBadge = document.getElementById('notif-badge-count');
    const mobCount = document.getElementById('mobile-badge-count');
    if (countBadge) countBadge.innerText = (notifs || []).length;
    if (mobCount) mobCount.innerText = (notifs || []).length;

    const feed = document.getElementById('notifications-feed');
    if (feed && Array.isArray(notifs)) {
      if (notifs.length === 0) {
        feed.innerHTML = '<div style="text-align: center; color: var(--text-muted); padding: 16px;">Sin notificaciones recientes</div>';
        return;
      }
      feed.innerHTML = notifs.map(n => `
        <div style="padding: 10px; background: #141416; border: 1px solid var(--border-subtle); border-radius: 4px; margin-bottom: 6px;">
          <div style="font-weight: 600; font-size: 12px; color: #ffffff;">${n.title}</div>
          <div style="font-size: 11px; color: var(--text-secondary); margin-top: 2px;">${n.message}</div>
          <div style="font-size: 10px; color: var(--text-muted); margin-top: 4px; font-family: var(--font-mono);">${new Date(n.timestamp).toLocaleTimeString()}</div>
        </div>
      `).join('');
    }
  } catch (_) {}
}

function checkNotificationPermissionUI() {
  if ('Notification' in window && Notification.permission === 'granted') {
    const btn = document.getElementById('btn-request-push');
    if (btn) btn.innerText = 'Activo';
  }
}

function enablePushNotifications() {
  if ('Notification' in window) {
    Notification.requestPermission().then(permission => {
      if (permission === 'granted') {
        alert('Notificaciones push activadas con éxito.');
        checkNotificationPermissionUI();
      }
    });
  }
}

function toggleAudioAlerts(enabled) {
  localStorage.setItem('TRADINGAGENTS_AUDIO_ENABLED', enabled ? '1' : '0');
}

// Mobile View Switcher
function switchMobileView(view) {
  document.querySelectorAll('.mobile-nav-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.mobile-view-pane').forEach(p => p.classList.remove('active'));

  const navBtn = document.getElementById(`mob-nav-${view}`);
  if (navBtn) navBtn.classList.add('active');

  const pane = document.getElementById(`view-${view}`);
  if (pane) pane.classList.add('active');
}
