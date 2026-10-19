export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', '*');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  const token = process.env.ALPACA_ACCESS_TOKEN || '042eba5a-ce10-43a1-8143-291cfd2998b6';
  
  try {
    const alpacaRes = await fetch('https://paper-api.alpaca.markets/v2/account', {
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      }
    });

    if (!alpacaRes.ok) {
      // Fallback to Render if Alpaca returns error
      const renderRes = await fetch('https://bot-trader-dear.onrender.com/api/balance');
      const renderData = await renderRes.json();
      return res.status(200).json(renderData);
    }

    const data = await alpacaRes.json();
    const equity = parseFloat(data.equity || '0');
    const lastEquity = parseFloat(data.last_equity || equity);
    const cash = parseFloat(data.cash || '0');
    const buyingPower = parseFloat(data.buying_power || '0');
    const dailyPnl = equity - lastEquity;
    const dailyDd = lastEquity > 0 ? Math.max(0, ((lastEquity - equity) / lastEquity) * 100) : 0;

    return res.status(200).json({
      cash: Math.round(cash * 100) / 100,
      equity: Math.round(equity * 100) / 100,
      buying_power: Math.round(buyingPower * 100) / 100,
      daily_pnl: Math.round(dailyPnl * 100) / 100,
      daily_drawdown_pct: Math.round(dailyDd * 100) / 100,
      currency: data.currency || 'USD',
      is_simulation: false
    });
  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
}
