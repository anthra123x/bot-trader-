export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', '*');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  const token = process.env.ALPACA_ACCESS_TOKEN || '042eba5a-ce10-43a1-8143-291cfd2998b6';

  try {
    const alpacaRes = await fetch('https://paper-api.alpaca.markets/v2/positions', {
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      }
    });

    if (!alpacaRes.ok) {
      return res.status(200).json([]);
    }

    const raw = await alpacaRes.json();
    const positions = (Array.isArray(raw) ? raw : []).map(p => {
      const qty = parseFloat(p.qty || '0');
      const entryPrice = parseFloat(p.avg_entry_price || '0');
      const currentPrice = parseFloat(p.current_price || '0');
      const marketVal = parseFloat(p.market_value || '0');
      const pnl = parseFloat(p.unrealized_pl || '0');
      const pnlPct = parseFloat(p.unrealized_plpc || '0') * 100;

      return {
        symbol: p.symbol,
        asset_class: (p.asset_class || 'crypto').toUpperCase(),
        quantity: qty,
        entry_price: entryPrice,
        current_price: currentPrice,
        market_value: Math.round(marketVal * 100) / 100,
        unrealized_pnl: Math.round(pnl * 100) / 100,
        unrealized_pnl_pct: Math.round(pnlPct * 100) / 100
      };
    });

    return res.status(200).json(positions);
  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
}
