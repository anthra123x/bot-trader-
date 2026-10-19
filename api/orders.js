export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', '*');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  const token = process.env.ALPACA_ACCESS_TOKEN || '042eba5a-ce10-43a1-8143-291cfd2998b6';

  try {
    const alpacaRes = await fetch('https://paper-api.alpaca.markets/v2/orders?status=all&limit=50', {
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      }
    });

    if (!alpacaRes.ok) {
      return res.status(200).json([]);
    }

    const raw = await alpacaRes.json();
    const orders = (Array.isArray(raw) ? raw : []).map(o => {
      const filledPrice = o.filled_avg_price ? parseFloat(o.filled_avg_price) : null;
      const limitPrice = o.limit_price ? parseFloat(o.limit_price) : null;
      const qty = parseFloat(o.qty || '0');

      return {
        order_id: o.id || '',
        symbol: o.symbol || '',
        side: (o.side || 'buy').toUpperCase(),
        order_type: (o.type || 'market').toUpperCase(),
        quantity: qty,
        filled_price: filledPrice,
        limit_price: limitPrice,
        status: (o.status || 'filled').toUpperCase(),
        created_at: o.created_at || o.submitted_at || new Date().toISOString()
      };
    });

    return res.status(200).json(orders);
  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
}
