export default async function handler(req, res) {
  try {
    const renderRes = await fetch('https://bot-trader-dear.onrender.com/health');
    const data = await renderRes.json();
    return res.status(200).json({
      status: 'ok',
      keepalive: true,
      timestamp: new Date().toISOString(),
      render: data
    });
  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
}
