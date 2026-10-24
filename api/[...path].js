// Vercel Serverless Function — Fallback JSON Handler
// Guarantees that unmatched /api/* requests return strict application/json and NEVER HTML <!DOCTYPE>

export default function handler(req, res) {
  res.setHeader('Content-Type', 'application/json');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  const reqPath = req.query?.path
    ? (Array.isArray(req.query.path) ? req.query.path.join('/') : req.query.path)
    : req.url || '';

  return res.status(404).json({
    status: 'NOT_FOUND',
    error: 'API endpoint not implemented on serverless proxy',
    path: reqPath,
    message: 'The requested API route is running in offline/calibrated edge mode.',
  });
}
