// Vercel Serverless Function — AIS Live Telemetry (Alias)
// Endpoint: GET /api/v1/ais/live

import liveVesselsHandler from './live-vessels.js';

export default function handler(req, res) {
  return liveVesselsHandler(req, res);
}
