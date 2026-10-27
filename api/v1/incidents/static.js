// Vercel Serverless Function — Static Incidents Catalog
// Endpoint: GET /api/v1/incidents/static

export default function handler(req, res) {
  res.setHeader('Content-Type', 'application/json');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  const incidents = [
    {
      id: 'INC-2026-001',
      title: 'Mumbai High North Slick',
      lat: 18.743,
      lng: 71.218,
      severity: 'CRITICAL',
      oil_type: 'Heavy Crude (Bonn IV)',
      oil_color: '#991B1B',
      area: '14.2 km²',
      top_vessel: 'CRUDE ATLAS',
      attribution_score: 0.88,
    },
    {
      id: 'INC-2026-002',
      title: 'Kochi Channel Bunker Leak',
      lat: 9.965,
      lng: 76.221,
      severity: 'HIGH',
      oil_type: 'HFO 380 (Bonn III)',
      oil_color: '#B45309',
      area: '6.8 km²',
      top_vessel: 'PACIFIC RUBY',
      attribution_score: 0.74,
    },
    {
      id: 'INC-2026-003',
      title: 'Paradip Offshore Dispersion',
      lat: 20.245,
      lng: 86.69,
      severity: 'MEDIUM',
      oil_type: 'Light Crude (Bonn II)',
      oil_color: '#1D4ED8',
      area: '3.1 km²',
      top_vessel: 'MARITIME VOYAGER',
      attribution_score: 0.61,
    },
    {
      id: 'INC-2026-004',
      title: 'Gulf of Kutch Anchor Discharge',
      lat: 22.48,
      lng: 69.15,
      severity: 'LOW',
      oil_type: 'Diesel / MGO (Bonn I)',
      oil_color: '#047857',
      area: '0.9 km²',
      top_vessel: 'OCEAN PIONEER',
      attribution_score: 0.42,
    },
  ];

  return res.status(200).json(incidents);
}
