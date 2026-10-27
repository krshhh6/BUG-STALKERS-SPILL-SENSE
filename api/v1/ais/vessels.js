// Vercel Serverless Function — All AIS Vessels Cache
// Endpoint: GET /api/v1/ais/vessels

export default function handler(req, res) {
  res.setHeader('Content-Type', 'application/json');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  const vessels = [
    {
      mmsi: '419001234',
      imo: '9412345',
      name: 'CRUDE ATLAS',
      flag: 'India',
      type: 'Crude Oil Tanker (VLCC)',
      lat: 18.758,
      lng: 71.206,
      sog: 4.1,
      cog: 182.4,
      cpa_nm: 1.2,
      ais_gap_hours: 4.58,
      distScore: 0.94,
      timeScore: 0.90,
      gapScore: 0.88,
      typeScore: 0.95,
      risk: 'CRITICAL',
    },
    {
      mmsi: '419005678',
      imo: '9523456',
      name: 'MARITIME KOHISTAN',
      flag: 'India',
      type: 'Product Tanker (Aframax)',
      lat: 18.785,
      lng: 71.256,
      sog: 12.4,
      cog: 165.0,
      cpa_nm: 3.8,
      ais_gap_hours: 3.55,
      distScore: 0.72,
      timeScore: 0.68,
      gapScore: 0.65,
      typeScore: 0.85,
      risk: 'HIGH',
    },
    {
      mmsi: '419007890',
      imo: '9634567',
      name: 'GULF NAVIGATOR',
      flag: 'Panama',
      type: 'Chemical Tanker',
      lat: 18.658,
      lng: 71.309,
      sog: 13.7,
      cog: 178.2,
      cpa_nm: 7.1,
      ais_gap_hours: 1.37,
      distScore: 0.40,
      timeScore: 0.45,
      gapScore: 0.25,
      typeScore: 0.70,
      risk: 'LOW',
    },
    {
      mmsi: '352002144',
      imo: '9745120',
      name: 'ORIENTAL STAR',
      flag: 'Liberia',
      type: 'Container Cargo Vessel',
      lat: 18.863,
      lng: 71.143,
      sog: 18.2,
      cog: 190.1,
      cpa_nm: 9.4,
      ais_gap_hours: 0.15,
      distScore: 0.25,
      timeScore: 0.30,
      gapScore: 0.10,
      typeScore: 0.35,
      risk: 'LOW',
    },
  ];

  return res.status(200).json({
    status: 'OK',
    count: vessels.length,
    vessels,
  });
}
