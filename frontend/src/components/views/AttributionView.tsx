import React, { useState, useEffect, useMemo } from 'react';
import type { Scenario, AttributionWeights, SarDriftPayload } from '../../types/dashboard';
import type { LiveSimulationResult } from '../../services/marineWeatherService';
import { SCENARIOS } from '../../data/scenarios';

interface AttributionViewProps {
  currentScenario?: Scenario | null;
  onSelectScenario?: (key: string) => void;
  onSelectTab?: (tab: any) => void;
  onInspectVesselOnMap?: (vessel: { name: string; lat: number; lng: number; mmsi: string }) => void;
  sarDriftPayload?: SarDriftPayload | null;
  activeDriftSimulation?: LiveSimulationResult | null;
}

interface CandidateVesselItem {
  mmsi: string;
  imo: string;
  name: string;
  flag: string;
  type: string;
  lat: number;
  lng: number;
  sog: number;
  cog: number;
  cpa_nm: number;
  ais_gap_hours: number;
  time_delta_hours?: number;
  attribution_score: number;
  risk: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
  dark_vessel_suspect?: boolean;
  trajectory?: Array<{
    lat: number;
    lng: number;
    sog?: number;
    cog?: number;
    timestamp?: number;
    time_iso?: string;
  }>;
  metrics?: {
    spatial_match_pct: number;
    temporal_alignment_pct: number;
    dark_gap_suspicion_pct: number;
    vessel_risk_prior_pct: number;
    weighted_dist_val?: number;
    weighted_time_val?: number;
    weighted_gap_val?: number;
    weighted_type_val?: number;
  };
}

const DEFAULT_SCENARIO_VESSELS: Record<string, CandidateVesselItem[]> = {
  'INC-2026-001': [
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
      time_delta_hours: 0.35,
      attribution_score: 0.88,
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
      time_delta_hours: 1.80,
      attribution_score: 0.65,
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
      time_delta_hours: 3.40,
      attribution_score: 0.38,
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
      time_delta_hours: 5.60,
      attribution_score: 0.21,
      risk: 'LOW',
    },
    {
      mmsi: '419009112',
      imo: '9856231',
      name: 'AL-ZUBARA',
      flag: 'Qatar',
      type: 'LNG Carrier',
      lat: 18.910,
      lng: 71.380,
      sog: 16.5,
      cog: 170.0,
      cpa_nm: 12.5,
      ais_gap_hours: 0.05,
      time_delta_hours: 7.20,
      attribution_score: 0.15,
      risk: 'LOW',
    },
  ],
  'INC-2026-002': [
    {
      mmsi: '419009988',
      imo: '9321456',
      name: 'PACIFIC GLORY',
      flag: 'India',
      type: 'Heavy Bunker Fuel Carrier',
      lat: 13.262,
      lng: 80.472,
      sog: 6.8,
      cog: 14.5,
      cpa_nm: 1.8,
      ais_gap_hours: 3.50,
      time_delta_hours: 0.40,
      attribution_score: 0.79,
      risk: 'CRITICAL',
    },
    {
      mmsi: '563001889',
      imo: '9812457',
      name: 'CHENNAI TRADER',
      flag: 'Singapore',
      type: 'Bulk Carrier',
      lat: 13.310,
      lng: 80.520,
      sog: 14.1,
      cog: 25.0,
      cpa_nm: 5.4,
      ais_gap_hours: 0.80,
      time_delta_hours: 2.50,
      attribution_score: 0.42,
      risk: 'MEDIUM',
    },
    {
      mmsi: '419004455',
      imo: '9512399',
      name: 'COROMANDEL PEARL',
      flag: 'India',
      type: 'Coastal Feeder Container',
      lat: 13.360,
      lng: 80.570,
      sog: 12.0,
      cog: 32.0,
      cpa_nm: 7.8,
      ais_gap_hours: 1.20,
      time_delta_hours: 4.10,
      attribution_score: 0.35,
      risk: 'MEDIUM',
    },
    {
      mmsi: '354001289',
      imo: '9678123',
      name: 'EASTERN FALCON',
      flag: 'Panama',
      type: 'Container Carrier',
      lat: 13.410,
      lng: 80.620,
      sog: 17.5,
      cog: 45.0,
      cpa_nm: 10.2,
      ais_gap_hours: 0.10,
      time_delta_hours: 5.80,
      attribution_score: 0.18,
      risk: 'LOW',
    },
  ],
  'INC-2026-003': [
    {
      mmsi: '419999000',
      imo: 'UNKNOWN',
      name: 'UNKNOWN (DARK SHIP)',
      flag: 'Unflagged / Blackout',
      type: 'Unidentified Tanker (SAR Echo)',
      lat: 10.460,
      lng: 93.130,
      sog: 8.5,
      cog: 295.0,
      cpa_nm: 0.9,
      ais_gap_hours: 14.20,
      time_delta_hours: 0.15,
      attribution_score: 0.92,
      risk: 'CRITICAL',
    },
    {
      mmsi: '636018992',
      imo: '9425112',
      name: 'MALACCA EXPLORER',
      flag: 'Liberia',
      type: 'Crude Oil Tanker',
      lat: 10.510,
      lng: 93.210,
      sog: 13.2,
      cog: 310.0,
      cpa_nm: 6.2,
      ais_gap_hours: 1.20,
      time_delta_hours: 2.20,
      attribution_score: 0.48,
      risk: 'MEDIUM',
    },
    {
      mmsi: '563009881',
      imo: '9781234',
      name: 'SUMATRA PRIDE',
      flag: 'Singapore',
      type: 'Bulk Carrier',
      lat: 10.580,
      lng: 93.290,
      sog: 11.5,
      cog: 315.0,
      cpa_nm: 8.5,
      ais_gap_hours: 0.30,
      time_delta_hours: 4.00,
      attribution_score: 0.28,
      risk: 'LOW',
    },
    {
      mmsi: '356002144',
      imo: '9845678',
      name: 'ANDAMAN LEADER',
      flag: 'Panama',
      type: 'Container Ship',
      lat: 10.640,
      lng: 93.360,
      sog: 19.0,
      cog: 320.0,
      cpa_nm: 11.0,
      ais_gap_hours: 0.20,
      time_delta_hours: 6.20,
      attribution_score: 0.16,
      risk: 'LOW',
    },
  ],
  'INC-2026-004': [
    {
      mmsi: '419003322',
      imo: '9245123',
      name: 'SEA PEARL',
      flag: 'India',
      type: 'Bunkering Barge / Coastal Tanker',
      lat: 15.428,
      lng: 73.655,
      sog: 2.1,
      cog: 172.0,
      cpa_nm: 0.8,
      ais_gap_hours: 2.50,
      time_delta_hours: 0.25,
      attribution_score: 0.84,
      risk: 'CRITICAL',
    },
    {
      mmsi: '419006543',
      imo: '9356124',
      name: 'MORMUGAO MARINER',
      flag: 'India',
      type: 'Bulk Carrier',
      lat: 15.485,
      lng: 73.710,
      sog: 11.4,
      cog: 185.0,
      cpa_nm: 4.2,
      ais_gap_hours: 0.40,
      time_delta_hours: 2.10,
      attribution_score: 0.45,
      risk: 'MEDIUM',
    },
    {
      mmsi: '419008765',
      imo: '9411234',
      name: 'MANDOVI VOYAGER',
      flag: 'Panama',
      type: 'Chemical Tanker',
      lat: 15.520,
      lng: 73.760,
      sog: 13.8,
      cog: 192.0,
      cpa_nm: 6.8,
      ais_gap_hours: 0.20,
      time_delta_hours: 3.80,
      attribution_score: 0.28,
      risk: 'LOW',
    },
    {
      mmsi: '419009999',
      imo: '9123456',
      name: 'ZUARI TIDE',
      flag: 'India',
      type: 'Fishing Trawler',
      lat: 15.580,
      lng: 73.820,
      sog: 7.2,
      cog: 210.0,
      cpa_nm: 9.5,
      ais_gap_hours: 0.10,
      time_delta_hours: 5.50,
      attribution_score: 0.18,
      risk: 'LOW',
    },
  ],
  'INC-2026-005': [
    {
      mmsi: '419008811',
      imo: '9512399',
      name: 'AL KHALEEJ STAR',
      flag: 'India',
      type: 'Crude Oil Tanker (VLCC)',
      lat: 22.615,
      lng: 69.450,
      sog: 3.2,
      cog: 75.0,
      cpa_nm: 1.1,
      ais_gap_hours: 3.8,
      attribution_score: 0.78,
      risk: 'HIGH',
    },
    {
      mmsi: '419007733',
      imo: '9488112',
      name: 'SAURASHTRA PRIDE',
      flag: 'India',
      type: 'Aframax Crude Tanker',
      lat: 22.625,
      lng: 69.380,
      sog: 12.1,
      cog: 82.0,
      cpa_nm: 4.8,
      ais_gap_hours: 0.8,
      attribution_score: 0.44,
      risk: 'MEDIUM',
    },
  ],
  'INC-2026-006': [
    {
      mmsi: '419004455',
      imo: '9398812',
      name: 'OCEAN VOYAGER',
      flag: 'India',
      type: 'Product Tanker (Aframax)',
      lat: 10.020,
      lng: 76.050,
      sog: 1.8,
      cog: 162.0,
      cpa_nm: 1.4,
      ais_gap_hours: 3.1,
      attribution_score: 0.72,
      risk: 'HIGH',
    },
    {
      mmsi: '419005522',
      imo: '9432109',
      name: 'MALABAR PIONEER',
      flag: 'Panama',
      type: 'Container Ship',
      lat: 10.150,
      lng: 75.980,
      sog: 14.6,
      cog: 158.0,
      cpa_nm: 7.2,
      ais_gap_hours: 0.5,
      attribution_score: 0.38,
      risk: 'LOW',
    },
  ],
  'INC-2026-007': [
    {
      mmsi: '419006677',
      imo: '9456781',
      name: 'EASTERN GLORY',
      flag: 'India',
      type: 'Crude Oil Tanker (VLCC)',
      lat: 20.280,
      lng: 86.780,
      sog: 2.4,
      cog: 35.0,
      cpa_nm: 1.2,
      ais_gap_hours: 2.8,
      attribution_score: 0.76,
      risk: 'HIGH',
    },
    {
      mmsi: '419007711',
      imo: '9389922',
      name: 'KALINGA VOYAGER',
      flag: 'Liberia',
      type: 'Bulk Carrier',
      lat: 20.350,
      lng: 86.850,
      sog: 11.8,
      cog: 40.0,
      cpa_nm: 6.5,
      ais_gap_hours: 0.6,
      attribution_score: 0.35,
      risk: 'LOW',
    },
  ],
  'INC-2026-008': [
    {
      mmsi: '636019944',
      imo: '9511200',
      name: 'PACIFIC ORCHID',
      flag: 'Liberia',
      type: 'Crude Oil Tanker (VLCC)',
      lat: 8.520,
      lng: 72.850,
      sog: 13.5,
      cog: 95.0,
      cpa_nm: 2.1,
      ais_gap_hours: 2.2,
      attribution_score: 0.68,
      risk: 'MEDIUM',
    },
  ],
};

export const AttributionView: React.FC<AttributionViewProps> = ({
  currentScenario,
  onSelectScenario,
  onSelectTab,
  onInspectVesselOnMap,
  sarDriftPayload,
  activeDriftSimulation,
}) => {
  const defaultKey = currentScenario?.id.includes('002')
    ? 'INC-002'
    : currentScenario?.id.includes('003')
    ? 'INC-003'
    : currentScenario?.id.includes('004')
    ? 'INC-004'
    : currentScenario?.id.includes('005')
    ? 'INC-005'
    : currentScenario?.id.includes('006')
    ? 'INC-006'
    : currentScenario?.id.includes('007')
    ? 'INC-007'
    : currentScenario?.id.includes('008')
    ? 'INC-008'
    : 'INC-001';

  const [selectedKey, setSelectedKey] = useState<string>(defaultKey);
  const activeScenario = SCENARIOS[selectedKey] || currentScenario || SCENARIOS['INC-001'];

  const [weights, setWeights] = useState<AttributionWeights>({
    dist: 0.30,
    time: 0.25,
    gap: 0.25,
    type: 0.20,
  });

  const [loading, setLoading] = useState(false);
  const [evidence, setEvidence] = useState<any>(null);

  const runPhysicsEngine = async () => {
    setLoading(true);
    try {
      const payload = {
        detection: {
          timestamp: new Date().toISOString(),
          sensor_id: "S1A_IW_GRDH",
          bounding_box: {
            min_lon: 71.218, min_lat: 18.743, max_lon: 71.220, max_lat: 18.745
          },
          confidence_score: 0.94,
          estimated_length_m: 220,
          estimated_beam_m: 32,
          heading_deg: 145
        },
        ais_history: [
          {
            mmsi: "419001234",
            timestamp: new Date(Date.now() - 4 * 3600000).toISOString(),
            lon: 71.180, lat: 18.710,
            sog_knots: 4.1, cog_deg: 140,
            vessel_length_m: 215, vessel_beam_m: 30,
            vessel_type: "Crude Oil Tanker"
          }
        ]
      };

      const res = await fetch("/api/v1/evidence/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (res.ok && res.headers.get("content-type")?.includes("application/json")) {
        const data = await res.json();
        setEvidence(data);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  type WeightKey = 'dist' | 'time' | 'gap' | 'type';

  const [autoBalance, setAutoBalance] = useState<boolean>(true);
  const [isCalibrationCollapsed, setIsCalibrationCollapsed] = useState<boolean>(false);

  const handleWeightChange = (changedKey: WeightKey, rawValue: number) => {
    const clampedVal = Math.max(0.0, Math.min(1.0, +(rawValue.toFixed(2))));

    if (!autoBalance) {
      setWeights((prev) => ({
        ...prev,
        [changedKey]: clampedVal,
      }));
      return;
    }

    const oldWeights = { ...weights };
    const remainingBudget = Math.max(0, +( (1.0 - clampedVal).toFixed(2) ));
    const otherKeys = (['dist', 'time', 'gap', 'type'] as WeightKey[]).filter((k) => k !== changedKey);
    const sumOthers = otherKeys.reduce((acc, k) => acc + (oldWeights[k] || 0), 0);

    const newWeights: Record<WeightKey, number> = {
      ...oldWeights,
      [changedKey]: clampedVal,
    };

    if (sumOthers > 0.001) {
      otherKeys.forEach((k) => {
        newWeights[k] = +((oldWeights[k] / sumOthers) * remainingBudget).toFixed(2);
      });
    } else {
      otherKeys.forEach((k) => {
        newWeights[k] = +(remainingBudget / otherKeys.length).toFixed(2);
      });
    }

    // Fix floating point precision so sum is strictly 1.00
    const currentSum = Object.values(newWeights).reduce((a, b) => a + b, 0);
    const delta = +( (1.0 - currentSum).toFixed(2) );
    if (Math.abs(delta) > 0.001 && otherKeys.length > 0) {
      newWeights[otherKeys[0]] = Math.max(0, +( (newWeights[otherKeys[0]] + delta).toFixed(2) ));
    }

    setWeights(newWeights);
  };


  const [aishubUsername, setAishubUsername] = useState<string>(() => {
    return localStorage.getItem('AISHUB_USERNAME') || '';
  });

  const [expandedMap, setExpandedMap] = useState<Record<string, boolean>>({});

  const toggleExpand = (mmsi: string, currentExpanded: boolean) => {
    setExpandedMap((prev) => ({
      ...prev,
      [mmsi]: !currentExpanded,
    }));
  };
  const [showLegalDetails, setShowLegalDetails] = useState<boolean>(false);
  const [showFeedConfig, setShowFeedConfig] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [feedSource, setFeedSource] = useState<string>('AISStream.io WebSocket');
  const [lastUpdated, setLastUpdated] = useState<string>('Live Calibrated Feed');

  const [streamTelemetry, setStreamTelemetry] = useState<{
    status: string;
    total_live_vessels_tracked: number;
    service: string;
    vessels?: any[];
  } | null>(null);
  const [historicalCandidates, setHistoricalCandidates] = useState<CandidateVesselItem[]>([]);

  const computeDistanceNm = (lat1: number, lon1: number, lat2: number, lon2: number): number => {
    if (!lat1 || !lon1 || !lat2 || !lon2) return 10.0;
    const R = 3440.065; // Earth radius in nautical miles
    const dLat = ((lat2 - lat1) * Math.PI) / 180;
    const dLon = ((lon2 - lon1) * Math.PI) / 180;
    const a =
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos((lat1 * Math.PI) / 180) * Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return Math.round(R * c * 10) / 10;
  };

  const fetchLiveAIS = async () => {
    setIsLoading(true);
    try {
      // 1. Query 4D Historical Trajectory Buffer for discharge origin envelope
      const targetOriginLat = activeDriftSimulation?.originCoordNumeric?.[0]
        ?? (sarDriftPayload?.lat ? sarDriftPayload.lat - 0.08 : (activeScenario?.lat ? activeScenario.lat - 0.08 : 18.663));
      const targetOriginLng = activeDriftSimulation?.originCoordNumeric?.[1]
        ?? (sarDriftPayload?.lng ? sarDriftPayload.lng - 0.06 : (activeScenario?.lng ? activeScenario.lng - 0.06 : 71.158));

      try {
        const histRes = await fetch(
          `/api/v1/ais/historical-candidates?incident_id=${encodeURIComponent(incidentId)}&origin_lat=${targetOriginLat}&origin_lng=${targetOriginLng}&horizon_hours=24`
        );
        if (histRes.ok && histRes.headers.get('content-type')?.includes('application/json')) {
          const hData = await histRes.json();
          if (hData?.candidates && hData.candidates.length > 0) {
            setHistoricalCandidates(hData.candidates);
            setFeedSource('4D Historical Trajectory Buffer (Indexed Intercept)');
          }
        }
      } catch {
        // silent fallback
      }

      const lat = activeScenario?.lat || 18.743;
      const lng = activeScenario?.lng || 71.218;
      const userParam = aishubUsername ? `&username=${encodeURIComponent(aishubUsername)}` : '';

      const liveRes = await fetch(`/api/v1/ais/live-vessels?lat=${lat}&lng=${lng}&radius_nm=60${userParam}`);
      const isLiveJson = liveRes.ok && liveRes.headers.get('content-type')?.includes('application/json');

      if (isLiveJson) {
        const lData = await liveRes.json();
        if (lData?.vessels) {
          setStreamTelemetry({
            status: lData.status === 'LIVE_STREAMING' ? 'ONLINE' : (lData.status || 'ONLINE'),
            total_live_vessels_tracked: lData.count || lData.vessels.length,
            service: lData.service || 'AISStream.io Indian EEZ WebSocket',
            vessels: lData.vessels,
          });
          setFeedSource(lData.source || 'AISStream.io Live Indian EEZ WebSocket');
          setLastUpdated(new Date().toLocaleTimeString());
          return;
        }
      }

      // Fallback: check stream-status endpoint
      const statusRes = await fetch('/api/v1/ais/stream-status');
      const isStatusJson = statusRes.ok && statusRes.headers.get('content-type')?.includes('application/json');
      if (isStatusJson) {
        const sData = await statusRes.json();
        setStreamTelemetry(sData);
        if (sData?.service) {
          setFeedSource(sData.service);
        }
        setLastUpdated(new Date().toLocaleTimeString());
        return;
      }

      // Graceful offline fallback: keep calibrated baseline
      setLastUpdated(new Date().toLocaleTimeString());
    } catch (err: any) {
      if (err?.name !== 'AbortError') {
        console.info('AIS operating in calibrated scenario mode');
      }
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchLiveAIS();
    const interval = setInterval(fetchLiveAIS, 10000);
    return () => clearInterval(interval);
  }, [activeScenario, aishubUsername]);

  // Synchronize when currentScenario changes from external sources
  useEffect(() => {
    if (currentScenario?.id) {
      const matchedKey = Object.keys(SCENARIOS).find(
        (k) => SCENARIOS[k].id === currentScenario.id || currentScenario.id.includes(k.replace('INC-', ''))
      );
      if (matchedKey) setSelectedKey(matchedKey);
    }
  }, [currentScenario]);

  const incidentId = activeScenario.id;

  // Compute total weights sum dynamically
  const totalW = useMemo(() => {
    return +( (weights.dist || 0) + (weights.time || 0) + (weights.gap || 0) + (weights.type || 0) ).toFixed(2) || 1.0;
  }, [weights]);

  // Normalized weight percentages for Bayesian prior breakdown
  const normDist = Math.round(((weights.dist || 0) / totalW) * 100);
  const normTime = Math.round(((weights.time || 0) / totalW) * 100);
  const normGap = Math.round(((weights.gap || 0) / totalW) * 100);
  const normType = Math.round(((weights.type || 0) / totalW) * 100);

  // Compute attribution score based on active sensitivity weights
  const computeVesselScore = (
    v: CandidateVesselItem,
    w: AttributionWeights
  ): { score: number; risk: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW'; metrics: any } => {
    const sumW = (w.dist || 0) + (w.time || 0) + (w.gap || 0) + (w.type || 0) || 1.0;

    // Use realistic distance to incident
    const cpa = (v.cpa_nm !== undefined && v.cpa_nm > 0)
      ? v.cpa_nm
      : (v.lat && v.lng && activeScenario?.lat && activeScenario?.lng)
      ? computeDistanceNm(v.lat, v.lng, activeScenario.lat, activeScenario.lng)
      : 12.0;

    // Spatial score (closer CPA -> higher suspicion: 0nm -> 1.0, 15nm -> 0.0)
    const sDist = Math.max(0.0, Math.min(1.0, 1.0 - cpa / 15.0));

    // Time alignment score (closer to reverse discharge window -> higher suspicion: 0h -> 1.0, 6h -> 0.0)
    const timeDelta = v.time_delta_hours ?? (cpa < 2 ? 0.35 : cpa < 5 ? 1.8 : 4.5);
    const sTime = Math.max(0.05, Math.min(1.0, 1.0 - timeDelta / 6.0));

    // Dark Ship AIS silence gap (larger gap -> higher suspicion: >=4h -> 1.0)
    const gapHours = v.ais_gap_hours ?? 0;
    const sGap = Math.max(0.0, Math.min(1.0, gapHours / 4.0));

    // Vessel risk prior based on ship category
    const t = (v.type || '').toLowerCase();
    const sType = t.includes('crude') || t.includes('dark') || t.includes('vlcc')
      ? 0.95
      : t.includes('product') || t.includes('bunker') || t.includes('aframax')
      ? 0.85
      : t.includes('chemical')
      ? 0.70
      : t.includes('cargo') || t.includes('container') || t.includes('bulk')
      ? 0.35
      : 0.20;

    const weightedDist = (w.dist * sDist) / sumW;
    const weightedTime = (w.time * sTime) / sumW;
    const weightedGap = (w.gap * sGap) / sumW;
    const weightedType = (w.type * sType) / sumW;

    const finalScore = weightedDist + weightedTime + weightedGap + weightedType;
    const rounded = Math.round(Math.min(1.0, Math.max(0.05, finalScore)) * 100) / 100;

    const risk: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW' =
      rounded >= 0.75 ? 'CRITICAL' : rounded >= 0.50 ? 'HIGH' : rounded >= 0.32 ? 'MEDIUM' : 'LOW';

    return {
      score: rounded,
      risk,
      metrics: {
        spatial_match_pct: Math.round(sDist * 100),
        temporal_alignment_pct: Math.round(sTime * 100),
        dark_gap_suspicion_pct: Math.round(sGap * 100),
        vessel_risk_prior_pct: Math.round(sType * 100),
        weighted_dist_val: +weightedDist.toFixed(2),
        weighted_time_val: +weightedTime.toFixed(2),
        weighted_gap_val: +weightedGap.toFixed(2),
        weighted_type_val: +weightedType.toFixed(2),
      },
    };
  };

  // FORENSIC INCIDENT SUSPECTS (For Active Spill AOI)
  const candidates: CandidateVesselItem[] = useMemo(() => {
    // Scenario-specific suspects: prioritize authentic 4D historical trajectory candidates from backend
    const scenarioSuspects = (historicalCandidates.length > 0)
      ? historicalCandidates.map(v => ({ ...v }))
      : (DEFAULT_SCENARIO_VESSELS[incidentId] || DEFAULT_SCENARIO_VESSELS['INC-2026-001'] || []).map(v => ({ ...v }));

    // Target discharge coordinates: preferentially use the dynamic reverse-drift origin coordinate (T-24h)!
    const targetOriginLat = activeDriftSimulation?.originCoordNumeric?.[0]
      ?? (sarDriftPayload?.lat ? sarDriftPayload.lat - 0.08 : (activeScenario?.lat ? activeScenario.lat - 0.08 : 18.663));
    const targetOriginLng = activeDriftSimulation?.originCoordNumeric?.[1]
      ?? (sarDriftPayload?.lng ? sarDriftPayload.lng - 0.06 : (activeScenario?.lng ? activeScenario.lng - 0.06 : 71.158));

    const liveNearbyVessels: CandidateVesselItem[] = [];
    if (streamTelemetry?.vessels && streamTelemetry.vessels.length > 0) {
      streamTelemetry.vessels.forEach((v) => {
        // Measure CPA relative to the reconstructed discharge origin envelope!
        const dist = computeDistanceNm(v.lat, v.lng, targetOriginLat, targetOriginLng);
        // Include vessels within 45 nm of this origin envelope
        if (dist <= 45.0) {
          // Avoid duplicate MMSIs if already in scenarioSuspects
          if (!scenarioSuspects.some(s => s.mmsi === String(v.mmsi))) {
            const vType = v.type || 'Commercial Vessel';
            liveNearbyVessels.push({
              mmsi: String(v.mmsi),
              imo: String(v.imo || 'N/A'),
              name: v.name || `LIVE-${v.mmsi}`,
              flag: v.flag || 'Indian Waters',
              type: vType,
              lat: v.lat,
              lng: v.lng,
              sog: typeof v.sog === 'number' ? Math.round(v.sog * 10) / 10 : 0,
              cog: typeof v.cog === 'number' ? Math.round(v.cog) : 0,
              cpa_nm: dist,
              ais_gap_hours: 0.1,
              time_delta_hours: 0.5,
              attribution_score: 0.5,
              risk: 'MEDIUM',
            });
          }
        }
      });
    }

    const combinedList = [...scenarioSuspects, ...liveNearbyVessels];

    return combinedList
      .map((v) => {
        const { score, risk, metrics } = computeVesselScore(v, weights);
        return {
          ...v,
          attribution_score: score,
          risk,
          metrics,
        };
      })
      .sort((a, b) => b.attribution_score - a.attribution_score);
  }, [incidentId, weights, streamTelemetry, activeScenario, activeDriftSimulation, sarDriftPayload]);

  const criticalCount = useMemo(
    () => candidates.filter((c) => c.risk === 'CRITICAL' || c.attribution_score >= 0.75).length,
    [candidates]
  );
  const highRiskCount = useMemo(
    () => candidates.filter((c) => c.risk === 'HIGH' || (c.attribution_score >= 0.5 && c.attribution_score < 0.75)).length,
    [candidates]
  );


  const handleRecalculate = () => {
    fetchLiveAIS();
  };

  const handleSaveUsername = (uname: string) => {
    setAishubUsername(uname);
    localStorage.setItem('AISHUB_USERNAME', uname);
  };

  const isCustomPreset = !(
    (weights.dist === 0.30 && weights.time === 0.25 && weights.gap === 0.25 && weights.type === 0.20) ||
    (weights.dist === 0.20 && weights.time === 0.15 && weights.gap === 0.45 && weights.type === 0.20) ||
    (weights.dist === 0.50 && weights.time === 0.20 && weights.gap === 0.10 && weights.type === 0.20)
  );
  return (
    <div id="tab-attribution" className="tab-content visible modern-dashboard-root">
      {/* 1. EXECUTIVE HEADER */}
      <div className="workspace-header-bar">
        <div>
          <h1 className="workspace-main-title">Vessel Attribution &amp; AIS Sensitivity Tuner</h1>
          <p className="workspace-sub-title">
            Suspect Attribution · Correlating AIS Vessel Tracks with Reverse Oil Drift Origin
          </p>
        </div>

        <div className="workspace-header-actions">
          <select
            value={selectedKey}
            onChange={(e) => {
              const k = e.target.value;
              setSelectedKey(k);
              if (SCENARIOS[k]) {
                onSelectScenario?.(k);
              }
            }}
            className="action-pill-btn secondary"
            style={{
              padding: '6px 14px',
              fontWeight: 600,
              cursor: 'pointer',
              appearance: 'auto',
            }}
          >
            {Object.entries(SCENARIOS).map(([key, sc]) => (
              <option key={key} value={key}>
                {sc.id} · {sc.title}
              </option>
            ))}
          </select>

          {isCustomPreset && (
            <button
              className="action-pill-btn secondary"
              onClick={() => setWeights({ dist: 0.30, time: 0.25, gap: 0.25, type: 0.20 })}
              title="Reset sensitivity weights to Balanced ML preset"
            >
              <span className="material-symbols-outlined" style={{ fontSize: 16 }}>restart_alt</span>
              <span>Reset Matrix</span>
            </button>
          )}

          <button
            className="action-pill-btn primary"
            onClick={handleRecalculate}
            disabled={isLoading}
            title="Poll live Indian EEZ transponder data and recalculate rankings"
          >
            <span
              className="material-symbols-outlined"
              style={{
                fontSize: 16,
                animation: isLoading ? 'spin 1s linear infinite' : 'none',
              }}
            >
              sync
            </span>
            <span>{isLoading ? 'Syncing...' : 'Sync Live AIS & Rerank'}</span>
          </button>
        </div>
      </div>

      {/* 2. EXECUTIVE METRIC CARDS */}
      <div className="executive-metrics-grid">
        <div className="metric-card-neumorphic">
          <div className="metric-card-header">
            <span className="metric-card-label">Screened Targets</span>
          </div>
          <div className="metric-card-body">
            <span className="metric-number">
              {candidates.length} <span className="metric-unit">Vessels</span>
            </span>
            <span className="metric-trend-pill neutral">
              T - 72h Window
            </span>
          </div>
          <div className="metric-card-footer">
            <span>{criticalCount} Critical · {highRiskCount} Moderate Suspicion</span>
            <span className="material-symbols-outlined arrow-icon">radar</span>
          </div>
        </div>

        <div className="metric-card-neumorphic">
          <div className="metric-card-header">
            <span className="metric-card-label">Primary Suspect Attribution</span>
          </div>
          <div className="metric-card-body">
            <span className="metric-number" style={{ fontSize: 17, color: '#ef4444' }}>
              {candidates[0]?.name || 'CRUDE ATLAS'}
            </span>
            <span className="metric-trend-pill positive" style={{ color: '#ef4444' }}>
              {(candidates[0]?.attribution_score ?? 0.88).toFixed(2)} Score
            </span>
          </div>
          <div className="metric-card-footer">
            <span>MMSI: {candidates[0]?.mmsi || '419001234'} · Flag: {candidates[0]?.flag || 'India'}</span>
            <span className="material-symbols-outlined arrow-icon">crisis_alert</span>
          </div>
        </div>

        <div className="metric-card-neumorphic">
          <div className="metric-card-header">
            <span className="metric-card-label">Surveillance AOI Envelope</span>
          </div>
          <div className="metric-card-body">
            <span className="metric-number">
              35 <span className="metric-unit">nm Radius</span>
            </span>
            <span className="metric-trend-pill positive">
              Reverse Kinematics
            </span>
          </div>
          <div className="metric-card-footer">
            <span>AOI: {activeScenario.title.toUpperCase()}</span>
            <span className="material-symbols-outlined arrow-icon">track_changes</span>
          </div>
        </div>

        <div className="metric-card-neumorphic">
          <div className="metric-card-header">
            <span className="metric-card-label">AIS Surveillance Feed</span>
          </div>
          <div className="metric-card-body">
            <span className="metric-number" style={{ color: streamTelemetry?.status === 'ONLINE' ? '#10b981' : 'inherit' }}>
              AISStream <span className="metric-unit">{streamTelemetry?.status === 'ONLINE' ? 'Live' : 'Standby'}</span>
            </span>
            <span className="metric-trend-pill neutral" style={{ color: streamTelemetry?.status === 'ONLINE' ? '#10b981' : undefined }}>
              {streamTelemetry?.status === 'ONLINE' ? 'Active EEZ Sync' : 'Simulated Feed'}
            </span>
          </div>
          <div className="metric-card-footer">
            <span>{candidates.length} Target Tracks Correlated · Indian EEZ</span>
            <span className="material-symbols-outlined arrow-icon">satellite_alt</span>
          </div>
        </div>
      </div>

      {/* 3. WORKFLOW NAV BAR */}
      <div className="workflow-nav-bar">
        <div className="workflow-title-area">
          <h2 className="workflow-title">Sensitivity Presets</h2>
          <span
            className="scenario-chip"
            style={{
              borderColor: streamTelemetry?.status === 'ONLINE' ? 'rgba(16, 185, 129, 0.4)' : 'rgba(56, 189, 248, 0.4)',
              color: streamTelemetry?.status === 'ONLINE' ? '#10b981' : 'var(--accent)',
            }}
            title={`Active Telemetry Feed: ${feedSource}`}
          >
            <span
              className="status-dot dot-live"
              style={{
                width: 6,
                height: 6,
                background: streamTelemetry?.status === 'ONLINE' ? '#10b981' : 'var(--accent)',
              }}
            />
            {streamTelemetry?.status === 'ONLINE' ? (feedSource.includes('AISStream') ? 'AISStream Live EEZ' : 'AISHub Relay') : 'AIS Standby'}
          </span>
        </div>

        <div className="workflow-tabs-strip">
          <button
            type="button"
            className={`workflow-tab-btn ${weights.dist === 0.30 && weights.time === 0.25 && weights.gap === 0.25 && weights.type === 0.20 ? 'active' : ''}`}
            onClick={() => setWeights({ dist: 0.30, time: 0.25, gap: 0.25, type: 0.20 })}
            title="Balanced ML Attribution (Distance 30%, Timing 25%, Blackout Gap 25%, Cargo Risk 20%)"
          >
            Balanced (30/25/25/20)
          </button>
          <button
            type="button"
            className={`workflow-tab-btn ${weights.dist === 0.20 && weights.time === 0.15 && weights.gap === 0.45 && weights.type === 0.20 ? 'active' : ''}`}
            onClick={() => setWeights({ dist: 0.20, time: 0.15, gap: 0.45, type: 0.20 })}
            title="Focus on Dark Ships with long AIS blackout gaps (45% Gap Weight)"
          >
            Dark Ship Focus
          </button>
          <button
            type="button"
            className={`workflow-tab-btn ${weights.dist === 0.50 && weights.time === 0.20 && weights.gap === 0.10 && weights.type === 0.20 ? 'active' : ''}`}
            onClick={() => setWeights({ dist: 0.50, time: 0.20, gap: 0.10, type: 0.20 })}
            title="Focus on Anchorage Proximity to spill drift origin (50% Distance Weight)"
          >
            Anchorage Focus
          </button>
          {isCustomPreset && (
            <button
              type="button"
              className="workflow-tab-btn active"
              style={{ borderColor: 'var(--accent)', color: 'var(--accent)' }}
              title="Custom Sensitivity Matrix Active: Click to reset to Balanced ML preset"
              onClick={() => setWeights({ dist: 0.30, time: 0.25, gap: 0.25, type: 0.20 })}
            >
              <span>✦ Custom ({(weights.dist * 100).toFixed(0)}/{(weights.time * 100).toFixed(0)}/{(weights.gap * 100).toFixed(0)}/{(weights.type * 100).toFixed(0)})</span>
              <span style={{ fontSize: 10, opacity: 0.75, textDecoration: 'underline', marginLeft: 2 }}>Reset</span>
            </button>
          )}
        </div>

        <div className="workflow-nav-actions">
          <div style={{ position: 'relative' }}>
            <button
              type="button"
              className={`action-pill-btn ${showFeedConfig ? 'primary' : 'secondary'}`}
              style={{ padding: '5px 11px', fontSize: 11, display: 'inline-flex', alignItems: 'center', gap: 5, whiteSpace: 'nowrap' }}
              onClick={() => setShowFeedConfig((prev) => !prev)}
              title="Configure AIS Stream Feed credentials and endpoints"
            >
              <span className="material-symbols-outlined" style={{ fontSize: 15 }}>settings_input_antenna</span>
              <span>AIS Feed Config</span>
            </button>

            {showFeedConfig && (
              <div
                style={{
                  position: 'absolute',
                  top: '125%',
                  right: 0,
                  width: 280,
                  background: 'var(--bg-surface, #1e293b)',
                  border: '1px solid var(--border-default, rgba(255,255,255,0.15))',
                  borderRadius: 10,
                  boxShadow: '0 10px 25px rgba(0,0,0,0.4)',
                  padding: '12px 14px',
                  zIndex: 100,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 8,
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--border-subtle)', paddingBottom: 6 }}>
                  <span style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-primary)' }}>AIS Stream Provider</span>
                  <span className="status-dot dot-live" style={{ width: 7, height: 7 }} />
                </div>

                <div>
                  <label style={{ fontSize: 10, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>
                    AISHub Account Username (Optional API Relay)
                  </label>
                  <div style={{ display: 'flex', gap: 6 }}>
                    <input
                      type="text"
                      placeholder="Enter AISHub Username..."
                      value={aishubUsername}
                      onChange={(e) => handleSaveUsername(e.target.value)}
                      style={{
                        fontSize: 11,
                        padding: '5px 8px',
                        borderRadius: 6,
                        border: '1px solid var(--border-subtle)',
                        background: 'var(--bg-raised)',
                        color: 'var(--text-primary)',
                        flex: 1,
                        outline: 'none',
                      }}
                    />
                    <button
                      type="button"
                      className="action-pill-btn secondary"
                      style={{ padding: '4px 8px', fontSize: 10 }}
                      onClick={() => setShowFeedConfig(false)}
                    >
                      Done
                    </button>
                  </div>
                </div>

                <div style={{ fontSize: 9.5, color: 'var(--text-secondary)', lineHeight: 1.3 }}>
                  Primary: <strong>AISStream.io WebSocket</strong> (Indian Ocean EEZ). Fallback: <strong>AISHub API</strong>.
                </div>
              </div>
            )}
          </div>

          <button
            type="button"
            className={`action-pill-btn ${isCalibrationCollapsed ? 'primary' : 'secondary'}`}
            style={{ padding: '5px 11px', fontSize: 11, display: 'inline-flex', alignItems: 'center', gap: 5, whiteSpace: 'nowrap' }}
            onClick={() => setIsCalibrationCollapsed((prev) => !prev)}
            title={isCalibrationCollapsed ? 'Show sensitivity calibration pane' : 'Collapse calibration sliders to maximize suspect vessels list'}
          >
            <span className="material-symbols-outlined" style={{ fontSize: 15 }}>
              {isCalibrationCollapsed ? 'tune' : 'fullscreen'}
            </span>
            <span>{isCalibrationCollapsed ? 'Show Sliders' : 'Focus Suspects'}</span>
          </button>
        </div>
      </div>

      {/* 4. ROUNDED CANVAS CONTAINER */}
      <div className="canvas-rounded-container">
        <div className="canvas-two-column" style={isCalibrationCollapsed ? { gridTemplateColumns: '1fr' } : undefined}>
          {/* LEFT PANE: SENSITIVITY WEIGHT TUNERS & ML PRIORS */}
          {!isCalibrationCollapsed && (
            <div className="canvas-pane">
              <div className="pane-header">
                <span className="pane-title">
                  <span className="material-symbols-outlined" style={{ fontSize: 18, color: 'var(--accent)' }}>tune</span>
                  Attribution Calibration Matrix
                </span>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <button
                    type="button"
                    className={`metric-trend-pill ${autoBalance ? 'positive' : 'neutral'}`}
                    style={{
                      fontSize: 10,
                      fontWeight: 700,
                      cursor: 'pointer',
                      border: 'none',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 4,
                    }}
                    onClick={() => setAutoBalance((prev) => !prev)}
                    title="When Auto-Balance is ON, adjusting one slider automatically scales remaining weights to sum cleanly to 1.00"
                  >
                    <span className="material-symbols-outlined" style={{ fontSize: 12 }}>balance</span>
                    <span>Auto-Balance: {autoBalance ? 'ON' : 'OFF'}</span>
                  </button>
                  <span className="metric-trend-pill positive" style={{ fontSize: 10, fontWeight: 700 }}>
                    Σ wi = {totalW.toFixed(2)}
                  </span>
                  <button
                    type="button"
                    style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 2, display: 'flex', color: 'var(--text-muted)' }}
                    title="Collapse Calibration Sliders"
                    onClick={() => setIsCalibrationCollapsed(true)}
                  >
                    <span className="material-symbols-outlined" style={{ fontSize: 16 }}>close</span>
                  </button>
                </div>
              </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              {/* Slider 1: Proximity to Slick Origin */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 11, fontWeight: 700, marginBottom: 6 }}>
                  <span style={{ color: 'var(--text-primary)' }}>Proximity to Slick Origin (Spatial)</span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>({normDist}%)</span>
                    <span className="mono" style={{ color: 'var(--accent)', minWidth: 32, textAlign: 'right' }}>{weights.dist.toFixed(2)}</span>
                  </div>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={weights.dist}
                  className="macos-slider"
                  style={{
                    background: `linear-gradient(to right, var(--accent) 0%, var(--accent) ${Math.min(100, weights.dist * 100)}%, var(--border-default) ${Math.min(100, weights.dist * 100)}%, var(--border-default) 100%)`
                  }}
                  onChange={(e) => handleWeightChange('dist', parseFloat(e.target.value))}
                />
                <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
                  Distance match relative to reverse-drift slick origin.
                </div>
              </div>

              {/* Slider 2: Discharge Window Timing */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 11, fontWeight: 700, marginBottom: 6 }}>
                  <span style={{ color: 'var(--text-primary)' }}>Discharge Window Timing (Temporal)</span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>({normTime}%)</span>
                    <span className="mono" style={{ color: 'var(--accent)', minWidth: 32, textAlign: 'right' }}>{weights.time.toFixed(2)}</span>
                  </div>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={weights.time}
                  className="macos-slider"
                  style={{
                    background: `linear-gradient(to right, var(--accent) 0%, var(--accent) ${Math.min(100, weights.time * 100)}%, var(--border-default) ${Math.min(100, weights.time * 100)}%, var(--border-default) 100%)`
                  }}
                  onChange={(e) => handleWeightChange('time', parseFloat(e.target.value))}
                />
                <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
                  Temporal correlation within the estimated discharge window.
                </div>
              </div>

              {/* Slider 3: Transponder Blackout Severity */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 11, fontWeight: 700, marginBottom: 6 }}>
                  <span style={{ color: 'var(--text-primary)' }}>Transponder Blackout Severity (Dark Gap)</span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>({normGap}%)</span>
                    <span className="mono" style={{ color: '#f59e0b', minWidth: 32, textAlign: 'right' }}>{weights.gap.toFixed(2)}</span>
                  </div>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={weights.gap}
                  className="macos-slider"
                  style={{
                    background: `linear-gradient(to right, #f59e0b 0%, #f59e0b ${Math.min(100, weights.gap * 100)}%, var(--border-default) ${Math.min(100, weights.gap * 100)}%, var(--border-default) 100%)`
                  }}
                  onChange={(e) => handleWeightChange('gap', parseFloat(e.target.value))}
                />
                <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
                  Flags deliberate AIS shutdowns or transponder gaps inside AOI.
                </div>
              </div>

              {/* Slider 4: Cargo Hazard Risk Profile */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 11, fontWeight: 700, marginBottom: 6 }}>
                  <span style={{ color: 'var(--text-primary)' }}>Cargo Hazard Risk Profile (Vessel Prior)</span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>({normType}%)</span>
                    <span className="mono" style={{ color: '#8b5cf6', minWidth: 32, textAlign: 'right' }}>{weights.type.toFixed(2)}</span>
                  </div>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={weights.type}
                  className="macos-slider"
                  style={{
                    background: `linear-gradient(to right, #8b5cf6 0%, #8b5cf6 ${Math.min(100, weights.type * 100)}%, var(--border-default) ${Math.min(100, weights.type * 100)}%, var(--border-default) 100%)`
                  }}
                  onChange={(e) => handleWeightChange('type', parseFloat(e.target.value))}
                />
                <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
                  Pollution hazard prior: VLCC &gt; Aframax &gt; Chemical Tanker &gt; Cargo.
                </div>
              </div>
            </div>

            {/* NORMALIZATION FORMULA CARD */}
            <div
              style={{
                background: 'var(--bg-raised)',
                padding: '12px 14px',
                borderRadius: 10,
                border: '1px solid var(--border-subtle)',
                marginTop: 4,
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                <span style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-primary)' }}>
                  Kinematic Normalization Formula:
                </span>
                <span className="mono" style={{ fontSize: 10, color: 'var(--accent)', fontWeight: 700 }}>
                  Σ(w) = {totalW.toFixed(2)}
                </span>
              </div>
              <div className="mono" style={{ fontSize: 11, color: 'var(--accent)', background: 'var(--bg-base)', padding: '8px 10px', borderRadius: 6, lineHeight: 1.4 }}>
                Score = ({weights.dist.toFixed(2)}·S_dist + {weights.time.toFixed(2)}·S_time + {weights.gap.toFixed(2)}·S_gap + {weights.type.toFixed(2)}·S_type) / {totalW.toFixed(2)}
              </div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 8, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span>Effective Bayesian Weights:</span>
                <span className="mono" style={{ color: 'var(--text-secondary)' }}>
                  {normDist}% Dist · {normTime}% Time · {normGap}% Gap · {normType}% Type
                </span>
              </div>
              <div style={{ display: 'flex', height: 4, borderRadius: 2, overflow: 'hidden', marginTop: 6, background: 'var(--border-subtle)' }}>
                <div style={{ width: `${normDist}%`, background: 'var(--accent)' }} title={`Proximity: ${normDist}%`} />
                <div style={{ width: `${normTime}%`, background: '#10b981' }} title={`Time: ${normTime}%`} />
                <div style={{ width: `${normGap}%`, background: '#f59e0b' }} title={`Silence Gap: ${normGap}%`} />
                <div style={{ width: `${normType}%`, background: '#8b5cf6' }} title={`Vessel Prior: ${normType}%`} />
              </div>
            </div>

            {/* LEGAL EVIDENCE FRAMEWORK */}
            <div
              style={{
                background: 'rgba(37, 99, 235, 0.06)',
                borderLeft: '4px solid var(--accent)',
                padding: '10px 12px',
                borderRadius: 8,
                fontSize: 11,
                color: 'var(--text-secondary)',
                lineHeight: 1.4,
                marginTop: 10,
              }}
            >
              <div
                style={{
                  fontWeight: 700,
                  color: 'var(--text-primary)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  cursor: 'pointer',
                }}
                onClick={() => setShowLegalDetails((prev) => !prev)}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span className="material-symbols-outlined" style={{ fontSize: 16, color: 'var(--accent)' }}>gavel</span>
                  <span>Legal Admissibility Framework</span>
                </div>
                <span style={{ fontSize: 10, color: 'var(--accent)', textDecoration: 'underline' }}>
                  {showLegalDetails ? 'Hide' : 'Details'}
                </span>
              </div>
              <div style={{ marginTop: 4 }}>
                Under <strong>MARPOL 73/78 Annex I</strong>, kinematic correlation establishes prima-facie evidence for ICG action.
              </div>
              {showLegalDetails && (
                <div style={{ marginTop: 6, paddingTop: 6, borderTop: '1px dashed var(--border-subtle)', fontSize: 10.5, color: 'var(--text-muted)' }}>
                  Complies with Section 356 of Merchant Shipping Act 1958. Spatiotemporal intersection of AIS trajectories with reverse OpenDrift envelopes qualifies for Indian Coast Guard detention warrants.
                </div>
              )}
            </div>

            {/* DYNAMIC DRIFT PHYSICS CORRELATION TEST */}
            <div
              style={{
                background: 'var(--bg-raised, rgba(255,255,255,0.02))',
                border: '1px solid var(--border-color, rgba(255,255,255,0.08))',
                padding: '10px 12px',
                borderRadius: 8,
                marginTop: 10,
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                <div style={{ fontWeight: 700, fontSize: 11.5, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span className="material-symbols-outlined" style={{ fontSize: 15, color: '#38bdf8' }}>science</span>
                  <span>Hydrodynamic Drift &amp; AIS Verification</span>
                </div>
                <button
                  className="action-pill-btn primary"
                  onClick={runPhysicsEngine}
                  disabled={loading}
                  style={{ padding: '2px 8px', fontSize: 10 }}
                >
                  {loading ? 'Evaluating...' : 'Run Physics Engine'}
                </button>
              </div>
              <p style={{ fontSize: 10.5, color: 'var(--text-muted)', margin: 0 }}>
                Correlates SAR envelope centroid with reverse AIS kinematic trajectory.
              </p>

              {evidence && (
                <div style={{ padding: '8px 10px', background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 6, marginTop: 8, fontSize: 11, fontFamily: 'monospace' }}>
                  <div style={{ color: '#10b981', fontWeight: 600, marginBottom: 4 }}>✓ Backend Evaluation Complete:</div>
                  <div style={{ color: 'var(--text-secondary)' }}>Status: <span style={{ color: '#fff' }}>{evidence.classification || 'CORRELATED'}</span></div>
                  {evidence.drift_applied?.delta_x_m !== undefined && (
                    <div style={{ color: 'var(--text-secondary)' }}>Drift Vector: <span style={{ color: '#38bdf8' }}>dx: {evidence.drift_applied.delta_x_m.toFixed(1)}m, dy: {evidence.drift_applied.delta_y_m.toFixed(1)}m</span></div>
                  )}
                  {evidence.confidence?.overall_confidence !== undefined && (
                    <div style={{ color: 'var(--text-secondary)' }}>Confidence Score: <span style={{ color: '#f59e0b' }}>{(evidence.confidence.overall_confidence * 100).toFixed(1)}%</span></div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

          {/* RIGHT PANE: RANKED CANDIDATE VESSELS */}
          <div className="canvas-pane">
            <div className="pane-header">
              <span className="pane-title">
                <span className="material-symbols-outlined" style={{ fontSize: 18, color: 'var(--accent)' }}>format_list_numbered</span>
                Ranked Suspect Vessels (AOI Spatiotemporal Envelope)
              </span>
              <span className="text-xs text-muted">
                Updated: {lastUpdated}
              </span>
            </div>

            {/* When calibration pane is collapsed, show active weights banner */}
            {isCalibrationCollapsed && (
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '7px 12px',
                  background: 'rgba(37, 99, 235, 0.08)',
                  borderRadius: 8,
                  marginBottom: 10,
                  border: '1px solid rgba(56, 189, 248, 0.3)',
                  fontSize: 11,
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--accent)' }}>
                  <span className="material-symbols-outlined" style={{ fontSize: 15 }}>tune</span>
                  <span>Active Calibration: <strong>{normDist}% Spatial · {normTime}% Temporal · {normGap}% Dark Gap · {normType}% Cargo Prior</strong> (Σ=1.00)</span>
                </div>
                <button
                  type="button"
                  className="action-pill-btn secondary"
                  style={{ padding: '2px 8px', fontSize: 10 }}
                  onClick={() => setIsCalibrationCollapsed(false)}
                >
                  Adjust Sliders
                </button>
              </div>
            )}

            {/* AISStream Live Surveillance Status Banner */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '8px 12px', background: 'var(--bg-raised)', borderRadius: 8, marginBottom: 12, border: '1px solid var(--border-subtle)', flexWrap: 'wrap', gap: 8 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span className={`status-dot ${streamTelemetry?.status === 'ONLINE' ? 'dot-live' : 'dot-ready'}`} style={{ width: 8, height: 8 }} />
                <span style={{ fontSize: 11, fontWeight: 700, color: streamTelemetry?.status === 'ONLINE' ? '#10B981' : 'var(--text-muted)' }}>
                  {streamTelemetry?.status === 'ONLINE' ? 'AISStream Indian EEZ Feed: Active' : 'Calibrated AIS Surveillance: Active'}
                </span>
                <span style={{ fontSize: 10, background: 'rgba(16, 185, 129, 0.15)', color: '#10B981', padding: '1px 8px', borderRadius: 12, fontWeight: 700 }}>
                  {candidates.length} Target Vessels in Zone ({criticalCount} Flagged)
                </span>
              </div>

              <div style={{ fontSize: 11, color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 4 }}>
                <span className="material-symbols-outlined" style={{ fontSize: 14, color: 'var(--accent)' }}>radar</span>
                <span>Spill AOI Envelope: <strong>35 nm Radius</strong></span>
              </div>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {candidates.map((v, idx) => {
                const riskClass = v.risk.toLowerCase();
                const scoreColor = v.attribution_score >= 0.75 ? '#ef4444' : v.attribution_score >= 0.5 ? '#f59e0b' : '#10b981';
                const isExpanded = expandedMap[v.mmsi] !== undefined ? expandedMap[v.mmsi] : idx === 0;

                return (
                  <div
                    key={`${v.mmsi}-${idx}`}
                    className={`vessel-candidate-card ${riskClass}`}
                    style={{
                      cursor: 'pointer',
                      transition: 'all 0.2s ease',
                      border: isExpanded ? '1px solid var(--accent)' : '1px solid var(--border-subtle)',
                      padding: isExpanded ? '12px 14px' : '8px 12px',
                    }}
                    onClick={() => toggleExpand(v.mmsi, isExpanded)}
                  >
                    {/* Header Row */}
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                        <div
                          style={{
                            width: 24,
                            height: 24,
                            borderRadius: '50%',
                            background: idx === 0 ? 'rgba(239, 68, 68, 0.15)' : 'var(--bg-raised)',
                            color: idx === 0 ? '#ef4444' : 'var(--text-muted)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontWeight: 800,
                            fontSize: 12,
                          }}
                        >
                          {idx + 1}
                        </div>
                        <div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            <span style={{ fontSize: 13, fontWeight: 700, color: idx === 0 ? '#ef4444' : 'var(--text-primary)' }}>
                              {v.name}
                            </span>
                            <span
                              className="metric-trend-pill"
                              style={{
                                fontSize: 9,
                                padding: '1px 6px',
                                background: v.risk === 'CRITICAL' ? 'rgba(239, 68, 68, 0.12)' : v.risk === 'HIGH' ? 'rgba(249, 115, 22, 0.12)' : 'rgba(56, 189, 248, 0.12)',
                                color: v.risk === 'CRITICAL' ? '#ef4444' : v.risk === 'HIGH' ? '#f97316' : '#38bdf8',
                              }}
                            >
                              {v.risk} SUSPICION
                            </span>
                          </div>

                          {/* Collapsed summary preview */}
                          {!isExpanded && (
                            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2, display: 'flex', gap: 8 }}>
                              <span>CPA: <strong>{v.cpa_nm} nm</strong></span>
                              <span>•</span>
                              <span>SOG: <strong>{v.sog} kn</strong></span>
                              <span>•</span>
                              <span>Gap: <strong style={{ color: (v.ais_gap_hours || 0) > 2 ? '#f59e0b' : 'inherit' }}>{v.ais_gap_hours || 0}h</strong></span>
                            </div>
                          )}
                        </div>
                      </div>

                      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                        <div style={{ textAlign: 'right' }}>
                          <div style={{ fontSize: 18, fontWeight: 800, color: scoreColor, fontFamily: 'monospace' }}>
                            {v.attribution_score.toFixed(2)}
                          </div>
                          <div style={{ fontSize: 9, color: 'var(--text-muted)' }}>Attribution Score</div>
                        </div>

                        <button
                          type="button"
                          style={{
                            background: 'none',
                            border: 'none',
                            padding: 4,
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            color: 'var(--text-muted)',
                          }}
                          onClick={(e) => {
                            e.stopPropagation();
                            toggleExpand(v.mmsi, isExpanded);
                          }}
                          title={isExpanded ? 'Collapse candidate details' : 'Expand candidate details'}
                        >
                          <span
                            className="material-symbols-outlined"
                            style={{
                              fontSize: 20,
                              transform: isExpanded ? 'rotate(180deg)' : 'none',
                              transition: 'transform 0.2s',
                            }}
                          >
                            expand_more
                          </span>
                        </button>
                      </div>
                    </div>

                    {/* Expanded Details */}
                    {isExpanded && (
                      <div
                        style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 6 }}
                        onClick={(e) => e.stopPropagation()}
                      >
                        <div style={{ fontSize: 10.5, color: 'var(--text-secondary)' }}>
                          MMSI: <span className="mono">{v.mmsi}</span> · IMO: <span className="mono">{v.imo}</span> · Flag: {v.flag} · {v.type}
                        </div>

                        <div className="candidate-telemetry-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 6, fontSize: 10, background: 'var(--bg-raised)', padding: '6px 8px', borderRadius: 6 }}>
                          <div>
                            <span className="text-muted">CPA: </span>
                            <strong>{v.cpa_nm} nm</strong>
                          </div>
                          <div>
                            <span className="text-muted">SOG: </span>
                            <strong>{v.sog} kn</strong>
                          </div>
                          <div>
                            <span className="text-muted">Heading: </span>
                            <strong>{v.cog}°</strong>
                          </div>
                          <div style={{ color: (v.ais_gap_hours || 0) > 2.0 ? '#f59e0b' : 'inherit' }}>
                            <span className="text-muted">AIS Gap: </span>
                            <strong>{v.ais_gap_hours || 0}h</strong>
                          </div>
                        </div>

                        {v.metrics && (
                          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 2 }}>
                            <div className="candidate-metrics-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 6, fontSize: 9.5 }}>
                              <div>
                                <div style={{ color: 'var(--text-muted)', marginBottom: 2, display: 'flex', justifyContent: 'space-between' }}>
                                  <span>Spatial</span>
                                  <span className="mono" style={{ color: 'var(--accent)' }}>+{v.metrics.weighted_dist_val}</span>
                                </div>
                                <div style={{ height: 4, borderRadius: 2, background: 'var(--border-subtle)', overflow: 'hidden' }}>
                                  <div style={{ width: `${v.metrics.spatial_match_pct}%`, height: '100%', background: 'var(--accent)' }} />
                                </div>
                                <span className="mono" style={{ fontSize: 9 }}>{v.metrics.spatial_match_pct}%</span>
                              </div>
                              <div>
                                <div style={{ color: 'var(--text-muted)', marginBottom: 2, display: 'flex', justifyContent: 'space-between' }}>
                                  <span>Time</span>
                                  <span className="mono" style={{ color: '#10b981' }}>+{v.metrics.weighted_time_val}</span>
                                </div>
                                <div style={{ height: 4, borderRadius: 2, background: 'var(--border-subtle)', overflow: 'hidden' }}>
                                  <div style={{ width: `${v.metrics.temporal_alignment_pct}%`, height: '100%', background: '#10b981' }} />
                                </div>
                                <span className="mono" style={{ fontSize: 9 }}>{v.metrics.temporal_alignment_pct}%</span>
                              </div>
                              <div>
                                <div style={{ color: 'var(--text-muted)', marginBottom: 2, display: 'flex', justifyContent: 'space-between' }}>
                                  <span>Dark Gap</span>
                                  <span className="mono" style={{ color: '#f59e0b' }}>+{v.metrics.weighted_gap_val}</span>
                                </div>
                                <div style={{ height: 4, borderRadius: 2, background: 'var(--border-subtle)', overflow: 'hidden' }}>
                                  <div style={{ width: `${v.metrics.dark_gap_suspicion_pct}%`, height: '100%', background: '#f59e0b' }} />
                                </div>
                                <span className="mono" style={{ fontSize: 9 }}>{v.metrics.dark_gap_suspicion_pct}%</span>
                              </div>
                              <div>
                                <div style={{ color: 'var(--text-muted)', marginBottom: 2, display: 'flex', justifyContent: 'space-between' }}>
                                  <span>Prior</span>
                                  <span className="mono" style={{ color: '#8b5cf6' }}>+{v.metrics.weighted_type_val}</span>
                                </div>
                                <div style={{ height: 4, borderRadius: 2, background: 'var(--border-subtle)', overflow: 'hidden' }}>
                                  <div style={{ width: `${v.metrics.vessel_risk_prior_pct}%`, height: '100%', background: '#8b5cf6' }} />
                                </div>
                                <span className="mono" style={{ fontSize: 9 }}>{v.metrics.vessel_risk_prior_pct}%</span>
                              </div>
                            </div>

                            <div style={{ display: 'flex', height: 4, borderRadius: 2, overflow: 'hidden', background: 'var(--border-subtle)' }} title={`Weighted breakdown: Spatial (+${v.metrics.weighted_dist_val}) + Time (+${v.metrics.weighted_time_val}) + Gap (+${v.metrics.weighted_gap_val}) + Prior (+${v.metrics.weighted_type_val}) = ${v.attribution_score.toFixed(2)}`}>
                              <div style={{ width: `${((v.metrics.weighted_dist_val || 0) / v.attribution_score) * 100}%`, background: 'var(--accent)' }} />
                              <div style={{ width: `${((v.metrics.weighted_time_val || 0) / v.attribution_score) * 100}%`, background: '#10b981' }} />
                              <div style={{ width: `${((v.metrics.weighted_gap_val || 0) / v.attribution_score) * 100}%`, background: '#f59e0b' }} />
                              <div style={{ width: `${((v.metrics.weighted_type_val || 0) / v.attribution_score) * 100}%`, background: '#8b5cf6' }} />
                            </div>
                          </div>
                        )}

                        {/* Operational Action Buttons Bar */}
                        <div style={{ display: 'flex', gap: 8, marginTop: 6, paddingTop: 6, borderTop: '1px solid var(--border-subtle)' }}>
                          <button
                            type="button"
                            className="action-pill-btn primary"
                            style={{ padding: '5px 12px', fontSize: 11, flex: 1, justifyContent: 'center' }}
                            onClick={() => {
                              onInspectVesselOnMap?.({
                                name: v.name,
                                lat: v.lat,
                                lng: v.lng,
                                mmsi: v.mmsi,
                              });
                            }}
                            title="Center tactical map on vessel's estimated track"
                          >
                            <span className="material-symbols-outlined" style={{ fontSize: 14 }}>explore</span>
                            <span>Inspect Track on Map</span>
                          </button>

                          <button
                            type="button"
                            className="action-pill-btn secondary"
                            style={{ padding: '5px 12px', fontSize: 11, flex: 1, justifyContent: 'center' }}
                            onClick={() => {
                              onSelectTab?.('evidence');
                            }}
                            title="Generate MARPOL Annex I Legal Evidence Dossier"
                          >
                            <span className="material-symbols-outlined" style={{ fontSize: 14 }}>gavel</span>
                            <span>MARPOL Dossier</span>
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
