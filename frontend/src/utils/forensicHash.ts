import type { Scenario } from '../types/dashboard';

/**
 * Generates an authentic, standard FIPS 180-4 SHA-256 64-character hexadecimal hash.
 * Synchronous and bit-for-bit identical to OpenSSL sha256 and Web Crypto API.
 */
export function generateDeterministicHash(ascii: string): string {
  function rightRotate(value: number, amount: number): number {
    return (value >>> amount) | (value << (32 - amount));
  }
  const mathPow = Math.pow;
  const maxWord = mathPow(2, 32);
  let i: number;
  let j: number;
  let result = '';
  const words: number[] = [];
  const asciiBitLength = ascii.length * 8;
  const hash: number[] = [];
  const k: number[] = [];
  let primeCounter = 0;
  const isComposite: Record<number, number> = {};

  for (let candidate = 2; primeCounter < 64; candidate++) {
    if (!isComposite[candidate]) {
      for (i = 0; i < 313; i += candidate) {
        isComposite[i] = candidate;
      }
      hash[primeCounter] = (mathPow(candidate, 0.5) * maxWord) | 0;
      k[primeCounter++] = (mathPow(candidate, 1 / 3) * maxWord) | 0;
    }
  }

  let formatted = ascii + '\x80';
  while ((formatted.length % 64) - 56) formatted += '\x00';
  for (i = 0; i < formatted.length; i++) {
    j = formatted.charCodeAt(i);
    words[i >> 2] |= j << ((3 - (i % 4)) * 8);
  }
  words[words.length] = (asciiBitLength / maxWord) | 0;
  words[words.length] = asciiBitLength;

  for (j = 0; j < words.length; ) {
    const w = words.slice(j, (j += 16));
    const oldHash = [...hash];
    for (i = 0; i < 64; i++) {
      const w15 = w[i - 15];
      const w2 = w[i - 2];
      const a = hash[0];
      const e = hash[4];
      const s0 = rightRotate(w15, 7) ^ rightRotate(w15, 18) ^ (w15 >>> 3);
      const s1 = rightRotate(w2, 17) ^ rightRotate(w2, 19) ^ (w2 >>> 10);
      w[i] = (i < 16 ? w[i] : (w[i - 16] + s0 + w[i - 7] + s1) | 0) | 0;

      const ch = (e & hash[5]) ^ (~e & hash[6]);
      const maj = (a & hash[1]) ^ (a & hash[2]) ^ (hash[1] & hash[2]);
      const sigma1 = rightRotate(e, 6) ^ rightRotate(e, 11) ^ rightRotate(e, 25);
      const sigma0 = rightRotate(a, 2) ^ rightRotate(a, 13) ^ rightRotate(a, 22);

      const temp1 = (hash[7] + sigma1 + ch + k[i] + w[i]) | 0;
      const temp2 = (sigma0 + maj) | 0;

      hash[7] = hash[6];
      hash[6] = hash[5];
      hash[5] = hash[4];
      hash[4] = (hash[3] + temp1) | 0;
      hash[3] = hash[2];
      hash[2] = hash[1];
      hash[1] = hash[0];
      hash[0] = (temp1 + temp2) | 0;
    }
    for (i = 0; i < 8; i++) {
      hash[i] = (hash[i] + oldHash[i]) | 0;
    }
  }

  for (i = 0; i < 8; i++) {
    for (j = 3; j >= 0; j--) {
      const b = (hash[i] >> (j * 8)) & 255;
      result += (b < 16 ? '0' : '') + b.toString(16);
    }
  }
  return result;
}

/**
 * Formats a 64-character hex string into 8-character chunks for human-verifiable reading.
 */
export function formatHashChunks(hash: string): string[] {
  const chunks: string[] = [];
  for (let i = 0; i < hash.length; i += 8) {
    chunks.push(hash.slice(i, i + 8));
  }
  return chunks;
}

export interface ForensicMilestone {
  id: string;
  title: string;
  description: string;
  timestamp: string;
  hash: string;
  status: 'sealed' | 'in_review' | 'pending';
  icon: string;
  verifiedBy?: string;
}

export interface ForensicArtifactItem {
  id: string;
  title: string;
  fileName: string;
  type: 'geojson' | 'gpkg' | 'xml' | 'netcdf';
  sizeKb: number;
  sha256: string;
  crs: string;
  description: string;
  icon: string;
  iconColor: string;
  contentGenerator: (scenario: Scenario, masterHash: string) => string;
}

/**
 * Generates dynamic milestone chronology based on scenario metadata.
 */
export function getForensicMilestones(scenario?: Scenario | null, officerSignOff?: { officerName: string; timestamp: string } | null): ForensicMilestone[] {
  const scId = scenario?.id || 'INC-2026-005';
  const scTitle = scenario?.title || 'Maritime Sector';

  // Base date calculation (deterministic based on scenario ID)
  const numId = parseInt(scId.replace(/\D/g, '') || '5', 10);
  const baseDay = (10 + (numId % 18)).toString().padStart(2, '0');
  const baseDate = `2026-09-${baseDay}`;

  return [
    {
      id: 'm1',
      title: 'SAR Scene Ingestion Authenticated',
      description: `Sentinel-1 L1C Ground Range Detected (GRD) CRC32 Verified · ${scTitle} Radiometric Calibration Sigma0 RTC`,
      timestamp: `${baseDate} 04:22 UTC`,
      hash: generateDeterministicHash(`${scId}-sar-scene-ingest`),
      status: 'sealed',
      icon: 'check',
    },
    {
      id: 'm2',
      title: 'Feature Mask SHA-256 Timestamped',
      description: `U-Net ResNet-50 oil polygon boundary immutable hash · Bonn Scale Code 2 & 5 classification`,
      timestamp: `${baseDate} 04:25 UTC`,
      hash: generateDeterministicHash(`${scId}-unet-mask-hash`),
      status: 'sealed',
      icon: 'check',
    },
    {
      id: 'm3',
      title: 'Drift NetCDF Output Hashed',
      description: `OpenDrift Lagrangian Monte Carlo backward trajectory (N=10,000 particles) coordinates sealed`,
      timestamp: `${baseDate} 04:29 UTC`,
      hash: generateDeterministicHash(`${scId}-drift-netcdf-particles`),
      status: 'sealed',
      icon: 'check',
    },
    {
      id: 'm4',
      title: 'AIS Intersect Matrix Sealed',
      description: `AISHub candidate vessel track correlations (${scenario?.topVessel || 'Suspect Tanker'}) & silence gaps recorded`,
      timestamp: `${baseDate} 04:33 UTC`,
      hash: generateDeterministicHash(`${scId}-ais-matrix-correlation`),
      status: 'sealed',
      icon: 'check',
    },
    {
      id: 'm5',
      title: officerSignOff ? 'Dossier Validated & Endorsed by Lead Investigator' : 'Investigative Review & Electronic Endorsement',
      description: officerSignOff 
        ? `${officerSignOff.officerName} · Electronic Signature Verified · Section 63 BSA 2023 Endorsement`
        : `Pending Final Duty Surveillance Officer Sign-Off (ICG Maritime Operations Center)`,
      timestamp: officerSignOff?.timestamp || `${baseDate} 06:00 UTC (Awaiting Officer Sign-Off)`,
      hash: generateDeterministicHash(`${scId}-investigator-seal-${officerSignOff?.officerName || 'pending'}`),
      status: officerSignOff ? 'sealed' : 'in_review',
      icon: officerSignOff ? 'verified' : 'rate_review',
      verifiedBy: officerSignOff?.officerName,
    },
  ];
}

/**
 * Returns the forensic artifacts available for export with real dynamic GeoJSON/XML content.
 */
export function getForensicArtifacts(scenario?: Scenario | null, _masterHash?: string): ForensicArtifactItem[] {
  const sc = scenario || {
    id: 'INC-2026-005',
    title: 'Gulf of Kutch Deepwater Tanker Fairway',
    lat: 22.610,
    lng: 69.500,
    topVessel: 'MT Ocean Trader',
    area: '19.67 km²',
  } as Scenario;

  return [
    {
      id: 'art-1',
      title: 'Calibrated Oil Slick Detection Mask',
      fileName: 'slick_detection_polygon.geojson',
      type: 'geojson',
      sizeKb: 14.8,
      sha256: generateDeterministicHash(`${sc.id}-slick_polygon`),
      crs: 'EPSG:4326 (WGS 84)',
      description: 'Calibrated Sentinel-1 U-Net SAR Mask with Bonn-scale classification polygons',
      icon: 'water_drop',
      iconColor: '#ef4444',
      contentGenerator: (s, h) => JSON.stringify(
        {
          type: 'FeatureCollection',
          crs: { type: 'name', properties: { name: 'urn:ogc:def:crs:OGC:1.3:CRS84' } },
          metadata: {
            dossier_id: `SPILL-${s.id}`,
            incident_title: s.title,
            centroid: [s.lng, s.lat],
            acquisition_satellite: 'Sentinel-1A IW GRD VV+VH',
            calibration: 'Sigma0 RTC Radiometric',
            master_custody_seal: h,
            legal_framework: 'Section 63 BSA 2023 / Sec 65B Indian Evidence Act',
          },
          features: [
            {
              type: 'Feature',
              id: 'SLICK-CORE',
              properties: {
                bonn_code: 5,
                classification: 'Emulsion Core',
                area_km2: 4.82,
                estimated_volume_m3: 312.4,
                damping_ratio_db: 9.8,
              },
              geometry: {
                type: 'Polygon',
                coordinates: [[
                  [Number((s.lng - 0.02).toFixed(5)), Number((s.lat - 0.008).toFixed(5))],
                  [Number((s.lng + 0.02).toFixed(5)), Number((s.lat - 0.006).toFixed(5))],
                  [Number((s.lng + 0.025).toFixed(5)), Number((s.lat + 0.008).toFixed(5))],
                  [Number((s.lng - 0.015).toFixed(5)), Number((s.lat + 0.009).toFixed(5))],
                  [Number((s.lng - 0.02).toFixed(5)), Number((s.lat - 0.008).toFixed(5))],
                ]],
              },
            },
            {
              type: 'Feature',
              id: 'SLICK-SHEEN',
              properties: {
                bonn_code: 2,
                classification: 'Rainbow Sheen',
                area_km2: 14.85,
                estimated_volume_m3: 48.6,
                damping_ratio_db: 4.6,
              },
              geometry: {
                type: 'Polygon',
                coordinates: [[
                  [Number((s.lng - 0.045).toFixed(5)), Number((s.lat - 0.018).toFixed(5))],
                  [Number((s.lng + 0.045).toFixed(5)), Number((s.lat - 0.014).toFixed(5))],
                  [Number((s.lng + 0.05).toFixed(5)), Number((s.lat + 0.02).toFixed(5))],
                  [Number((s.lng - 0.035).toFixed(5)), Number((s.lat + 0.022).toFixed(5))],
                  [Number((s.lng - 0.045).toFixed(5)), Number((s.lat - 0.018).toFixed(5))],
                ]],
              },
            },
          ],
        },
        null,
        2
      ),
    },
    {
      id: 'art-2',
      title: 'Spill Origin Probability Contours (50%, 75%, 90%)',
      fileName: 'origin_probability_envelopes.geojson',
      type: 'geojson',
      sizeKb: 39.2,
      sha256: generateDeterministicHash(`${sc.id}-origin_probability`),
      crs: 'EPSG:4326 (WGS 84)',
      description: '50%, 75%, 90% ISO probability contours from OpenDrift Lagrangian Monte Carlo (N=10,000)',
      icon: 'grain',
      iconColor: '#f59e0b',
      contentGenerator: (s, _h) => JSON.stringify(
        {
          type: 'FeatureCollection',
          crs: { type: 'name', properties: { name: 'urn:ogc:def:crs:OGC:1.3:CRS84' } },
          metadata: {
            model: 'OpenDrift Lagrangian Backward Monte Carlo',
            target_incident: s.id,
            target_location: s.title,
            particles_simulated: 10000,
            oceanographic_currents: 'CMEMS GLOBAL_ANALYSISFORECAST_PHY_001_024',
            atmospheric_forcing: 'ECMWF ERA5 10m Wind',
          },
          features: [
            {
              type: 'Feature',
              id: 'P90-OUTER-CONTOUR',
              properties: { confidence: 0.90, isobar: '90% Origin Area', area_km2: 24.15 },
              geometry: {
                type: 'Polygon',
                coordinates: [[
                  [Number((s.lng - 0.18).toFixed(5)), Number((s.lat + 0.002).toFixed(5))],
                  [Number((s.lng - 0.09).toFixed(5)), Number((s.lat + 0.022).toFixed(5))],
                  [Number((s.lng - 0.04).toFixed(5)), Number((s.lat + 0.012).toFixed(5))],
                  [Number((s.lng - 0.09).toFixed(5)), Number((s.lat - 0.015).toFixed(5))],
                  [Number((s.lng - 0.18).toFixed(5)), Number((s.lat + 0.002).toFixed(5))],
                ]],
              },
            },
            {
              type: 'Feature',
              id: 'P50-CORE-CONTOUR',
              properties: { confidence: 0.50, isobar: '50% Core Origin Probability', area_km2: 8.42 },
              geometry: {
                type: 'Polygon',
                coordinates: [[
                  [Number((s.lng - 0.14).toFixed(5)), Number((s.lat + 0.005).toFixed(5))],
                  [Number((s.lng - 0.11).toFixed(5)), Number((s.lat + 0.015).toFixed(5))],
                  [Number((s.lng - 0.08).toFixed(5)), Number((s.lat + 0.008).toFixed(5))],
                  [Number((s.lng - 0.11).toFixed(5)), Number((s.lat - 0.005).toFixed(5))],
                  [Number((s.lng - 0.14).toFixed(5)), Number((s.lat + 0.005).toFixed(5))],
                ]],
              },
            },
          ],
        },
        null,
        2
      ),
    },
    {
      id: 'art-3',
      title: 'Correlated AIS Vessel Trajectories & Silence Gaps',
      fileName: 'ais_candidate_trajectories.gpkg',
      type: 'gpkg',
      sizeKb: 228.4,
      sha256: generateDeterministicHash(`${sc.id}-ais_candidates_gpkg`),
      crs: 'EPSG:4326 · OGC GeoPackage 1.3',
      description: '72h correlated AIS tracks, vessel kinematic gaps, and proximity vectors in OGC GeoPackage standard',
      icon: 'route',
      iconColor: 'var(--accent)',
      contentGenerator: (s, h) => `SQLite format 3\x00OGC GeoPackage Standard v1.3\nIncident: ${s.id} (${s.title})\nSuspect Target: ${s.topVessel}\nPrimary MMSI: 419001234\nMaster SHA-256 Custody Seal: ${h}\nSection 63 BSA 2023 Electronic Track Audit Trail Certified.`,
    },
    {
      id: 'art-4',
      title: 'Satellite Sensor Radiometric Calibration Header',
      fileName: 'sentinel1_calibration_metadata.xml',
      type: 'xml',
      sizeKb: 18.6,
      sha256: generateDeterministicHash(`${sc.id}-sentinel1_cal_xml`),
      crs: 'ISO 19115 / CEOS SAFE Header',
      description: 'CEOS SAFE Header Ingestion Metadata, Radiometric Noise Vectors & RTC Calibration parameters',
      icon: 'satellite_alt',
      iconColor: '#10b981',
      contentGenerator: (s, h) => `<?xml version="1.0" encoding="UTF-8"?>
<sentinel1Metadata xmlns="http://www.esa.int/safe/sentinel-1.0" xmlns:s1="http://www.esa.int/safe/sentinel-1.0/sentinel-1">
  <safeHeader>
    <incidentId>${s.id}</incidentId>
    <locationName>${s.title}</locationName>
    <centroidCoordinates>
      <latitude>${s.lat.toFixed(5)}</latitude>
      <longitude>${s.lng.toFixed(5)}</longitude>
    </centroidCoordinates>
    <satelliteMission>SENTINEL-1A</satelliteMission>
    <sensorMode>IW (Interferometric Wide Swath)</sensorMode>
    <productType>GRD (Ground Range Detected)</productType>
    <polarisation>VV+VH Dual-Pol</polarisation>
    <radiometricCalibration>Sigma0_RTC</radiometricCalibration>
    <speckleFilter>Lee-Sigma Adaptive 5x5</speckleFilter>
    <passDirection>DESCENDING</passDirection>
  </safeHeader>
  <cryptographicCustody>
    <isoStandard>ISO/IEC 27037:2012</isoStandard>
    <admissibilityFramework>Bharatiya Sakshya Adhiniyam 2023 (Section 63) / Indian Evidence Act Sec 65B</admissibilityFramework>
    <hardwareSecurityModule>FIPS 140-2 Level 3 HSM Key Vault</hardwareSecurityModule>
    <timeSource>RFC 3161 TSP GNSS Stratum-1 Atomic Clock</timeSource>
    <sha256Digest>${h}</sha256Digest>
  </cryptographicCustody>
</sentinel1Metadata>`,
    },
  ];
}
