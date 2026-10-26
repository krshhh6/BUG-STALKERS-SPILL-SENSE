export interface Scenario {
  id: string;
  title: string;
  sub: string;
  lat: number;
  lng: number;
  oilType: string;
  oilColor: string;
  oilFill: string;
  sev: string;
  sevClass: 'chip-c' | 'chip-h' | 'chip-m' | 'chip-l';
  area?: string;
  topVessel: string;
  diagVessel: string;
  diagDetails: string;
  scores: number[];
}

export interface CandidateVessel {
  name: string;
  mmsi: string;
  flag: string;
  type: string;
  score: number;
  distScore: number;
  timeScore: number;
  gapScore: number;
  typeScore: number;
  aisStatus: string;
  risk: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
}

export type TabType = 'dashboard' | 'drift' | 'attribution' | 'evidence' | 'analytics' | 'detection' | 'globe';
export type DimensionMode = '2D' | '3D';
export type BaseLayerType = 'bhuvan-satellite' | 'satellite' | 'sar' | 'carto-voyager' | 'carto-dark' | 'sar-vh' | 'opensea' | 'msn' | 'day' | 'dark' | 'bhuvan-vector' | 'google-satellite' | 'arcgis' | 'bhuvan';

export type CopernicusLayerId = 'true-color' | 'sar-vv' | 'sar-vh' | 'swir-oil' | 'false-color' | 'ndwi' | 'thermal' | 'nisar-ls' | 'eos-04';

export interface BhoonidhiNisarGranule {
  id: string;
  dataset: 'NISAR_S' | 'NISAR_L' | 'NISAR_JOINT' | 'EOS04';
  product: 'L2-GCOV' | 'L1-RSLC';
  acquisitionTime: string;
  orbitDirection: 'ASCENDING' | 'DESCENDING';
  sBandDampingDb: number;
  lBandDampingDb: number;
  dfdi: number;
  verdict: 'CONFIRMED_MINERAL_OIL' | 'BIOGENIC_LOOKALIKE';
  confidence: number;
}

export interface CopernicusLayerInfo {
  id: CopernicusLayerId;
  name: string;
  bands: string;
  desc: string;
  badge: string;
  badgeColor: string;
  thumbBg: string;
}

export interface AttributionWeights {
  dist: number;
  time: number;
  gap: number;
  type: number;
  temporal?: number;
  distance?: number;
  aisGap?: number;
  vesselType?: number;
}

export interface DetectedPolygon {
  geometry: {
    type: string;
    coordinates: [number, number][][];
  };
  confidence: number;
  area_km2: number;
  mean_damping_db: number;
  slick_centroid: [number, number];
}

export interface DetectionResult {
  status: 'idle' | 'loading' | 'detected' | 'no_oil_detected' | 'no_recent_imagery' | 'error';
  scene_timestamp?: string;
  sensor?: string;
  scene_id?: string;
  polarization?: string;
  orbit_direction?: string;
  polygons: DetectedPolygon[];
  total_detected_area_km2?: number;
  model_version?: string;
  message?: string;
  error?: string;
}

export interface CropBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface CropInfo extends CropBox {
  originalWidth: number;
  originalHeight: number;
  isCropped: boolean;
  aspectRatio: number;
  wasCenterCropped?: boolean;
}

export interface SarClassificationResult {
  imageFile: string;
  prediction: 'oil_spill' | 'no_oil' | 'invalid_sar';
  confidence: number;
  inferenceTimeMs: number;
  gradcamDataUrl?: string;
  errorMessage?: string;
  rejectionReason?: string;
  segmentationMask?: string;
  spillAreaPercent?: number;
  segmentationTimeMs?: number;
  cropInfo?: CropInfo;
  metrics?: {
    meanBrightness: number;
    brightRatio: number;
    sharpTransitions: number;
    isColor: boolean;
  };
  majorSpillBoundingBox?: CropBox;
  focusedSlickDataUrl?: string;
}

export interface SarDriftPayload {
  imageSrc: string;
  maskSrc?: string;
  fileName: string;
  prediction: 'oil_spill' | 'no_oil' | 'invalid_sar';
  confidence: number;
  spillAreaPercent: number;
  estimatedAreaKm2: number;
  inferenceTimeMs: number;
  segmentationTimeMs?: number;
  timestamp: string;
  lat: number;
  lng: number;
  locationName: string;
  cropInfo?: CropInfo;
  metrics?: {
    meanBrightness?: number;
    brightRatio?: number;
    sharpTransitions?: number;
    isColor?: boolean;
  };
}


