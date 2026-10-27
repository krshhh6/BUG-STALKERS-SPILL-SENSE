import React, { useState, useEffect } from 'react';
import type { TabType, Scenario, SarDriftPayload } from '../../types/dashboard';
import { SCENARIOS } from '../../data/scenarios';
import {
  fetchLiveMetOcean,
  computeLiveDriftSimulation,
  type LiveSimulationResult,
} from '../../services/marineWeatherService';

import { DriftTrajectoryMap } from '../DriftTrajectoryMap';

interface DriftViewProps {
  onSelectTab: (tab: TabType) => void;
  currentScenario?: Scenario | null;
  onSelectScenario?: (key: string) => void;
  sarDriftPayload?: SarDriftPayload | null;
  onDriftSimulationUpdated?: (result: LiveSimulationResult) => void;
}

interface TimestepForecast {
  timeOffsetHours: number;
  label: string;
  timestamp: string;
  lat: number;
  lng: number;
  areaKm2: number;
  slickRadiusKm: number;
  driftSpeedKnots: number;
  headingDeg: number;
  distanceToCoastKm: number;
  evaporatedPct: number;
  emulsifiedPct: number;
  dispersedPct: number;
  remainingSurfacePct: number;
  viscosityCSt: number;
  waterContentPct: number;
  threatLevel: 'CRITICAL' | 'HIGH' | 'MODERATE' | 'LOW';
  containmentRecommendation: string;
}

interface ScenarioDriftProfile {
  scenarioId: string;
  incidentName: string;
  seaBasin: string;
  oilType: string;
  apiGravity: string;
  initialObservedCoords: string;
  initialAreaKm2: number;
  currentVector: string;
  windVector: string;
  seaState: string;
  sstCelsius: number;
  // Origin Backtracking
  originCoords: string;
  originWindow: string;
  originAreaKm2: number;
  suspectVessel: string;
  backtrackConfidencePct: number;
  // Future Forecast Timesteps
  forecasts: TimestepForecast[];
  coastalZoneName: string;
  overallLandfallEta: string;
  vulnerableHabitats: string[];
}

const DRIFT_PROFILES: Record<string, ScenarioDriftProfile> = {
  'INC-001': {
    scenarioId: 'INC-2026-001',
    incidentName: 'Mumbai High Basin Crude Spill',
    seaBasin: 'Arabian Sea Offshore EEZ',
    oilType: 'Bombay High Heavy Crude',
    apiGravity: '33.4° API',
    initialObservedCoords: '18.743°N, 71.218°E',
    initialAreaKm2: 4.82,
    currentVector: '0.42 m/s @ 065° (ENE Surface Jet)',
    windVector: '6.8 m/s @ 245° (WSW Monsoon Wind)',
    seaState: 'Sea State 3 (Wave H_s: 1.4 m)',
    sstCelsius: 28.4,
    originCoords: '18.62°N, 71.18°E',
    originWindow: 'Discharge Window: T - 20h to - 24h',
    originAreaKm2: 68.4,
    suspectVessel: 'CRUDE ATLAS (MMSI 419001234)',
    backtrackConfidencePct: 88,
    coastalZoneName: 'Alibaug Coastline & Murud-Janjira Shelf',
    overallLandfallEta: '46h 30m (Critical Watch)',
    vulnerableHabitats: [
      'Alibaug Intertidal Mudflats & Mangrove Creeks (38 km ENE)',
      'Kihim Beach Artisanal Fisheries & Marine Protected Zone',
      'Mumbai Harbour Shipping Channel & Desalination Intakes',
    ],
    forecasts: [
      {
        timeOffsetHours: 6,
        label: 'T + 6 Hours',
        timestamp: 'T + 06:00 UTC',
        lat: 18.778,
        lng: 71.275,
        areaKm2: 9.4,
        slickRadiusKm: 1.73,
        driftSpeedKnots: 0.65,
        headingDeg: 68,
        distanceToCoastKm: 78.5,
        evaporatedPct: 15,
        emulsifiedPct: 8,
        dispersedPct: 5,
        remainingSurfacePct: 72,
        viscosityCSt: 42,
        waterContentPct: 18,
        threatLevel: 'LOW',
        containmentRecommendation: 'Deploy Tier-1 Fast-Response Ocean Boom (1,200m) at upstream leading edge.',
      },
      {
        timeOffsetHours: 12,
        label: 'T + 12 Hours',
        timestamp: 'T + 12:00 UTC',
        lat: 18.815,
        lng: 71.342,
        areaKm2: 17.8,
        slickRadiusKm: 2.38,
        driftSpeedKnots: 0.68,
        headingDeg: 66,
        distanceToCoastKm: 64.2,
        evaporatedPct: 24,
        emulsifiedPct: 19,
        dispersedPct: 9,
        remainingSurfacePct: 48,
        viscosityCSt: 165,
        waterContentPct: 35,
        threatLevel: 'MODERATE',
        containmentRecommendation: 'Chemical dispersant application window optimal before emulsification exceeds 30%.',
      },
      {
        timeOffsetHours: 24,
        label: 'T + 24 Hours',
        timestamp: 'T + 24:00 UTC',
        lat: 18.892,
        lng: 71.488,
        areaKm2: 34.6,
        slickRadiusKm: 3.32,
        driftSpeedKnots: 0.72,
        headingDeg: 64,
        distanceToCoastKm: 47.1,
        evaporatedPct: 32,
        emulsifiedPct: 31,
        dispersedPct: 14,
        remainingSurfacePct: 23,
        viscosityCSt: 580,
        waterContentPct: 52,
        threatLevel: 'HIGH',
        containmentRecommendation: 'Position offshore skimmer vessels (ICGS Samudra Prahari) along 71.48°E interception meridian.',
      },
      {
        timeOffsetHours: 36,
        label: 'T + 36 Hours',
        timestamp: 'T + 36:00 UTC',
        lat: 18.968,
        lng: 71.632,
        areaKm2: 52.1,
        slickRadiusKm: 4.07,
        driftSpeedKnots: 0.70,
        headingDeg: 65,
        distanceToCoastKm: 31.8,
        evaporatedPct: 36,
        emulsifiedPct: 38,
        dispersedPct: 17,
        remainingSurfacePct: 9,
        viscosityCSt: 1150,
        waterContentPct: 65,
        threatLevel: 'CRITICAL',
        containmentRecommendation: 'Establish nearshore containment barriers across Alibaug creek entrances.',
      },
      {
        timeOffsetHours: 48,
        label: 'T + 48 Hours',
        timestamp: 'T + 48:00 UTC',
        lat: 19.042,
        lng: 71.775,
        areaKm2: 68.5,
        slickRadiusKm: 4.67,
        driftSpeedKnots: 0.67,
        headingDeg: 67,
        distanceToCoastKm: 18.4,
        evaporatedPct: 38,
        emulsifiedPct: 41,
        dispersedPct: 18,
        remainingSurfacePct: 3,
        viscosityCSt: 1850,
        waterContentPct: 72,
        threatLevel: 'CRITICAL',
        containmentRecommendation: 'Shoreline pre-impact shoreline protection crews deployed with sorbent booms.',
      },
    ],
  },
  'INC-002': {
    scenarioId: 'INC-2026-002',
    incidentName: 'Chennai–Ennore Corridor Fuel Spill',
    seaBasin: 'Bay of Bengal EEZ',
    oilType: 'Heavy Bunker Fuel (IFO-380)',
    apiGravity: '12.8° API (High Viscosity)',
    initialObservedCoords: '13.250°N, 80.460°E',
    initialAreaKm2: 2.40,
    currentVector: '0.58 m/s @ 340° (NNW Coastal Current)',
    windVector: '8.2 m/s @ 120° (SE Sea Breeze)',
    seaState: 'Sea State 4 (Wave H_s: 1.8 m)',
    sstCelsius: 29.1,
    originCoords: '13.150°N, 80.420°E',
    originWindow: 'Discharge Window: T - 18h to - 22h',
    originAreaKm2: 42.1,
    suspectVessel: 'PACIFIC GLORY (MMSI 419009988)',
    backtrackConfidencePct: 78,
    coastalZoneName: 'Pulicat Lake Estuary & Ennore Shoals',
    overallLandfallEta: '18h 45m (IMMINENT THREAT)',
    vulnerableHabitats: [
      'Pulicat Lagoon Saltwater Barrier Sandbars (9.5 km NNW)',
      'Ennore Thermal Power Plant Cooling Water Intakes (12 km WNW)',
      'Kattupalli Port Marina & Coromandel Mangroves',
    ],
    forecasts: [
      {
        timeOffsetHours: 6,
        label: 'T + 6 Hours',
        timestamp: 'T + 06:00 UTC',
        lat: 13.295,
        lng: 80.430,
        areaKm2: 5.2,
        slickRadiusKm: 1.29,
        driftSpeedKnots: 0.85,
        headingDeg: 338,
        distanceToCoastKm: 14.2,
        evaporatedPct: 8,
        emulsifiedPct: 15,
        dispersedPct: 3,
        remainingSurfacePct: 74,
        viscosityCSt: 1450,
        waterContentPct: 28,
        threatLevel: 'MODERATE',
        containmentRecommendation: 'Dispatch high-buoyancy ocean booms to enclose Ennore Outer Channel.',
      },
      {
        timeOffsetHours: 12,
        label: 'T + 12 Hours',
        timestamp: 'T + 12:00 UTC',
        lat: 13.345,
        lng: 80.398,
        areaKm2: 10.8,
        slickRadiusKm: 1.85,
        driftSpeedKnots: 0.88,
        headingDeg: 335,
        distanceToCoastKm: 8.6,
        evaporatedPct: 12,
        emulsifiedPct: 32,
        dispersedPct: 6,
        remainingSurfacePct: 50,
        viscosityCSt: 3800,
        waterContentPct: 48,
        threatLevel: 'HIGH',
        containmentRecommendation: 'Heavy fuel oil emulsifying rapidly. Dispersants ineffective; switch to mechanical skimming.',
      },
      {
        timeOffsetHours: 24,
        label: 'T + 24 Hours',
        timestamp: 'T + 24:00 UTC',
        lat: 13.438,
        lng: 80.335,
        areaKm2: 21.4,
        slickRadiusKm: 2.61,
        driftSpeedKnots: 0.84,
        headingDeg: 332,
        distanceToCoastKm: 2.4,
        evaporatedPct: 15,
        emulsifiedPct: 48,
        dispersedPct: 8,
        remainingSurfacePct: 29,
        viscosityCSt: 8200,
        waterContentPct: 68,
        threatLevel: 'CRITICAL',
        containmentRecommendation: 'Urgent shore defense: seal Pulicat tidal inlet with bubble barrier and sorbent curtains.',
      },
      {
        timeOffsetHours: 36,
        label: 'T + 36 Hours',
        timestamp: 'T + 36:00 UTC',
        lat: 13.520,
        lng: 80.285,
        areaKm2: 32.5,
        slickRadiusKm: 3.22,
        driftSpeedKnots: 0.80,
        headingDeg: 330,
        distanceToCoastKm: 0.0,
        evaporatedPct: 17,
        emulsifiedPct: 54,
        dispersedPct: 9,
        remainingSurfacePct: 20,
        viscosityCSt: 14500,
        waterContentPct: 75,
        threatLevel: 'CRITICAL',
        containmentRecommendation: 'Beach cleanup protocol activated; coastal tar-ball collection teams mobilized.',
      },
      {
        timeOffsetHours: 48,
        label: 'T + 48 Hours',
        timestamp: 'T + 48:00 UTC',
        lat: 13.590,
        lng: 80.245,
        areaKm2: 41.0,
        slickRadiusKm: 3.61,
        driftSpeedKnots: 0.75,
        headingDeg: 328,
        distanceToCoastKm: 0.0,
        evaporatedPct: 18,
        emulsifiedPct: 58,
        dispersedPct: 10,
        remainingSurfacePct: 14,
        viscosityCSt: 18200,
        waterContentPct: 80,
        threatLevel: 'CRITICAL',
        containmentRecommendation: 'Subsurface net retrieval for dense weathered bitumen residue.',
      },
    ],
  },
  'INC-003': {
    scenarioId: 'INC-2026-003',
    incidentName: 'Andaman Sea Bilge Discharge',
    seaBasin: 'Andaman Sea Malacca Approach',
    oilType: 'Oily Bilge Water / Light Fuel Mixture',
    apiGravity: '38.0° API',
    initialObservedCoords: '10.456°N, 93.123°E',
    initialAreaKm2: 0.95,
    currentVector: '0.32 m/s @ 315° (NW Monsoon Drift)',
    windVector: '5.4 m/s @ 210° (SSW Breeze)',
    seaState: 'Sea State 2 (Wave H_s: 0.9 m)',
    sstCelsius: 29.8,
    originCoords: '10.380°N, 93.080°E',
    originWindow: 'Discharge Window: T - 12h to - 16h',
    originAreaKm2: 24.5,
    suspectVessel: 'UNKNOWN (DARK VESSEL / SAR DETECTED)',
    backtrackConfidencePct: 74,
    coastalZoneName: 'Little Andaman Sea Corridor',
    overallLandfallEta: '> 72h (Deep Ocean Drift)',
    vulnerableHabitats: [
      'Little Andaman Coral Reef Enclave (62 km West)',
      'International Commercial Shipping Channel 7',
    ],
    forecasts: [
      {
        timeOffsetHours: 6,
        label: 'T + 6 Hours',
        timestamp: 'T + 06:00 UTC',
        lat: 10.485,
        lng: 93.148,
        areaKm2: 2.1,
        slickRadiusKm: 0.82,
        driftSpeedKnots: 0.52,
        headingDeg: 322,
        distanceToCoastKm: 68.2,
        evaporatedPct: 32,
        emulsifiedPct: 5,
        dispersedPct: 14,
        remainingSurfacePct: 49,
        viscosityCSt: 12,
        waterContentPct: 10,
        threatLevel: 'LOW',
        containmentRecommendation: 'High natural evaporation rate; monitor via satellite SAR passes.',
      },
      {
        timeOffsetHours: 12,
        label: 'T + 12 Hours',
        timestamp: 'T + 12:00 UTC',
        lat: 10.520,
        lng: 93.175,
        areaKm2: 4.2,
        slickRadiusKm: 1.16,
        driftSpeedKnots: 0.55,
        headingDeg: 320,
        distanceToCoastKm: 64.0,
        evaporatedPct: 48,
        emulsifiedPct: 8,
        dispersedPct: 22,
        remainingSurfacePct: 22,
        viscosityCSt: 25,
        waterContentPct: 15,
        threatLevel: 'LOW',
        containmentRecommendation: 'Thin rainbow sheen rapidly dissipating under tropical insolation.',
      },
      {
        timeOffsetHours: 24,
        label: 'T + 24 Hours',
        timestamp: 'T + 24:00 UTC',
        lat: 10.590,
        lng: 93.220,
        areaKm2: 7.8,
        slickRadiusKm: 1.58,
        driftSpeedKnots: 0.58,
        headingDeg: 318,
        distanceToCoastKm: 58.5,
        evaporatedPct: 58,
        emulsifiedPct: 10,
        dispersedPct: 26,
        remainingSurfacePct: 6,
        viscosityCSt: 45,
        waterContentPct: 20,
        threatLevel: 'LOW',
        containmentRecommendation: 'Residual sheen below MARPOL regulatory threshold within 30 hours.',
      },
      {
        timeOffsetHours: 36,
        label: 'T + 36 Hours',
        timestamp: 'T + 36:00 UTC',
        lat: 10.655,
        lng: 93.265,
        areaKm2: 9.5,
        slickRadiusKm: 1.74,
        driftSpeedKnots: 0.54,
        headingDeg: 315,
        distanceToCoastKm: 53.0,
        evaporatedPct: 64,
        emulsifiedPct: 11,
        dispersedPct: 23,
        remainingSurfacePct: 2,
        viscosityCSt: 60,
        waterContentPct: 22,
        threatLevel: 'LOW',
        containmentRecommendation: 'Dispersed background baseline level.',
      },
      {
        timeOffsetHours: 48,
        label: 'T + 48 Hours',
        timestamp: 'T + 48:00 UTC',
        lat: 10.720,
        lng: 93.310,
        areaKm2: 10.2,
        slickRadiusKm: 1.80,
        driftSpeedKnots: 0.51,
        headingDeg: 312,
        distanceToCoastKm: 48.0,
        evaporatedPct: 67,
        emulsifiedPct: 12,
        dispersedPct: 20,
        remainingSurfacePct: 1,
        viscosityCSt: 75,
        waterContentPct: 24,
        threatLevel: 'LOW',
        containmentRecommendation: 'No active shoreline risk detected.',
      },
    ],
  },
  'INC-004': {
    scenarioId: 'INC-2026-004',
    incidentName: 'Goa Coastal Bunkering Sheen',
    seaBasin: 'Goa Coastal Waters',
    oilType: 'Marine Gas Oil (MGO)',
    apiGravity: '36.5° API',
    initialObservedCoords: '15.420°N, 73.650°E',
    initialAreaKm2: 1.75,
    currentVector: '0.28 m/s @ 165° (SSE Nearshore Tidal)',
    windVector: '4.8 m/s @ 280° (Westerly Onshore Breeze)',
    seaState: 'Sea State 2 (Wave H_s: 0.8 m)',
    sstCelsius: 28.7,
    originCoords: '15.460°N, 73.610°E',
    originWindow: 'Discharge Window: T - 08h to - 12h',
    originAreaKm2: 18.2,
    suspectVessel: 'SEA PEARL (MMSI 419003322)',
    backtrackConfidencePct: 82,
    coastalZoneName: 'Mormugao Port Outer Anchorage & Baina Beach',
    overallLandfallEta: '32h 10m',
    vulnerableHabitats: [
      'Baina Beach Coastal Tourism & Fishing Staging Area (11 km ESE)',
      'Zuari River Estuarine Mangrove Complex',
    ],
    forecasts: [
      {
        timeOffsetHours: 6,
        label: 'T + 6 Hours',
        timestamp: 'T + 06:00 UTC',
        lat: 15.395,
        lng: 73.675,
        areaKm2: 3.4,
        slickRadiusKm: 1.04,
        driftSpeedKnots: 0.48,
        headingDeg: 155,
        distanceToCoastKm: 9.8,
        evaporatedPct: 22,
        emulsifiedPct: 4,
        dispersedPct: 8,
        remainingSurfacePct: 66,
        viscosityCSt: 18,
        waterContentPct: 12,
        threatLevel: 'LOW',
        containmentRecommendation: 'Port authority tug with containment sweep deployed in anchorage fairway.',
      },
      {
        timeOffsetHours: 12,
        label: 'T + 12 Hours',
        timestamp: 'T + 12:00 UTC',
        lat: 15.365,
        lng: 73.705,
        areaKm2: 6.2,
        slickRadiusKm: 1.40,
        driftSpeedKnots: 0.50,
        headingDeg: 152,
        distanceToCoastKm: 7.2,
        evaporatedPct: 34,
        emulsifiedPct: 8,
        dispersedPct: 14,
        remainingSurfacePct: 44,
        viscosityCSt: 32,
        waterContentPct: 22,
        threatLevel: 'MODERATE',
        containmentRecommendation: 'Deploy sorbent boom line across Zuari river mouth approach.',
      },
      {
        timeOffsetHours: 24,
        label: 'T + 24 Hours',
        timestamp: 'T + 24:00 UTC',
        lat: 15.310,
        lng: 73.748,
        areaKm2: 11.8,
        slickRadiusKm: 1.94,
        driftSpeedKnots: 0.52,
        headingDeg: 150,
        distanceToCoastKm: 3.9,
        evaporatedPct: 46,
        emulsifiedPct: 14,
        dispersedPct: 18,
        remainingSurfacePct: 22,
        viscosityCSt: 58,
        waterContentPct: 35,
        threatLevel: 'HIGH',
        containmentRecommendation: 'Mobilize Coast Guard District HQ 11 response craft with disc skimmers.',
      },
      {
        timeOffsetHours: 36,
        label: 'T + 36 Hours',
        timestamp: 'T + 36:00 UTC',
        lat: 15.250,
        lng: 73.785,
        areaKm2: 17.5,
        slickRadiusKm: 2.36,
        driftSpeedKnots: 0.49,
        headingDeg: 148,
        distanceToCoastKm: 1.1,
        evaporatedPct: 52,
        emulsifiedPct: 18,
        dispersedPct: 20,
        remainingSurfacePct: 10,
        viscosityCSt: 95,
        waterContentPct: 45,
        threatLevel: 'CRITICAL',
        containmentRecommendation: 'Shoreline boom defense active along Baina Beach shorelines.',
      },
      {
        timeOffsetHours: 48,
        label: 'T + 48 Hours',
        timestamp: 'T + 48:00 UTC',
        lat: 15.195,
        lng: 73.820,
        areaKm2: 21.0,
        slickRadiusKm: 2.59,
        driftSpeedKnots: 0.45,
        headingDeg: 145,
        distanceToCoastKm: 0.0,
        evaporatedPct: 55,
        emulsifiedPct: 20,
        dispersedPct: 21,
        remainingSurfacePct: 4,
        viscosityCSt: 140,
        waterContentPct: 50,
        threatLevel: 'CRITICAL',
        containmentRecommendation: 'Post-landfall recovery and environmental remediation monitoring.',
      },
    ],
  },
};

function createSarDriftProfile(payload: SarDriftPayload): ScenarioDriftProfile {
  const area = payload.estimatedAreaKm2;
  const lat = payload.lat;
  const lng = payload.lng;
  const confPct = Math.round(payload.confidence * 100);

  // Derive dynamic met-ocean trajectory parameters from incident geolocation
  const isWestCoast = lng < 75.0;
  const isEastCoast = lng >= 79.0 && lng < 85.0;
  const netHeadingDeg = isEastCoast ? 30 : isWestCoast ? 65 : 80;
  const driftSpeedKnots = 0.65;
  const headingRad = (netHeadingDeg * Math.PI) / 180.0;

  const kmPerDegLat = 111.32;
  const kmPerDegLng = 111.32 * Math.cos((lat * Math.PI) / 180.0);
  const netSpeedKmh = driftSpeedKnots * 1.852;
  const uKmh = netSpeedKmh * Math.sin(headingRad);
  const vKmh = netSpeedKmh * Math.cos(headingRad);

  // Dynamic 24h backtrack origin
  const originLat = +(lat - (vKmh * 22.0) / kmPerDegLat).toFixed(3);
  const originLng = +(lng - (uKmh * 22.0) / kmPerDegLng).toFixed(3);

  const steps = [
    { hours: 6, label: 'T + 6 Hours', spread: 1.55, distKm: 76.5, evap: 15, emul: 6, disp: 5, threat: 'LOW' as const, rec: 'Deploy containment boom around primary detected slick core.' },
    { hours: 12, label: 'T + 12 Hours', spread: 2.30, distKm: 62.0, evap: 25, emul: 14, disp: 8, threat: 'MODERATE' as const, rec: 'Offshore skimming craft deployed along leading edge vector.' },
    { hours: 24, label: 'T + 24 Hours', spread: 3.65, distKm: 44.5, evap: 36, emul: 28, disp: 12, threat: 'HIGH' as const, rec: 'Seal vulnerable coastal tidal inlets with bubble barriers and sorbent curtains.' },
    { hours: 36, label: 'T + 36 Hours', spread: 5.10, distKm: 27.0, evap: 44, emul: 40, disp: 16, threat: 'HIGH' as const, rec: 'Pre-position shoreline protection units and beach cleanup recovery gear.' },
    { hours: 48, label: 'T + 48 Hours', spread: 6.80, distKm: 13.5, evap: 50, emul: 52, disp: 18, threat: 'CRITICAL' as const, rec: 'Active shoreline defense protocol; nearshore skimming & absorbent deployment.' },
  ];

  return {
    scenarioId: `SAR-${payload.fileName.replace(/[^a-zA-Z0-9]/g, '_').substring(0, 14).toUpperCase()}`,
    incidentName: `Live SAR AI Detection: ${payload.fileName}`,
    seaBasin: `${payload.locationName} Maritime Basin`,
    oilType: 'Sentinel-1 Ingested Hydrocarbon (SpillSegNet Masked)',
    apiGravity: '32.5° API (Medium-Heavy Marine)',
    initialObservedCoords: `${lat.toFixed(3)}°N, ${lng.toFixed(3)}°E`,
    initialAreaKm2: area,
    currentVector: `${(driftSpeedKnots * 0.514).toFixed(2)} m/s @ ${String(netHeadingDeg).padStart(3, '0')}°`,
    windVector: '6.5 m/s @ 245° (Stokes Windage 3.5%)',
    seaState: 'Dynamic Satellite Met-Ocean (CMEMS + ERA5)',
    sstCelsius: 28.6,
    originCoords: `${Math.abs(originLat).toFixed(3)}°N, ${Math.abs(originLng).toFixed(3)}°E`,
    originWindow: `SAR Pass: ${new Date(payload.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} UTC`,
    originAreaKm2: +(area * 6.5).toFixed(1),
    suspectVessel: 'AIS TRACK CORRELATION ACTIVE',
    backtrackConfidencePct: Math.min(99, Math.round(confPct * 0.95)),
    coastalZoneName: `${payload.locationName} Intertidal Shelf`,
    overallLandfallEta: `${Math.max(16, Math.round(54 - area * 1.5))}h (Active Watch)`,
    vulnerableHabitats: [
      `${payload.locationName} Sensitive Coastal Mangroves & Inlets`,
      'Marine Protected Biome & Artisanal Fishery Shelf',
      'High-Density International Commercial Shipping Channel',
    ],
    forecasts: steps.map((s) => {
      const stepArea = +(area * s.spread).toFixed(2);
      const slickRadiusKm = +(Math.sqrt(stepArea / Math.PI)).toFixed(2);
      const remainingSurfacePct = Math.max(1, 100 - (s.evap + s.emul + s.disp));
      const stepLat = +(lat + (vKmh * s.hours) / kmPerDegLat).toFixed(3);
      const stepLng = +(lng + (uKmh * s.hours) / kmPerDegLng).toFixed(3);
      return {
        timeOffsetHours: s.hours,
        label: s.label,
        timestamp: `T + ${String(s.hours).padStart(2, '0')}:00 UTC`,
        lat: stepLat,
        lng: stepLng,
        areaKm2: stepArea,
        slickRadiusKm,
        driftSpeedKnots,
        headingDeg: netHeadingDeg,
        distanceToCoastKm: s.distKm,
        evaporatedPct: s.evap,
        emulsifiedPct: s.emul,
        dispersedPct: s.disp,
        remainingSurfacePct,
        viscosityCSt: 65 + s.hours * 45,
        waterContentPct: +(s.emul * 1.25).toFixed(1),
        threatLevel: s.threat,
        containmentRecommendation: s.rec,
      };
    }),
  };
}

export const DriftView: React.FC<DriftViewProps> = ({
  onSelectTab,
  currentScenario,
  onSelectScenario,
  sarDriftPayload,
  onDriftSimulationUpdated,
}) => {
  const [activeSarPayload, setActiveSarPayload] = useState<SarDriftPayload | null>(sarDriftPayload || null);

  // Determine active scenario key
  const defaultKey = sarDriftPayload
    ? 'SAR-DETECTION'
    : currentScenario?.id.includes('002')
    ? 'INC-002'
    : currentScenario?.id.includes('003')
    ? 'INC-003'
    : currentScenario?.id.includes('004')
    ? 'INC-004'
    : 'INC-001';

  const [selectedKey, setSelectedKey] = useState<string>(defaultKey);
  const [activeMode, setActiveMode] = useState<'forward' | 'backward' | 'unified'>('forward');
  const [selectedStepIndex, setSelectedStepIndex] = useState<number>(2); // T+24h by default
  const [isSimulating, setIsSimulating] = useState<boolean>(false);
  const [simProgress, setSimProgress] = useState<number>(100);
  const [exportNotice, setExportNotice] = useState<string | null>(null);

  const [liveResult, setLiveResult] = useState<LiveSimulationResult | null>(null);
  const [isLoadingLive, setIsLoadingLive] = useState<boolean>(false);

  // When sarDriftPayload prop updates, switch immediately to SAR-DETECTION and trigger live calculation
  useEffect(() => {
    if (sarDriftPayload) {
      setActiveSarPayload(sarDriftPayload);
      setSelectedKey('SAR-DETECTION');
      setSelectedStepIndex(2);
      refreshLiveDriftData('SAR-DETECTION', sarDriftPayload);
    }
  }, [sarDriftPayload]);

  interface ApiBacktrackResult {
    originLat: number;
    originLng: number;
    backtrackPath: [number, number][];
    envelope50: [number, number][];
    envelope75: [number, number][];
    envelope90: [number, number][];
    historicalCandidates: any[];
  }

  const fetchApiBacktrackSimulation = async (
    incidentId: string,
    lat: number,
    lng: number,
    horizonHours: number = 24,
    particleCount: number = 1000
  ): Promise<ApiBacktrackResult | null> => {
    try {
      const res = await fetch('/api/v1/drift/backtrack', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          incident_id: incidentId,
          lat,
          lng,
          horizon_hours: horizonHours,
          particle_count: particleCount,
        }),
      });
      if (!res.ok) return null;
      const data = await res.json();
      const features: any[] = data.envelopes_geojson?.features || [];

      const getPolyCoords = (conf: string): [number, number][] => {
        const feat = features.find((f: any) => f.geometry?.type === 'Polygon' && f.properties?.confidence?.includes(conf));
        if (!feat || !feat.geometry?.coordinates?.[0]) return [];
        return feat.geometry.coordinates[0].map(([lon, lat]: [number, number]) => [lat, lon] as [number, number]);
      };

      const lineFeat = features.find((f: any) => f.geometry?.type === 'LineString');
      const backtrackPath: [number, number][] = lineFeat?.geometry?.coordinates
        ? lineFeat.geometry.coordinates.map(([lon, lat]: [number, number]) => [lat, lon] as [number, number])
        : [];

      return {
        originLat: data.origin_coordinates.lat,
        originLng: data.origin_coordinates.lon,
        backtrackPath,
        envelope50: getPolyCoords('50'),
        envelope75: getPolyCoords('75'),
        envelope90: getPolyCoords('95').length > 0 ? getPolyCoords('95') : getPolyCoords('90'),
        historicalCandidates: data.historical_candidates || [],
      };
    } catch (err) {
      console.debug('[DriftView] Backend /api/v1/drift/backtrack unavailable, using client-side physics fallback:', err);
      return null;
    }
  };

  const refreshLiveDriftData = async (keyOverride?: string, customPayload?: SarDriftPayload | null) => {
    const k = keyOverride || selectedKey;
    const activePayload = customPayload !== undefined ? customPayload : activeSarPayload;

    if (k === 'SAR-DETECTION' && activePayload) {
      setIsLoadingLive(true);
      try {
        const [metOcean, apiBacktrack] = await Promise.all([
          fetchLiveMetOcean(activePayload.lat, activePayload.lng),
          fetchApiBacktrackSimulation('SAR-DETECTION', activePayload.lat, activePayload.lng, 24, 1000)
        ]);
        const computed = computeLiveDriftSimulation(
          activePayload.lat,
          activePayload.lng,
          activePayload.estimatedAreaKm2,
          'Sentinel-1 Detected Hydrocarbon Fraction',
          metOcean,
          `${activePayload.locationName} Shelf`,
          [
            `${activePayload.locationName} Sensitive Mangroves`,
            'Marine Sanctuary & Protected Fisheries Ground',
            'Commercial Navigational Shipping Corridor',
          ]
        );
        if (apiBacktrack) {
          if (apiBacktrack.backtrackPath.length > 0) computed.backtrackPath = apiBacktrack.backtrackPath;
          if (apiBacktrack.envelope50.length > 0) computed.envelope50 = apiBacktrack.envelope50;
          if (apiBacktrack.envelope75.length > 0) computed.envelope75 = apiBacktrack.envelope75;
          if (apiBacktrack.envelope90.length > 0) computed.envelope90 = apiBacktrack.envelope90;
          computed.originCoordNumeric = [apiBacktrack.originLat, apiBacktrack.originLng];
          computed.originCoords = `${apiBacktrack.originLat.toFixed(3)}°N, ${apiBacktrack.originLng.toFixed(3)}°E`;
        }
        setLiveResult(computed);
        onDriftSimulationUpdated?.(computed);
      } catch (e) {
        console.warn('[DriftView] Live met-ocean fetch error for SAR Detection:', e);
      } finally {
        setIsLoadingLive(false);
      }
      return;
    }

    const activeSc = SCENARIOS[k] || currentScenario || SCENARIOS['INC-001'];
    const p = DRIFT_PROFILES[k] || DRIFT_PROFILES['INC-001'];
    setIsLoadingLive(true);
    try {
      const [metOcean, apiBacktrack] = await Promise.all([
        fetchLiveMetOcean(activeSc.lat, activeSc.lng),
        fetchApiBacktrackSimulation(activeSc.id || k, activeSc.lat, activeSc.lng, 24, 1000)
      ]);
      const computed = computeLiveDriftSimulation(
        activeSc.lat,
        activeSc.lng,
        p.initialAreaKm2,
        activeSc.oilType,
        metOcean,
        p.coastalZoneName,
        p.vulnerableHabitats
      );
      if (apiBacktrack) {
        if (apiBacktrack.backtrackPath.length > 0) computed.backtrackPath = apiBacktrack.backtrackPath;
        if (apiBacktrack.envelope50.length > 0) computed.envelope50 = apiBacktrack.envelope50;
        if (apiBacktrack.envelope75.length > 0) computed.envelope75 = apiBacktrack.envelope75;
        if (apiBacktrack.envelope90.length > 0) computed.envelope90 = apiBacktrack.envelope90;
        computed.originCoordNumeric = [apiBacktrack.originLat, apiBacktrack.originLng];
        computed.originCoords = `${apiBacktrack.originLat.toFixed(3)}°N, ${apiBacktrack.originLng.toFixed(3)}°E`;
      }
      setLiveResult(computed);
      onDriftSimulationUpdated?.(computed);
    } catch (e) {
      console.warn('[DriftView] Live met-ocean fetch error:', e);
    } finally {
      setIsLoadingLive(false);
    }
  };

  // Sync when currentScenario changes
  useEffect(() => {
    if (currentScenario?.id && selectedKey !== 'SAR-DETECTION') {
      const matchedKey = Object.keys(SCENARIOS).find(
        (k) => SCENARIOS[k].id === currentScenario.id || currentScenario.id.includes(k.replace('INC-', ''))
      );
      if (matchedKey) {
        setSelectedKey(matchedKey);
        refreshLiveDriftData(matchedKey);
      }
    }
  }, [currentScenario]);

  useEffect(() => {
    refreshLiveDriftData(selectedKey);
  }, [selectedKey]);

  const profile = (selectedKey === 'SAR-DETECTION' && activeSarPayload)
    ? createSarDriftProfile(activeSarPayload)
    : (DRIFT_PROFILES[selectedKey] || DRIFT_PROFILES['INC-001']);

  const activeForecasts = liveResult ? liveResult.forecasts : profile.forecasts;
  const currentForecast = activeForecasts[selectedStepIndex] || activeForecasts[2];
  const activeOriginCoords = liveResult ? liveResult.originCoords : profile.originCoords;
  const activeOriginAreaKm2 = liveResult ? liveResult.originAreaKm2 : profile.originAreaKm2;
  const activeLandfallEta = liveResult ? liveResult.overallLandfallEta : profile.overallLandfallEta;
  const activeSuspectVessel = (liveResult && liveResult.suspectVessel) ? liveResult.suspectVessel : profile.suspectVessel;
  const activeConfidencePct = (liveResult && liveResult.backtrackConfidencePct) ? liveResult.backtrackConfidencePct : profile.backtrackConfidencePct;

  const handleRunSimulation = () => {
    setIsSimulating(true);
    setSimProgress(0);
    refreshLiveDriftData(selectedKey);
    const interval = setInterval(() => {
      setSimProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsSimulating(false);
          return 100;
        }
        return prev + 25;
      });
    }, 250);
  };

  const handleExportGeoJSON = () => {
    const geojson = {
      type: 'FeatureCollection',
      incident: profile.incidentName,
      scenarioId: profile.scenarioId,
      exportedAt: new Date().toISOString(),
      features: activeForecasts.map((f) => ({
        type: 'Feature',
        geometry: {
          type: 'Point',
          coordinates: [f.lng, f.lat],
        },
        properties: {
          timestep: f.label,
          timestamp: f.timestamp,
          area_km2: f.areaKm2,
          slick_radius_km: f.slickRadiusKm,
          drift_speed_knots: f.driftSpeedKnots,
          heading_deg: f.headingDeg,
          distance_to_coast_km: f.distanceToCoastKm,
          evaporated_pct: f.evaporatedPct,
          emulsified_pct: f.emulsifiedPct,
          dispersed_pct: f.dispersedPct,
          remaining_surface_pct: f.remainingSurfacePct,
          viscosity_cst: f.viscosityCSt,
          threat_level: f.threatLevel,
          recommendation: f.containmentRecommendation,
        },
      })),
    };

    const blob = new Blob([JSON.stringify(geojson, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `future_drift_trajectory_${profile.scenarioId}.geojson`;
    a.click();
    URL.revokeObjectURL(url);
    setExportNotice('Exported GeoJSON trajectory package.');
    setTimeout(() => setExportNotice(null), 3500);
  };

  return (
    <div id="tab-drift" className="tab-content visible modern-dashboard-root">
      {/* 1. EXECUTIVE HEADER */}
      <div
        className="workspace-header-bar"
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          gap: 16,
          flexWrap: 'nowrap',
          marginBottom: 18,
        }}
      >
        <div style={{ minWidth: 0, flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 3 }}>
            <span
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 5,
                padding: '2px 8px',
                borderRadius: 6,
                background: 'rgba(56, 189, 248, 0.12)',
                border: '1px solid rgba(56, 189, 248, 0.28)',
                color: 'var(--accent)',
                fontSize: 10,
                fontWeight: 700,
                letterSpacing: '0.04em',
                textTransform: 'uppercase',
              }}
            >
              <span className="material-symbols-outlined" style={{ fontSize: 13 }}>waves</span>
              Lagrangian Tracker (N=1,000)
            </span>
          </div>
          <h1 className="workspace-main-title" style={{ fontSize: 19, fontWeight: 800, margin: 0, letterSpacing: '-0.02em', whiteSpace: 'nowrap' }}>
            Hydrodynamic Drift Simulation
          </h1>
          <p className="workspace-sub-title" style={{ margin: '2px 0 0', fontSize: 11, color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
            OpenDrift &amp; OpenOil Framework · CMEMS Currents &amp; ERA5 Wind
          </p>
        </div>

        {/* STREAMLINED ACTION CONTROLS */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            flexWrap: 'nowrap',
            flexShrink: 0,
            background: 'var(--bg-surface)',
            padding: '4px 6px',
            borderRadius: 12,
            border: '1px solid var(--border-subtle)',
            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.04)',
          }}
        >
          {/* Scenario Select */}
          <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
            <span
              className="material-symbols-outlined"
              style={{
                position: 'absolute',
                left: 8,
                fontSize: 15,
                color: 'var(--accent)',
                pointerEvents: 'none',
              }}
            >
              emergency
            </span>
            <select
              value={selectedKey}
              onChange={(e) => {
                const k = e.target.value;
                setSelectedKey(k);
                setSelectedStepIndex(2);
                if (k !== 'SAR-DETECTION' && SCENARIOS[k]) {
                  onSelectScenario?.(k);
                }
              }}
              style={{
                height: 32,
                padding: '0 24px 0 28px',
                borderRadius: 8,
                fontSize: 11.5,
                fontWeight: 600,
                color: 'var(--text-primary)',
                background: 'var(--bg-raised)',
                border: '1px solid var(--border-default)',
                cursor: 'pointer',
                appearance: 'auto',
                outline: 'none',
                maxWidth: 200,
                whiteSpace: 'nowrap',
              }}
            >
              {activeSarPayload && (
                <option value="SAR-DETECTION">
                  🛰️ SAR AI · {activeSarPayload.fileName}
                </option>
              )}
              {Object.entries(SCENARIOS).map(([key, sc]) => {
                const shortId = sc.id.replace('2026-', '');
                const cleanTitle = sc.title.split('(')[0].replace('Offshore Basin', '').replace('Shipping Channel', '').trim();
                return (
                  <option key={key} value={key}>
                    {shortId} · {cleanTitle}
                  </option>
                );
              })}
            </select>
          </div>

          <div style={{ width: 1, height: 18, background: 'var(--border-subtle)' }} />

          {/* Sync Met-Ocean */}
          <button
            onClick={() => refreshLiveDriftData(selectedKey)}
            disabled={isLoadingLive}
            title="Fetch real-time ocean currents and wind from live Copernicus/Open-Meteo API"
            style={{
              height: 32,
              padding: '0 10px',
              borderRadius: 8,
              fontSize: 11,
              fontWeight: 600,
              display: 'inline-flex',
              alignItems: 'center',
              gap: 5,
              whiteSpace: 'nowrap',
              background: 'var(--bg-raised)',
              border: '1px solid var(--border-default)',
              color: 'var(--text-primary)',
              cursor: isLoadingLive ? 'not-allowed' : 'pointer',
              transition: 'all 0.15s ease',
            }}
          >
            <span
              className="material-symbols-outlined"
              style={{
                fontSize: 15,
                color: '#38bdf8',
                animation: isLoadingLive ? 'spin 1s linear infinite' : 'none',
              }}
            >
              satellite_alt
            </span>
            <span>{isLoadingLive ? 'Syncing...' : 'Sync Met-Ocean'}</span>
          </button>

          {/* Run Drift */}
          <button
            onClick={handleRunSimulation}
            disabled={isSimulating}
            style={{
              height: 32,
              padding: '0 10px',
              borderRadius: 8,
              fontSize: 11,
              fontWeight: 600,
              display: 'inline-flex',
              alignItems: 'center',
              gap: 5,
              whiteSpace: 'nowrap',
              background: 'var(--bg-raised)',
              border: '1px solid var(--border-default)',
              color: 'var(--text-primary)',
              cursor: isSimulating ? 'not-allowed' : 'pointer',
              transition: 'all 0.15s ease',
            }}
          >
            <span
              className="material-symbols-outlined"
              style={{
                fontSize: 15,
                color: 'var(--accent)',
                animation: isSimulating ? 'spin 1s linear infinite' : 'none',
              }}
            >
              {isSimulating ? 'sync' : 'play_circle'}
            </span>
            <span>{isSimulating ? `${simProgress}%` : 'Run Drift'}</span>
          </button>

          {/* Attribution */}
          <button
            onClick={() => onSelectTab('attribution')}
            style={{
              height: 32,
              padding: '0 12px',
              borderRadius: 8,
              fontSize: 11,
              fontWeight: 700,
              display: 'inline-flex',
              alignItems: 'center',
              gap: 5,
              whiteSpace: 'nowrap',
              background: 'var(--accent)',
              border: '1px solid transparent',
              color: '#FFFFFF',
              cursor: 'pointer',
              boxShadow: '0 2px 6px rgba(37, 99, 235, 0.25)',
              transition: 'all 0.15s ease',
            }}
          >
            <span>Attribution</span>
            <span className="material-symbols-outlined" style={{ fontSize: 14 }}>arrow_forward</span>
          </button>
        </div>
      </div>

      {exportNotice && (
        <div
          style={{
            margin: '0 0 12px',
            padding: '8px 14px',
            background: 'rgba(16, 185, 129, 0.12)',
            border: '1px solid rgba(16, 185, 129, 0.28)',
            borderRadius: 8,
            fontSize: 12,
            color: '#10b981',
            fontWeight: 600,
            display: 'flex',
            alignItems: 'center',
            gap: 8,
          }}
        >
          <span className="material-symbols-outlined" style={{ fontSize: 18 }}>check_circle</span>
          {exportNotice}
        </div>
      )}

      {/* 2. EXECUTIVE METRIC CARDS */}
      <div className="executive-metrics-grid" style={{ marginBottom: 10 }}>
        <div className="metric-card-neumorphic">
          <div className="metric-card-header">
            <span className="metric-card-label">Origin Probability Core</span>
          </div>
          <div className="metric-card-body">
            <span className="metric-number" style={{ fontSize: 19 }}>
              {activeOriginCoords.split(',')[0]}
              <span className="metric-unit" style={{ fontSize: 13, marginLeft: 4 }}>
                {activeOriginCoords.split(',')[1]}
              </span>
            </span>
            <span className="metric-trend-pill positive">
              50% Probability
            </span>
          </div>
          <div className="metric-card-footer">
            <span>{profile.originWindow}</span>
            <span className="material-symbols-outlined arrow-icon">history</span>
          </div>
        </div>

        <div className="metric-card-neumorphic">
          <div className="metric-card-header">
            <span className="metric-card-label">Dispersion Envelope</span>
          </div>
          <div className="metric-card-body">
            <span className="metric-number">
              {activeOriginAreaKm2} <span className="metric-unit">km²</span>
            </span>
            <span className="metric-trend-pill neutral">
              75% Uncertainty
            </span>
          </div>
          <div className="metric-card-footer">
            <span>N=1,000 Lagrangian Particles</span>
            <span className="material-symbols-outlined arrow-icon">grain</span>
          </div>
        </div>

        <div className="metric-card-neumorphic">
          <div className="metric-card-header">
            <span className="metric-card-label">Shoreline Landfall ETA</span>
          </div>
          <div className="metric-card-body">
            <span
              className="metric-number"
              style={{
                fontSize: 19,
                color: currentForecast.distanceToCoastKm < 15 ? '#ef4444' : 'inherit',
              }}
            >
              {activeLandfallEta.replace(/\(.*?\)/g, '').trim()}
            </span>
            <span className={`metric-trend-pill ${currentForecast.threatLevel === 'CRITICAL' ? 'positive' : 'neutral'}`} style={{ color: currentForecast.threatLevel === 'CRITICAL' ? '#ef4444' : undefined }}>
              {currentForecast.threatLevel} THREAT
            </span>
          </div>
          <div className="metric-card-footer">
            <span>{profile.coastalZoneName}</span>
            <span className="material-symbols-outlined arrow-icon">warning</span>
          </div>
        </div>

        <div className="metric-card-neumorphic">
          <div className="metric-card-header">
            <span className="metric-card-label">CMEMS Surface Jet</span>
          </div>
          <div className="metric-card-body">
            <span className="metric-number">
              {liveResult ? `${liveResult.metOcean.currentSpeedKnots}` : '0.82'} <span className="metric-unit">kn</span>
            </span>
            <span className="metric-trend-pill neutral">
              {liveResult ? `${liveResult.metOcean.currentCompassLabel} (${liveResult.metOcean.currentDirectionDeg}°)` : 'ENE @ 065°'}
            </span>
          </div>
          <div className="metric-card-footer">
            <span>Stokes Windage: {liveResult ? `${liveResult.metOcean.windSpeedKnots} kn` : '13.2 kn'}</span>
            <span className="material-symbols-outlined arrow-icon">air</span>
          </div>
        </div>
      </div>

      {/* 3. WORKFLOW NAV BAR */}
      <div
        className="workflow-nav-bar"
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          gap: 12,
          marginBottom: 0,
        }}
      >
        {/* Left: Title + Mode Tabs grouped together without gap */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
          <h2 className="workflow-title" style={{ margin: 0, whiteSpace: 'nowrap' }}>
            Simulation Horizons
          </h2>

          <div className="workflow-tabs-strip">
            <button
              className={`workflow-tab-btn ${activeMode === 'forward' ? 'active' : ''}`}
              onClick={() => setActiveMode('forward')}
            >
              Future Forecast (T0 → T+48h)
            </button>
            <button
              className={`workflow-tab-btn ${activeMode === 'backward' ? 'active' : ''}`}
              onClick={() => setActiveMode('backward')}
            >
              Origin Backtrack (T-24h → T0)
            </button>
            <button
              className={`workflow-tab-btn ${activeMode === 'unified' ? 'active' : ''}`}
              onClick={() => setActiveMode('unified')}
            >
              Unified Spatiotemporal
            </button>
          </div>
        </div>

        {/* Right: Active Scenario Context Chip + Export Button */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexShrink: 0 }}>
          <span
            className="scenario-chip"
            style={{
              borderColor: 'rgba(56, 189, 248, 0.4)',
              color: 'var(--accent)',
              padding: '3px 9px',
              fontSize: 10.5,
              whiteSpace: 'nowrap',
            }}
          >
            {profile.incidentName} · {currentForecast.label}
          </span>

          <button
            className="action-pill-btn secondary"
            onClick={handleExportGeoJSON}
            style={{ fontSize: 11, padding: '4px 12px', height: 30, whiteSpace: 'nowrap' }}
          >
            <span className="material-symbols-outlined" style={{ fontSize: 15 }}>download</span>
            <span>Export GeoJSON</span>
          </button>
        </div>
      </div>

      {/* 3.5 DYNAMIC SAR AI DETECTION PROVENANCE CARD */}
      {selectedKey === 'SAR-DETECTION' && activeSarPayload && (
        <div
          style={{
            margin: '0 0 16px',
            padding: '16px 20px',
            background: 'linear-gradient(135deg, rgba(239, 68, 68, 0.08) 0%, rgba(37, 99, 235, 0.08) 100%)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            borderRadius: 14,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 20,
            flexWrap: 'wrap',
            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.25)',
          }}
        >
          {/* Left: Thumbnail & Mask Previews */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
            <div style={{ position: 'relative', width: 72, height: 72, borderRadius: 10, overflow: 'hidden', border: '1px solid rgba(255, 255, 255, 0.2)', background: '#000', flexShrink: 0 }}>
              <img
                src={activeSarPayload.imageSrc}
                alt="Evaluated SAR Scene"
                style={{ width: '100%', height: '100%', objectFit: 'cover' }}
              />
              {activeSarPayload.maskSrc && (
                <img
                  src={activeSarPayload.maskSrc}
                  alt="SpillSegNet Mask"
                  style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', objectFit: 'cover', opacity: 0.85 }}
                />
              )}
            </div>

            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                <span
                  style={{
                    background: 'rgba(239, 68, 68, 0.2)',
                    color: '#ef4444',
                    border: '1px solid rgba(239, 68, 68, 0.4)',
                    fontSize: 10,
                    fontWeight: 700,
                    padding: '2px 8px',
                    borderRadius: 4,
                    textTransform: 'uppercase',
                    letterSpacing: '0.05em',
                  }}
                >
                  🛰️ Ingested SAR AI Detection
                </span>
                <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                  {new Date(activeSarPayload.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })} UTC
                </span>
              </div>
              <h3 style={{ margin: 0, fontSize: 15, fontWeight: 700, color: 'var(--text-primary)' }}>
                {activeSarPayload.fileName}
              </h3>
              <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginTop: 2 }}>
                Slick Coordinates: <span className="mono" style={{ color: 'var(--text-secondary)' }}>{activeSarPayload.lat.toFixed(3)}°N, {activeSarPayload.lng.toFixed(3)}°E</span> · {activeSarPayload.locationName}
              </div>
            </div>
          </div>

          {/* Middle: Ingested Metrics Telemetry - 100% Dynamic from Evaluated Image */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 20, flexWrap: 'wrap' }}>
            <div style={{ textAlign: 'center', minWidth: 90 }}>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Classification</div>
              <div style={{ fontSize: 17, fontWeight: 800, color: activeSarPayload.confidence > 0.85 ? '#ef4444' : '#f59e0b', marginTop: 2 }}>
                {(activeSarPayload.confidence * 100).toFixed(1)}%
              </div>
              <div style={{ fontSize: 9.5, color: 'var(--text-muted)' }}>{activeSarPayload.inferenceTimeMs}ms (DualPolNet)</div>
            </div>

            <div style={{ width: 1, height: 36, background: 'rgba(255, 255, 255, 0.1)' }} />

            <div style={{ textAlign: 'center', minWidth: 90 }}>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Mask Coverage</div>
              <div style={{ fontSize: 17, fontWeight: 800, color: '#f59e0b', marginTop: 2 }}>
                {activeSarPayload.spillAreaPercent.toFixed(1)}%
              </div>
              <div style={{ fontSize: 9.5, color: 'var(--text-muted)' }}>{activeSarPayload.segmentationTimeMs ? `${activeSarPayload.segmentationTimeMs}ms` : 'SpillSegNet'}</div>
            </div>

            <div style={{ width: 1, height: 36, background: 'rgba(255, 255, 255, 0.1)' }} />

            <div style={{ textAlign: 'center', minWidth: 100 }}>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Calculated Area</div>
              <div style={{ fontSize: 19, fontWeight: 800, color: 'var(--accent)', marginTop: 2 }}>
                {activeSarPayload.estimatedAreaKm2} <span style={{ fontSize: 12 }}>km²</span>
              </div>
              <div style={{ fontSize: 9.5, color: 'var(--text-muted)' }}>Initial Slick Surface</div>
            </div>
          </div>

          {/* Right: Quick Action to Re-evaluate or Switch back */}
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <button
              className="action-pill-btn secondary"
              onClick={() => onSelectTab('detection')}
              style={{ fontSize: 11, padding: '6px 14px' }}
              title="Return to SAR Detection Lab to inspect or select another scene"
            >
              <span className="material-symbols-outlined" style={{ fontSize: 15 }}>science</span>
              <span>Inspect in Lab</span>
            </button>
          </div>
        </div>
      )}

      {/* 3.5. INTERACTIVE HYDRODYNAMIC DRIFT TRAJECTORY & DISPERSION MAP */}
      <DriftTrajectoryMap
        simulation={liveResult}
        scenarioName={profile.incidentName}
        observedCoord={
          (selectedKey === 'SAR-DETECTION' && activeSarPayload)
            ? [activeSarPayload.lat, activeSarPayload.lng]
            : [
                (SCENARIOS[selectedKey] || currentScenario || SCENARIOS['INC-001']).lat,
                (SCENARIOS[selectedKey] || currentScenario || SCENARIOS['INC-001']).lng,
              ]
        }
        selectedStepIndex={selectedStepIndex}
        activeMode={activeMode}
        onSelectStepIndex={(idx) => setSelectedStepIndex(idx)}
        suspectVesselName={activeSuspectVessel}
        height={400}
      />

      {/* 4. ROUNDED CANVAS CONTAINER */}
      <div className="canvas-rounded-container" style={{ marginTop: 0 }}>
        <div className="canvas-two-column" style={{ paddingTop: 6 }}>
          {/* LEFT PANE: ACTIVE HYDRODYNAMIC SIMULATION */}
          <div className="canvas-pane">
            {(activeMode === 'forward' || activeMode === 'unified') && (
              <>
                <div className="pane-header">
                  <span className="pane-title">
                    <span className="material-symbols-outlined" style={{ fontSize: 18, color: 'var(--accent)' }}>
                      radar
                    </span>
                    Future Slick Dispersion &amp; Trajectory Forecast
                  </span>
                  <span
                    className="metric-trend-pill"
                    style={{
                      background:
                        currentForecast.threatLevel === 'CRITICAL'
                          ? 'rgba(239, 68, 68, 0.12)'
                          : 'rgba(245, 158, 11, 0.12)',
                      color: currentForecast.threatLevel === 'CRITICAL' ? '#ef4444' : '#f59e0b',
                      fontSize: 10,
                      fontWeight: 700,
                    }}
                  >
                    LANDFALL THREAT: {currentForecast.threatLevel}
                  </span>
                </div>

                {/* TIMESTEP STEP SELECTOR / SCRUBBER */}
                <div>
                  <div
                    style={{
                      fontSize: 11,
                      fontWeight: 700,
                      color: 'var(--text-muted)',
                      textTransform: 'uppercase',
                      marginBottom: 8,
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                    }}
                  >
                    <span style={{ letterSpacing: '0.04em' }}>Forecast Horizon Timeline (Forward Hydrodynamic Steps)</span>
                    <span style={{ color: 'var(--accent)', fontWeight: 700 }}>
                      Active: {currentForecast.label} ({currentForecast.timestamp})
                    </span>
                  </div>

                  <div className="drift-timestep-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 8 }}>
                    {activeForecasts.map((f, idx) => {
                      const isSelected = selectedStepIndex === idx;
                      const threatBg = f.threatLevel === 'CRITICAL' ? '#ef4444' : f.threatLevel === 'HIGH' ? '#f97316' : f.threatLevel === 'MODERATE' ? '#f59e0b' : '#10b981';
                      return (
                        <button
                          key={f.label}
                          onClick={() => setSelectedStepIndex(idx)}
                          className={`timeline-step-btn ${isSelected ? 'selected' : ''}`}
                          style={{
                            padding: '10px 8px',
                            borderRadius: 10,
                            border: isSelected ? '1.5px solid var(--accent)' : '1px solid var(--border-subtle)',
                            background: isSelected ? 'rgba(37, 99, 235, 0.12)' : 'var(--bg-raised)',
                            cursor: 'pointer',
                            textAlign: 'center',
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            gap: 3,
                          }}
                        >
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            <span
                              className={`timeline-pulse-dot ${isSelected ? 'pulsing' : ''}`}
                              style={{ width: 7, height: 7, background: threatBg }}
                            />
                            <span style={{ fontSize: 11.5, fontWeight: isSelected ? 800 : 700, color: isSelected ? 'var(--accent)' : 'var(--text-primary)' }}>
                              {f.label}
                            </span>
                          </div>
                          <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                            {f.areaKm2} km² · {f.distanceToCoastKm > 0 ? `${f.distanceToCoastKm} km` : 'Coast'}
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </div>

                {/* CRITICAL FUTURE SUMMARY METRICS */}
                <div className="drift-metrics-tri-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10 }}>
                  <div
                    style={{
                      background: 'var(--bg-raised)',
                      padding: '12px 14px',
                      borderRadius: 12,
                      border: '1px solid var(--border-subtle)',
                      boxShadow: '0 1px 4px rgba(0,0,0,0.02)',
                    }}
                  >
                    <div className="text-xs text-muted fw-600" style={{ letterSpacing: '0.03em' }}>Projected Centroid</div>
                    <div className="mono fw-700" style={{ fontSize: 14, color: 'var(--accent)', marginTop: 4 }}>
                      {currentForecast.lat.toFixed(3)}°N, {currentForecast.lng.toFixed(3)}°E
                    </div>
                    <div className="text-xs text-muted" style={{ marginTop: 3, fontSize: 10.5 }}>
                      Drift: <strong>{currentForecast.driftSpeedKnots} kn</strong> @ {currentForecast.headingDeg}°
                    </div>
                  </div>

                  <div
                    style={{
                      background: 'var(--bg-raised)',
                      padding: '12px 14px',
                      borderRadius: 12,
                      border: '1px solid var(--border-subtle)',
                      boxShadow: '0 1px 4px rgba(0,0,0,0.02)',
                    }}
                  >
                    <div className="text-xs text-muted fw-600" style={{ letterSpacing: '0.03em' }}>Fay Spreading Area</div>
                    <div className="mono fw-700" style={{ fontSize: 14, marginTop: 4 }}>
                      {currentForecast.areaKm2} km²
                      <span style={{ fontSize: 10.5, color: '#ef4444', marginLeft: 5, fontWeight: 700 }}>
                        (+{((currentForecast.areaKm2 / profile.initialAreaKm2 - 1) * 100).toFixed(0)}%)
                      </span>
                    </div>
                    <div className="text-xs text-muted" style={{ marginTop: 3, fontSize: 10.5 }}>
                      Slick Radius: <strong>{currentForecast.slickRadiusKm} km</strong>
                    </div>
                  </div>

                  <div
                    style={{
                      background: 'var(--bg-raised)',
                      padding: '12px 14px',
                      borderRadius: 12,
                      border: '1px solid var(--border-subtle)',
                      boxShadow: '0 1px 4px rgba(0,0,0,0.02)',
                    }}
                  >
                    <div className="text-xs text-muted fw-600" style={{ letterSpacing: '0.03em' }}>Shoreline Distance</div>
                    <div
                      className="mono fw-700"
                      style={{
                        fontSize: 14,
                        marginTop: 4,
                        color: currentForecast.distanceToCoastKm < 15 ? '#ef4444' : 'var(--text-primary)',
                      }}
                    >
                      {currentForecast.distanceToCoastKm > 0 ? `${currentForecast.distanceToCoastKm} km to coast` : 'SHORELINE IMPACT'}
                    </div>
                    <div className="text-xs text-muted" style={{ marginTop: 3, fontSize: 10.5 }}>
                      Zone: <strong>{profile.coastalZoneName.split('&')[0].trim()}</strong>
                    </div>
                  </div>
                </div>

                {/* COASTAL THREAT & HABITAT IMPACT BANNER */}
                <div
                  style={{
                    padding: '14px 16px',
                    borderRadius: 12,
                    background:
                      currentForecast.threatLevel === 'CRITICAL'
                        ? 'rgba(239, 68, 68, 0.08)'
                        : 'rgba(245, 158, 11, 0.08)',
                    borderLeft: `4px solid ${
                      currentForecast.threatLevel === 'CRITICAL' ? '#ef4444' : '#f59e0b'
                    }`,
                    borderTop: '1px solid rgba(245, 158, 11, 0.2)',
                    borderRight: '1px solid rgba(245, 158, 11, 0.2)',
                    borderBottom: '1px solid rgba(245, 158, 11, 0.2)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 8,
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span className="material-symbols-outlined" style={{ color: currentForecast.threatLevel === 'CRITICAL' ? '#ef4444' : '#f59e0b', fontSize: 18 }}>
                        warning
                      </span>
                      <span style={{ fontSize: 12.5, fontWeight: 700, color: 'var(--text-primary)' }}>
                        Coastal Threat Assessment
                      </span>
                    </div>
                    <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                      <span
                        style={{
                          fontSize: 10,
                          fontWeight: 700,
                          padding: '2px 8px',
                          borderRadius: 8,
                          background: currentForecast.threatLevel === 'CRITICAL' ? 'rgba(239, 68, 68, 0.18)' : 'rgba(245, 158, 11, 0.18)',
                          color: currentForecast.threatLevel === 'CRITICAL' ? '#ef4444' : '#b45309',
                        }}
                      >
                        ETA: {activeLandfallEta}
                      </span>
                      <span
                        style={{
                          fontSize: 10,
                          fontWeight: 700,
                          padding: '2px 8px',
                          borderRadius: 8,
                          background: 'rgba(59, 130, 246, 0.12)',
                          color: 'var(--accent)',
                        }}
                      >
                        CVI: HIGH
                      </span>
                    </div>
                  </div>

                  <div style={{ fontSize: 11, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                    Slick advection driven by combined CMEMS current ({profile.currentVector}) and ERA5 Stokes windage ({profile.windVector}).
                    Projected trajectory targets the sensitive intertidal shelf zone: <strong>{profile.coastalZoneName}</strong>.
                  </div>

                  <div>
                    <div style={{ fontSize: 10.5, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                      Vulnerable Marine Receptors in Cone:
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                      {profile.vulnerableHabitats.map((hab, idx) => (
                        <span
                          key={idx}
                          style={{
                            fontSize: 10.5,
                            background: 'var(--bg-surface)',
                            border: '1px solid var(--border-subtle)',
                            padding: '3px 8px',
                            borderRadius: 6,
                            color: 'var(--text-primary)',
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: 4,
                          }}
                        >
                          <span style={{ color: 'var(--accent)', fontSize: 12 }}>•</span>
                          {hab}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>

                {/* OIL WEATHERING & MASS BALANCE */}
                <div style={{ background: 'var(--bg-raised)', padding: '12px 14px', borderRadius: 12, border: '1px solid var(--border-subtle)' }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 8, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <span className="material-symbols-outlined" style={{ fontSize: 16, color: 'var(--accent)' }}>water</span>
                      <span>OpenOil Weathering &amp; Mass Balance at {currentForecast.label}</span>
                    </div>
                    <div style={{ display: 'flex', gap: 8, fontSize: 10.5 }}>
                      <span style={{ background: 'var(--bg-surface)', padding: '2px 6px', borderRadius: 4, border: '1px solid var(--border-subtle)' }}>
                        Viscosity: <strong className="mono">{currentForecast.viscosityCSt} cSt</strong>
                      </span>
                      <span style={{ background: 'var(--bg-surface)', padding: '2px 6px', borderRadius: 4, border: '1px solid var(--border-subtle)' }}>
                        Water Content: <strong className="mono">{currentForecast.waterContentPct}%</strong>
                      </span>
                    </div>
                  </div>

                  <div
                    style={{
                      height: 18,
                      borderRadius: 6,
                      overflow: 'hidden',
                      display: 'flex',
                      background: 'var(--bg-base)',
                      border: '1px solid var(--border-subtle)',
                      marginBottom: 10,
                      boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.1)',
                    }}
                  >
                    <div title="Evaporated" style={{ width: `${currentForecast.evaporatedPct}%`, background: '#38bdf8', transition: 'width 0.3s ease' }} />
                    <div title="Emulsified Mousse" style={{ width: `${currentForecast.emulsifiedPct}%`, background: '#d97706', transition: 'width 0.3s ease' }} />
                    <div title="Naturally Dispersed" style={{ width: `${currentForecast.dispersedPct}%`, background: '#10b981', transition: 'width 0.3s ease' }} />
                    <div title="Persistent Surface Slick" style={{ width: `${currentForecast.remainingSurfacePct}%`, background: '#ef4444', transition: 'width 0.3s ease' }} />
                  </div>

                  <div className="drift-weathering-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8, fontSize: 10.5 }}>
                    <div style={{ background: 'var(--bg-surface)', padding: '6px 10px', borderRadius: 6, border: '1px solid var(--border-subtle)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 6 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6, minWidth: 0 }}>
                        <span style={{ width: 8, height: 8, borderRadius: 2, background: '#38bdf8', flexShrink: 0 }} />
                        <span className="text-muted" style={{ whiteSpace: 'nowrap' }}>Evaporated:</span>
                      </div>
                      <strong className="mono">{currentForecast.evaporatedPct}%</strong>
                    </div>
                    <div style={{ background: 'var(--bg-surface)', padding: '6px 10px', borderRadius: 6, border: '1px solid var(--border-subtle)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 6 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6, minWidth: 0 }}>
                        <span style={{ width: 8, height: 8, borderRadius: 2, background: '#d97706', flexShrink: 0 }} />
                        <span className="text-muted" style={{ whiteSpace: 'nowrap' }}>Emulsified:</span>
                      </div>
                      <strong className="mono">{currentForecast.emulsifiedPct}%</strong>
                    </div>
                    <div style={{ background: 'var(--bg-surface)', padding: '6px 10px', borderRadius: 6, border: '1px solid var(--border-subtle)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 6 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6, minWidth: 0 }}>
                        <span style={{ width: 8, height: 8, borderRadius: 2, background: '#10b981', flexShrink: 0 }} />
                        <span className="text-muted" style={{ whiteSpace: 'nowrap' }}>Dispersed:</span>
                      </div>
                      <strong className="mono">{currentForecast.dispersedPct}%</strong>
                    </div>
                    <div style={{ background: 'var(--bg-surface)', padding: '6px 10px', borderRadius: 6, border: '1px solid var(--border-subtle)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 6 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6, minWidth: 0 }}>
                        <span style={{ width: 8, height: 8, borderRadius: 2, background: '#ef4444', flexShrink: 0 }} />
                        <span className="text-muted" style={{ whiteSpace: 'nowrap' }}>Surface Slick:</span>
                      </div>
                      <strong className="mono">{currentForecast.remainingSurfacePct}%</strong>
                    </div>
                  </div>
                </div>

                {/* TACTICAL COUNTERMEASURE RECOMMENDATION */}
                <div
                  style={{
                    background: 'linear-gradient(135deg, rgba(37, 99, 235, 0.08) 0%, rgba(16, 185, 129, 0.06) 100%)',
                    padding: '12px 14px',
                    borderRadius: 10,
                    border: '1px solid rgba(37, 99, 235, 0.25)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 12,
                  }}
                >
                  <span className="material-symbols-outlined" style={{ fontSize: 24, color: 'var(--accent)' }}>
                    shield
                  </span>
                  <div>
                    <div style={{ fontSize: 11.5, fontWeight: 700, color: 'var(--text-primary)' }}>
                      Tactical Containment Directive:
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 2 }}>
                      {currentForecast.containmentRecommendation}
                    </div>
                  </div>
                </div>
              </>
            )}

            {/* BACKWARD ORIGIN ENVELOPE ANALYSIS */}
            {(activeMode === 'backward' || activeMode === 'unified') && (
              <div style={{ marginTop: activeMode === 'unified' ? 14 : 0, display: 'flex', flexDirection: 'column', gap: 12 }}>
                <div className="pane-header">
                  <span className="pane-title">
                    <span className="material-symbols-outlined" style={{ fontSize: 18, color: '#f59e0b' }}>
                      history
                    </span>
                    Origin Probability Envelope Analysis (Reverse Time)
                  </span>
                  <span
                    className="metric-trend-pill"
                    style={{ background: 'rgba(217,119,6,.12)', color: '#d97706', fontSize: 10, fontWeight: 700 }}
                  >
                    PHYSICS BACKTRACKING
                  </span>
                </div>

                <div
                  style={{
                    background: 'rgba(245, 158, 11, 0.08)',
                    borderLeft: '4px solid #f59e0b',
                    padding: '10px 12px',
                    borderRadius: 8,
                    fontSize: 11,
                    color: 'var(--text-secondary)',
                    lineHeight: 1.5,
                  }}
                >
                  <strong>Operational Hydrodynamic Rule:</strong> Detected slick position is never the discharge point.
                  OpenDrift reverses the advection-diffusion equation to isolate the exact space-time envelope where discharge statistically occurred.
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10 }}>
                  <div style={{ background: 'var(--bg-raised)', padding: 10, borderRadius: 8, border: '1px solid var(--border-subtle)' }}>
                    <div className="text-xs text-muted fw-600">50% Core Probability</div>
                    <div className="text-sm fw-700" style={{ color: '#d97706', marginTop: 2 }}>{activeOriginCoords}</div>
                    <div className="text-xs text-muted" style={{ fontSize: 10 }}>{profile.originWindow}</div>
                  </div>
                  <div style={{ background: 'var(--bg-raised)', padding: 10, borderRadius: 8, border: '1px solid var(--border-subtle)' }}>
                    <div className="text-xs text-muted fw-600">75% Probability Area</div>
                    <div className="text-sm fw-700" style={{ marginTop: 2 }}>{activeOriginAreaKm2} km²</div>
                    <div className="text-xs text-muted" style={{ fontSize: 10 }}>Spatiotemporal uncertainty</div>
                  </div>
                  <div style={{ background: 'var(--bg-raised)', padding: 10, borderRadius: 8, border: '1px solid var(--border-subtle)' }}>
                    <div className="text-xs text-muted fw-600">Total Particles</div>
                    <div className="text-sm fw-700" style={{ marginTop: 2 }}>1,000 Lagrangian</div>
                    <div className="text-xs text-muted" style={{ fontSize: 10 }}>Runge-Kutta 4th Order</div>
                  </div>
                </div>

                {/* ATTRIBUTED SUSPECT VESSEL MATCH CARD */}
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '10px 14px',
                    background: 'var(--bg-raised)',
                    borderRadius: 10,
                    border: '1px solid var(--border-subtle)',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <span className="material-symbols-outlined" style={{ fontSize: 20, color: '#ef4444' }}>
                      directions_boat
                    </span>
                    <div>
                      <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-primary)' }}>
                        Attributed Discharge Suspect: {activeSuspectVessel}
                      </div>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                        Backtrack Intersection Confidence: {activeConfidencePct}% · Spatiotemporal correlation confirmed
                      </div>
                    </div>
                  </div>
                  <button
                    className="action-pill-btn secondary"
                    onClick={() => onSelectTab('attribution')}
                    style={{ padding: '4px 10px', fontSize: 11, gap: 4 }}
                  >
                    <span>View AIS Track</span>
                    <span className="material-symbols-outlined" style={{ fontSize: 14 }}>arrow_forward</span>
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* RIGHT PANE: HYDRODYNAMIC ENGINE PARAMETERS & OCEAN FORCING */}
          <div className="canvas-pane">
            <div className="pane-header">
              <span className="pane-title">
                <span className="material-symbols-outlined" style={{ fontSize: 18, color: 'var(--accent)' }}>tune</span>
                Hydrodynamic Engine Parameters
              </span>
              <span className="metric-trend-pill neutral" style={{ fontSize: 10 }}>
                ODE SOLVER: RK4
              </span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              <div style={{ background: 'var(--bg-raised)', padding: '12px 14px', borderRadius: 10, border: '1px solid var(--border-subtle)' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span className="text-xs text-muted fw-600">Integration Step</span>
                  <span className="material-symbols-outlined" style={{ fontSize: 16, color: 'var(--text-muted)' }}>timer</span>
                </div>
                <div className="mono fw-700" style={{ fontSize: 14, marginTop: 4, color: 'var(--text-primary)' }}>15 minutes</div>
              </div>
              <div style={{ background: 'var(--bg-raised)', padding: '12px 14px', borderRadius: 10, border: '1px solid var(--border-subtle)' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span className="text-xs text-muted fw-600">Wind Drag Coeff</span>
                  <span className="material-symbols-outlined" style={{ fontSize: 16, color: 'var(--text-muted)' }}>air</span>
                </div>
                <div className="mono fw-700" style={{ fontSize: 14, marginTop: 4, color: 'var(--text-primary)' }}>3.5% (Stokes Drift)</div>
              </div>
              <div style={{ background: 'var(--bg-raised)', padding: '12px 14px', borderRadius: 10, border: '1px solid var(--border-subtle)' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span className="text-xs text-muted fw-600">Horiz Diffusivity</span>
                  <span className="material-symbols-outlined" style={{ fontSize: 16, color: 'var(--text-muted)' }}>grain</span>
                </div>
                <div className="mono fw-700" style={{ fontSize: 14, marginTop: 4, color: 'var(--text-primary)' }}>10 m²/s</div>
              </div>
              <div style={{ background: 'var(--bg-raised)', padding: '12px 14px', borderRadius: 10, border: '1px solid var(--border-subtle)' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span className="text-xs text-muted fw-600">Current Depth</span>
                  <span className="material-symbols-outlined" style={{ fontSize: 16, color: 'var(--text-muted)' }}>waves</span>
                </div>
                <div className="mono fw-700" style={{ fontSize: 14, marginTop: 4, color: 'var(--text-primary)' }}>0.0 – 1.0 m (Ekman)</div>
              </div>
            </div>

            {/* ACTIVE ENVIRONMENTAL FORCING */}
            <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: 14 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
                <div style={{ fontSize: 11.5, fontWeight: 700, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span className="material-symbols-outlined" style={{ fontSize: 16, color: 'var(--accent)' }}>public</span>
                  <span>Active Environmental Forcing:</span>
                </div>
                <span style={{ fontSize: 10.5, color: liveResult ? '#10b981' : '#f59e0b', fontWeight: 700, background: liveResult ? 'rgba(16, 185, 129, 0.1)' : 'rgba(245, 158, 11, 0.1)', padding: '2px 8px', borderRadius: 12 }}>
                  {liveResult ? `🟢 Live API: ${liveResult.metOcean.source.split('(')[0].trim()}` : '🟡 Calibrated CMEMS'}
                </span>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', borderRadius: 10, overflow: 'hidden', border: '1px solid var(--border-subtle)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 12px', background: 'var(--bg-surface)', borderBottom: '1px solid var(--border-subtle)', fontSize: 11 }}>
                  <span className="text-muted fw-600">CMEMS Current:</span>
                  <span className="mono fw-600">
                    {liveResult
                      ? `${liveResult.metOcean.currentSpeedMs} m/s (${liveResult.metOcean.currentSpeedKnots} kn) @ ${liveResult.metOcean.currentDirectionDeg}°`
                      : profile.currentVector}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 12px', background: 'var(--bg-raised)', borderBottom: '1px solid var(--border-subtle)', fontSize: 11 }}>
                  <span className="text-muted fw-600">ERA5 10m Wind:</span>
                  <span className="mono fw-600">
                    {liveResult
                      ? `${liveResult.metOcean.windSpeedMs} m/s (${liveResult.metOcean.windSpeedKnots} kn) @ ${liveResult.metOcean.windDirectionDeg}°`
                      : profile.windVector}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 12px', background: 'var(--bg-surface)', borderBottom: '1px solid var(--border-subtle)', fontSize: 11 }}>
                  <span className="text-muted fw-600">Sea State &amp; Temp:</span>
                  <span className="mono fw-600">
                    {liveResult
                      ? `${liveResult.metOcean.temperatureCelsius}°C · ${liveResult.metOcean.seaStateDescription}`
                      : `${profile.sstCelsius}°C (${profile.seaState})`}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 12px', background: 'rgba(37, 99, 235, 0.06)', borderBottom: '1px solid var(--border-subtle)', fontSize: 11 }}>
                  <span className="fw-700" style={{ color: 'var(--accent)' }}>Net Drift Vector:</span>
                  <span className="mono fw-700" style={{ color: '#FFFFFF', background: 'var(--accent)', padding: '2px 8px', borderRadius: 6, fontSize: 11.5 }}>
                    {liveResult
                      ? `${liveResult.metOcean.netDriftSpeedKnots} kn @ ${liveResult.metOcean.netDriftHeadingDeg}°`
                      : `${profile.forecasts[0].driftSpeedKnots} kn @ ${profile.forecasts[0].headingDeg}°`}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 12px', background: 'var(--bg-surface)', fontSize: 11 }}>
                  <span className="text-muted fw-600">Oil Grade:</span>
                  <span className="mono fw-600" style={{ color: 'var(--text-primary)' }}>{profile.oilType} ({profile.apiGravity})</span>
                </div>
              </div>
            </div>

            {/* SIMULATION TRAJECTORY TABLE */}
            <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: 14 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
                <div style={{ fontSize: 11.5, fontWeight: 700, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span className="material-symbols-outlined" style={{ fontSize: 16, color: 'var(--accent)' }}>route</span>
                  <span>Simulation Trajectory Waypoints (5 Steps)</span>
                </div>
                <span className="text-xs text-muted" style={{ fontSize: 10.5 }}>Click any row to jump timestep</span>
              </div>

              <div style={{ overflowX: 'auto', borderRadius: 10, border: '1px solid var(--border-subtle)', boxShadow: '0 1px 4px rgba(0,0,0,0.02)' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                  <thead>
                    <tr style={{ background: 'var(--bg-raised)', borderBottom: '1px solid var(--border-subtle)', textAlign: 'left' }}>
                      <th style={{ padding: '8px 10px', color: 'var(--text-muted)', fontWeight: 700, fontSize: 10, textTransform: 'uppercase' }}>Step</th>
                      <th style={{ padding: '8px 10px', color: 'var(--text-muted)', fontWeight: 700, fontSize: 10, textTransform: 'uppercase' }}>Coordinates</th>
                      <th style={{ padding: '8px 10px', color: 'var(--text-muted)', fontWeight: 700, fontSize: 10, textTransform: 'uppercase' }}>Area</th>
                      <th style={{ padding: '8px 10px', color: 'var(--text-muted)', fontWeight: 700, fontSize: 10, textTransform: 'uppercase' }}>Coast</th>
                      <th style={{ padding: '8px 10px', color: 'var(--text-muted)', fontWeight: 700, fontSize: 10, textTransform: 'uppercase' }}>Threat</th>
                    </tr>
                  </thead>
                  <tbody>
                    {activeForecasts.map((f, idx) => {
                      const isSelected = selectedStepIndex === idx;
                      return (
                        <tr
                          key={f.label}
                          onClick={() => setSelectedStepIndex(idx)}
                          style={{
                            borderBottom: idx < activeForecasts.length - 1 ? '1px solid var(--border-subtle)' : 'none',
                            cursor: 'pointer',
                            background: isSelected ? 'rgba(37, 99, 235, 0.12)' : idx % 2 === 0 ? 'var(--bg-surface)' : 'var(--bg-raised)',
                            transition: 'background 0.15s ease',
                          }}
                        >
                          <td style={{ padding: '8px 10px', fontWeight: isSelected ? 800 : 600, color: isSelected ? 'var(--accent)' : 'var(--text-primary)' }}>
                            {f.label}
                          </td>
                          <td style={{ padding: '8px 10px', fontFamily: 'monospace', fontSize: 10.5 }}>
                            {f.lat.toFixed(2)}°, {f.lng.toFixed(2)}°
                          </td>
                          <td style={{ padding: '8px 10px', fontWeight: 600 }}>{f.areaKm2} km²</td>
                          <td style={{ padding: '8px 10px', color: f.distanceToCoastKm < 15 ? '#ef4444' : 'inherit' }}>
                            {f.distanceToCoastKm > 0 ? `${f.distanceToCoastKm} km` : '0 km'}
                          </td>
                          <td style={{ padding: '8px 10px' }}>
                            <span
                              className="metric-trend-pill"
                              style={{
                                fontSize: 9.5,
                                padding: '2px 6px',
                                borderRadius: 6,
                                fontWeight: 700,
                                background:
                                  f.threatLevel === 'CRITICAL'
                                    ? 'rgba(239, 68, 68, 0.14)'
                                    : f.threatLevel === 'HIGH'
                                    ? 'rgba(249, 115, 22, 0.14)'
                                    : f.threatLevel === 'MODERATE'
                                    ? 'rgba(245, 158, 11, 0.14)'
                                    : 'rgba(16, 185, 129, 0.14)',
                                color:
                                  f.threatLevel === 'CRITICAL'
                                    ? '#ef4444'
                                    : f.threatLevel === 'HIGH'
                                    ? '#f97316'
                                    : f.threatLevel === 'MODERATE'
                                    ? '#d97706'
                                    : '#10b981',
                              }}
                            >
                              {f.threatLevel}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

