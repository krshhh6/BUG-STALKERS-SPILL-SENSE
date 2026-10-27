/**
 * GlobeView — 3D Tactical Common Operating Picture (COP) for Spill Sense
 *
 * Integrated with God's Eye View (GEV) capabilities:
 * - Cesium Ion 3D Photorealistic Cloud + Cesium World Terrain
 * - Real-time AISStream.io live vessel tracking across the Indian Ocean & EEZ
 * - Real Sentinel-1 SAR imagery draping & Monte Carlo Lagrangian drift envelopes
 * - In-app C2 API Keys & Live Feeds configuration
 * - Google Earth-grade 3D camera orientation, navigation inertia, and 360° turntable orbit
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import 'cesium/Build/Cesium/Widgets/widgets.css';
import type { Scenario } from '../../types/dashboard';
import type { LiveIncident } from '../../hooks/useIncidents';
import {
  computeBackwardDriftGeometry,
  createAoiForScenario,
  type DriftGeometry,
} from '../../utils/geoContours';
import {
  DEFAULT_SCENARIO_VESSELS,
  type CandidateVesselItem,
} from './AttributionView';
import { ApiKeySetupModal } from '../modals/ApiKeySetupModal';

import * as Cesium from 'cesium';

if (typeof window !== 'undefined') {
  (window as Window & { CESIUM_BASE_URL?: string }).CESIUM_BASE_URL = '/cesium/';
}

// ─── Component Props ──────────────────────────────────────────────────────────
interface GlobeViewProps {
  currentScenario: Scenario | null;
  incidents: LiveIncident[];
  scenarios: Record<string, Scenario>;
  onSelectTab?: (tab: import('../../types/dashboard').TabType) => void;
}

type BasemapId = 'satellite' | 'ion' | 'dark' | 'marine';

// ─── Geographic Distance Helper (Haversine km) ─────────────────────────────────
function computeDistanceKm(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const R = 6371;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

// ─── Native Canvas 2D Rounded Rectangle Helper ──────────────────────────────
function drawRoundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number
) {
  if (typeof ctx.roundRect === 'function') {
    ctx.roundRect(x, y, w, h, r);
  } else {
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
  }
}

// ─── Glassmorphic Vessel Callout Card Generator (High-DPI Retina) ────────────
const vesselCardCache = new Map<string, { dataUrl: string; width: number; height: number }>();

function getGlassmorphicVesselCard(
  name: string,
  type: string,
  sog: number,
  _cog: number,
  risk: string,
  isTanker: boolean
): { dataUrl: string; width: number; height: number } {
  const cacheKey = `${name}_${type}_${sog.toFixed(1)}_${risk}_${isTanker}`;
  if (vesselCardCache.has(cacheKey)) {
    return vesselCardCache.get(cacheKey)!;
  }

  // Clean vessel type text
  let cleanType = type || 'Vessel';
  cleanType = cleanType
    .replace(' / Commercial Vessel', '')
    .replace(' / Ore Freighter', '')
    .replace('Commercial Vessel', 'Cargo');

  const scale = 2; // Retina supersampling
  const cardWidth = 178;
  const cardHeight = 44;
  const radius = 8;

  const canvas = document.createElement('canvas');
  canvas.width = cardWidth * scale;
  canvas.height = cardHeight * scale;
  const ctx = canvas.getContext('2d')!;
  ctx.scale(scale, scale);

  // A. Frosted Deep-Ocean Glass Gradient
  ctx.beginPath();
  drawRoundRect(ctx, 1, 1, cardWidth - 2, cardHeight - 2, radius);
  const bgGrad = ctx.createLinearGradient(0, 0, 0, cardHeight);
  bgGrad.addColorStop(0, 'rgba(10, 18, 36, 0.84)');
  bgGrad.addColorStop(1, 'rgba(4, 9, 20, 0.92)');
  ctx.fillStyle = bgGrad;
  ctx.fill();

  // B. Glass Rim Bevel Highlight (Optical Refraction)
  ctx.beginPath();
  drawRoundRect(ctx, 1.5, 1.5, cardWidth - 3, cardHeight - 3, radius);
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.18)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // C. Risk-Toned Luminous Border
  ctx.beginPath();
  drawRoundRect(ctx, 0.5, 0.5, cardWidth - 1, cardHeight - 1, radius);
  const borderGrad = ctx.createLinearGradient(0, 0, cardWidth, 0);
  if (isTanker || risk === 'CRITICAL') {
    borderGrad.addColorStop(0, 'rgba(239, 68, 68, 0.85)');
    borderGrad.addColorStop(1, 'rgba(239, 68, 68, 0.25)');
  } else if (risk === 'HIGH') {
    borderGrad.addColorStop(0, 'rgba(249, 115, 22, 0.80)');
    borderGrad.addColorStop(1, 'rgba(249, 115, 22, 0.20)');
  } else {
    borderGrad.addColorStop(0, 'rgba(56, 189, 248, 0.70)');
    borderGrad.addColorStop(1, 'rgba(56, 189, 248, 0.20)');
  }
  ctx.strokeStyle = borderGrad;
  ctx.lineWidth = 1;
  ctx.stroke();

  // D. Live Pulsing Status Beacon
  const dotX = 11;
  const dotY = 14;
  ctx.beginPath();
  ctx.arc(dotX, dotY, 3, 0, 2 * Math.PI);
  const dotColor = isTanker ? '#ef4444' : '#22c55e';
  ctx.fillStyle = dotColor;
  ctx.shadowColor = dotColor;
  ctx.shadowBlur = 6;
  ctx.fill();
  ctx.shadowColor = 'transparent';

  // E. Title (Vessel Name)
  ctx.font = 'bold 11px Inter, system-ui, -apple-system, sans-serif';
  ctx.fillStyle = '#ffffff';
  const cleanName = name.replace('MMSI: ', '');
  const displayName = cleanName.length > 18 ? cleanName.slice(0, 17) + '…' : cleanName;
  ctx.fillText(displayName, dotX + 8, dotY + 4);

  // F. Subtitle (Type & Speed)
  ctx.font = '500 9.5px Inter, system-ui, -apple-system, sans-serif';
  ctx.fillStyle = '#94a3b8';
  const speedStr = `${sog.toFixed(1)} kn`;
  const subText = `${cleanType} · ${speedStr}`;
  const displaySub = subText.length > 27 ? subText.slice(0, 26) + '…' : subText;
  ctx.fillText(displaySub, 11, dotY + 18);

  const result = {
    dataUrl: canvas.toDataURL('image/png'),
    width: cardWidth,
    height: cardHeight,
  };
  vesselCardCache.set(cacheKey, result);
  return result;
}

// ─── Glassmorphic Incident / Spill Badge Card Generator ──────────────────────
const badgeCardCache = new Map<string, { dataUrl: string; width: number; height: number }>();

function getGlassmorphicBadgeCard(
  icon: string,
  title: string,
  subtitle: string,
  accentColor: string
): { dataUrl: string; width: number; height: number } {
  const cacheKey = `${icon}_${title}_${subtitle}_${accentColor}`;
  if (badgeCardCache.has(cacheKey)) {
    return badgeCardCache.get(cacheKey)!;
  }

  const scale = 2;
  const cardWidth = 186;
  const cardHeight = 44;
  const radius = 8;

  const canvas = document.createElement('canvas');
  canvas.width = cardWidth * scale;
  canvas.height = cardHeight * scale;
  const ctx = canvas.getContext('2d')!;
  ctx.scale(scale, scale);

  // A. Frosted Glass Background
  ctx.beginPath();
  drawRoundRect(ctx, 1, 1, cardWidth - 2, cardHeight - 2, radius);
  const bgGrad = ctx.createLinearGradient(0, 0, 0, cardHeight);
  bgGrad.addColorStop(0, 'rgba(18, 14, 30, 0.86)');
  bgGrad.addColorStop(1, 'rgba(8, 6, 16, 0.94)');
  ctx.fillStyle = bgGrad;
  ctx.fill();

  // B. Bevel Highlight
  ctx.beginPath();
  drawRoundRect(ctx, 1.5, 1.5, cardWidth - 3, cardHeight - 3, radius);
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.16)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // C. Accent Tinted Border
  ctx.beginPath();
  drawRoundRect(ctx, 0.5, 0.5, cardWidth - 1, cardHeight - 1, radius);
  ctx.strokeStyle = accentColor;
  ctx.lineWidth = 1;
  ctx.stroke();

  // D. Icon / Dot
  const dotX = 11;
  const dotY = 14;
  ctx.beginPath();
  ctx.arc(dotX, dotY, 3.5, 0, 2 * Math.PI);
  ctx.fillStyle = accentColor;
  ctx.shadowColor = accentColor;
  ctx.shadowBlur = 6;
  ctx.fill();
  ctx.shadowColor = 'transparent';

  // E. Title
  ctx.font = 'bold 11px Inter, system-ui, -apple-system, sans-serif';
  ctx.fillStyle = '#ffffff';
  const displayTitle = title.length > 20 ? title.slice(0, 19) + '…' : title;
  ctx.fillText(displayTitle, dotX + 8, dotY + 4);

  // F. Subtitle
  ctx.font = '500 9.5px Inter, system-ui, -apple-system, sans-serif';
  ctx.fillStyle = '#cbd5e1';
  const displaySub = subtitle.length > 28 ? subtitle.slice(0, 27) + '…' : subtitle;
  ctx.fillText(displaySub, 11, dotY + 18);

  const result = {
    dataUrl: canvas.toDataURL('image/png'),
    width: cardWidth,
    height: cardHeight,
  };
  badgeCardCache.set(cacheKey, result);
  return result;
}


function getGlassmorphicIncidentCard(title: string, subtitle: string) {
  return getGlassmorphicBadgeCard('📍', title, subtitle, '#f97316');
}

// ─── High-DPI Canvas Vessel Pin Generator (100% Reliable Hardware WebGL Texture) ──
const vesselPinCache = new Map<string, { dataUrl: string; width: number; height: number }>();

function getVesselPinBillboard(risk: string, headingDeg: number): { dataUrl: string; width: number; height: number } {
  const roundedHeading = Math.round(((headingDeg || 0) % 360) / 4) * 4;
  const cacheKey = `${risk}_${roundedHeading}`;
  if (vesselPinCache.has(cacheKey)) {
    return vesselPinCache.get(cacheKey)!;
  }

  const canvas = document.createElement('canvas');
  const size = 32;
  const dpr = 2;
  canvas.width = size * dpr;
  canvas.height = size * dpr;
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    return { dataUrl: '', width: size, height: size };
  }
  ctx.scale(dpr, dpr);

  const color =
    risk === 'CRITICAL' ? '#ef4444' : risk === 'HIGH' ? '#f97316' : risk === 'MEDIUM' ? '#38bdf8' : '#10b981';
  const center = size / 2;

  // Outer glow aura
  ctx.beginPath();
  ctx.arc(center, center, 12.5, 0, 2 * Math.PI);
  ctx.fillStyle = '#0a1020';
  ctx.shadowColor = color;
  ctx.shadowBlur = 8;
  ctx.fill();

  // Crisp border
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.stroke();
  ctx.shadowColor = 'transparent';

  // Inner directional indicator arrow
  ctx.save();
  ctx.translate(center, center);
  ctx.rotate(((headingDeg || 0) * Math.PI) / 180);
  ctx.beginPath();
  ctx.moveTo(0, -7.5);
  ctx.lineTo(5, 6);
  ctx.lineTo(0, 3);
  ctx.lineTo(-5, 6);
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.restore();

  const result = {
    dataUrl: canvas.toDataURL('image/png'),
    width: size,
    height: size,
  };
  vesselPinCache.set(cacheKey, result);
  return result;
}

function createVesselSvgPin(risk: string, headingDeg: number): string {
  return getVesselPinBillboard(risk, headingDeg).dataUrl;
}


// ─── Sentinel-1 Orbital Ground Track Generator ───────────────────────────────
function generateSentinelSwath(
  lat: number,
  lng: number
): {
  track: Array<[number, number]>;
  swathLeft: Array<[number, number]>;
  swathRight: Array<[number, number]>;
} {
  const track: Array<[number, number]> = [];
  const swathLeft: Array<[number, number]> = [];
  const swathRight: Array<[number, number]> = [];

  const halfSwathDeg = 1.15; // ~250 km IW mode swath width (~125 km on each side)
  for (let i = -16; i <= 16; i++) {
    const pLat = lat + i * 0.7;
    const pLng = lng - (pLat - lat) * 0.165;
    track.push([pLat, pLng]);
    swathLeft.push([pLat, pLng - halfSwathDeg]);
    swathRight.push([pLat, pLng + halfSwathDeg]);
  }

  return { track, swathLeft, swathRight };
}

// ─── Layer DataSources Bundle ─────────────────────────────────────────────────
interface LayerDataSources {
  slickDS: Cesium.CustomDataSource;
  driftDS: Cesium.CustomDataSource;
  vesselDS: Cesium.CustomDataSource;
  liveVesselDS: Cesium.CustomDataSource;
  swathDS: Cesium.CustomDataSource;
  sarDS: Cesium.CustomDataSource;
  incidentDS: Cesium.CustomDataSource;
}

// ─── Main Component ───────────────────────────────────────────────────────────
export const GlobeView: React.FC<GlobeViewProps> = ({
  currentScenario,
  incidents,
  scenarios,
  onSelectTab,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<Cesium.Viewer | null>(null);
  const layerSourcesRef = useRef<LayerDataSources | null>(null);
  const eezDSRef = useRef<Cesium.GeoJsonDataSource | null>(null);
  const baseLayerRef = useRef<Cesium.ImageryLayer | null>(null);
  const currentBasemapRef = useRef<BasemapId>('satellite');
  const orbitRAFRef = useRef<number | null>(null);
  const lastFrameTimeRef = useRef<number>(0);

  // HUD direct DOM refs (eliminates 60fps React re-renders on mousemove)
  const cursorHudRef = useRef<HTMLSpanElement>(null);
  const altHudRef = useRef<HTMLSpanElement>(null);

  // Status & UI State
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [cesiumReady, setCesiumReady] = useState(false);
  const shouldFlyToScenarioRef = useRef<boolean>(false);
  const driftGeomRef = useRef<DriftGeometry | null>(null);

  // Active scenario: Do NOT default/pre-set to an oil spill if none selected
  const [activeScenarioKey, setActiveScenarioKey] = useState<string>(
    currentScenario
      ? Object.keys(scenarios).find((k) => scenarios[k].id === currentScenario.id) || ''
      : ''
  );
  const [basemap, setBasemap] = useState<BasemapId>('satellite');

  // Layer Toggles
  const [showEez, setShowEez] = useState(true);
  const [showDrift, setShowDrift] = useState(true);
  const [showVessels, setShowVessels] = useState(true);
  const [showLiveAis, setShowLiveAis] = useState(true);
  const [showSarSwath, setShowSarSwath] = useState(true);
  const [showSarDrape, setShowSarDrape] = useState(true);
  const [showSlick, setShowSlick] = useState(true);
  const [isOrbiting, setIsOrbiting] = useState(false);

  // Live AISStream Fleet State (real-time websocket feed)
  const [liveAisCount, setLiveAisCount] = useState<number>(0);
  const [isApiKeyModalOpen, setIsApiKeyModalOpen] = useState<boolean>(false);

  // Selected Target for Tactical C2 Dossier
  const [selectedVessel, setSelectedVessel] = useState<CandidateVesselItem | null>(null);

  // Active scenario object: Only resolved if a scenario is actively selected
  const activeScenario = activeScenarioKey && scenarios[activeScenarioKey]
    ? scenarios[activeScenarioKey]
    : currentScenario || null;
  const incidentKeyNormalized = activeScenarioKey
    ? (activeScenarioKey.includes('2026') ? activeScenarioKey : activeScenario?.id || activeScenarioKey)
    : 'INC-2026-001';

  // ── Apply Basemap Imagery cleanly without flash ────────────────────────────
  const applyBasemap = useCallback(async (type: BasemapId) => {
    if (!viewerRef.current) return;
    const viewer = viewerRef.current;
    const C = Cesium;

    let provider: Cesium.ImageryProvider;
    try {
      if (type === 'ion') {
        // Cesium Ion / Bing Maps Aerial with Labels
        provider = await C.IonImageryProvider.fromAssetId(2);
      } else if (type === 'dark') {
        provider = new C.UrlTemplateImageryProvider({
          url: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
          subdomains: ['a', 'b', 'c', 'd'],
          maximumLevel: 18,
        });
      } else if (type === 'marine') {
        provider = new C.UrlTemplateImageryProvider({
          url: 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png',
          subdomains: ['a', 'b', 'c', 'd'],
          maximumLevel: 18,
        });
      } else {
        // Default Esri World Imagery (High Resolution Satellite)
        // Capped at level 17 to prevent missing-tile placeholders in remote coastal regions
        provider = new C.UrlTemplateImageryProvider({
          url: 'https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
          maximumLevel: 17,
        });
      }

      const newLayer = new C.ImageryLayer(provider);
      viewer.imageryLayers.add(newLayer, 0);

      if (baseLayerRef.current) {
        viewer.imageryLayers.remove(baseLayerRef.current, true);
      }
      baseLayerRef.current = newLayer;
      currentBasemapRef.current = type;
    } catch (e) {
      console.warn(`[GlobeView] Basemap ${type} fallback notice:`, e);
      // Fallback to Esri satellite on error
      if (type !== 'satellite') {
        applyBasemap('satellite');
      }
    }
  }, []);

  // ── 1. Initialize Cesium Viewer (runs ONCE on mount) ───────────────────────
  useEffect(() => {
    let mounted = true;

    (async () => {
      try {
        if (!mounted || !containerRef.current) return;
        const C = Cesium;

        // Set Cesium Ion Token from environment or saved config
        const ionToken =
          import.meta.env.VITE_CESIUM_ION_TOKEN ||
          'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJub25jZSI6IkI3d3E5c284Y19rbHI2YW0iLCJqdGkiOiIzOWJkMDZmYi1jMDAxLTRlZWEtOWFkZi1hNDJhOTYwOGJlYWIiLCJpZCI6NTM1MzQ3LCJzdWIiOiJrcnNoaGg2IiwiaXNzIjoiaHR0cHM6Ly9hcGkuY2VzaXVtLmNvbSIsImF1ZCI6IkdvZGV5ZSIsImlhdCI6MTc5Mjk5NDMyNH0.40jNwEHo63NVHA5jMBIsrd3Z57fskDJG3tFVxNGR5uY';
        if (ionToken) {
          C.Ion.defaultAccessToken = ionToken;
        }

        // Initialize viewer with Google Earth-grade high-fidelity settings
        const viewer = new C.Viewer(containerRef.current, {
          baseLayer: false,
          baseLayerPicker: false,
          geocoder: false,
          homeButton: false,
          sceneModePicker: false,
          navigationHelpButton: false,
          animation: false,
          timeline: false,
          fullscreenButton: false,
          infoBox: false,
          selectionIndicator: false,
          shadows: false,
          requestRenderMode: false,
          msaaSamples: 4,
          contextOptions: {
            webgl: {
              preserveDrawingBuffer: true,
              powerPreference: 'high-performance',
              antialias: true,
            },
          },
        });

        // Atmosphere & Photorealistic Visual Settings
        viewer.scene.globe.show = true;
        viewer.scene.globe.enableLighting = false; // Daylight satellite illumination
        viewer.scene.globe.showGroundAtmosphere = true;
        viewer.scene.globe.baseColor = C.Color.fromCssColorString('#0b1528');
        viewer.scene.globe.maximumScreenSpaceError = 1.5;

        // Cesium World Terrain with water reflections
        try {
          if (ionToken) {
            viewer.terrainProvider = await C.createWorldTerrainAsync({
              requestWaterMask: true,
              requestVertexNormals: true,
            });
          }
        } catch (terrErr) {
          console.warn('[GlobeView] Cesium World Terrain fallback notice:', terrErr);
        }

        // Atmosphere styling
        if (viewer.scene.skyAtmosphere) {
          viewer.scene.skyAtmosphere.show = true;
          viewer.scene.skyAtmosphere.atmosphereLightIntensity = 10.0;
          viewer.scene.skyAtmosphere.saturationShift = 0.0;
          viewer.scene.skyAtmosphere.brightnessShift = 0.0;
        }

        // Realistic atmospheric haze / fog (Google Earth depth effect)
        viewer.scene.fog.enabled = true;
        viewer.scene.fog.density = 0.0001;
        viewer.scene.fog.screenSpaceErrorFactor = 2.0;

        // Google Earth-style smooth camera inertia controls
        const ssc = viewer.scene.screenSpaceCameraController;
        ssc.enableRotate = true;
        ssc.enableTranslate = true;
        ssc.enableZoom = true;
        ssc.enableTilt = true;
        ssc.enableLook = false;
        ssc.inertiaSpin = 0.85;
        ssc.inertiaTranslate = 0.85;
        ssc.inertiaZoom = 0.8;
        ssc.minimumZoomDistance = 1500;
        ssc.maximumZoomDistance = 25_000_000;

        // Hide default Cesium credit watermark
        const credit = viewer.cesiumWidget.creditContainer as HTMLElement;
        if (credit) credit.style.display = 'none';

        // Add default Satellite basemap synchronously
        const satProvider = new C.UrlTemplateImageryProvider({
          url: 'https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
          maximumLevel: 17,
        });
        const satLayer = new C.ImageryLayer(satProvider);
        viewer.imageryLayers.add(satLayer, 0);
        baseLayerRef.current = satLayer;
        currentBasemapRef.current = 'satellite';

        // Load India Exclusive Economic Zone (EEZ) GeoJSON
        try {
          const eezDS = await C.GeoJsonDataSource.load('/india-eez.geojson', {
            stroke: C.Color.fromCssColorString('#06b6d4').withAlpha(0.9),
            strokeWidth: 2,
            fill: C.Color.fromCssColorString('#06b6d4').withAlpha(0.04),
          });
          viewer.dataSources.add(eezDS);
          eezDSRef.current = eezDS;
        } catch (eezErr) {
          console.warn('[GlobeView] EEZ GeoJSON load notice:', eezErr);
        }

        // Dedicated CustomDataSources for instantaneous flicker-free toggling
        const slickDS = new C.CustomDataSource('slick');
        const driftDS = new C.CustomDataSource('drift');
        const vesselDS = new C.CustomDataSource('vessels');
        const liveVesselDS = new C.CustomDataSource('liveVessels');
        const swathDS = new C.CustomDataSource('swath');
        const sarDS = new C.CustomDataSource('sar');
        const incidentDS = new C.CustomDataSource('incidents');

        await viewer.dataSources.add(slickDS);
        await viewer.dataSources.add(driftDS);
        await viewer.dataSources.add(vesselDS);
        await viewer.dataSources.add(liveVesselDS);
        await viewer.dataSources.add(swathDS);
        await viewer.dataSources.add(sarDS);
        await viewer.dataSources.add(incidentDS);

        layerSourcesRef.current = { slickDS, driftDS, vesselDS, liveVesselDS, swathDS, sarDS, incidentDS };

        // Mouse Move Listener — Uses pickEllipsoid (pure math, no worker crashes, no React re-renders)
        const handler = new C.ScreenSpaceEventHandler(viewer.scene.canvas);
        handler.setInputAction((movement: { endPosition: Cesium.Cartesian2 }) => {
          if (!mounted) return;
          const cartesian = viewer.camera.pickEllipsoid(
            movement.endPosition,
            viewer.scene.globe.ellipsoid
          );
          if (cartesian && cursorHudRef.current) {
            const carto = C.Cartographic.fromCartesian(cartesian);
            const latDeg = C.Math.toDegrees(carto.latitude).toFixed(3);
            const lngDeg = C.Math.toDegrees(carto.longitude).toFixed(3);
            cursorHudRef.current.innerText = `${latDeg}°N, ${lngDeg}°E`;
          }
          if (altHudRef.current) {
            const altKm = Math.round(viewer.camera.positionCartographic.height / 1000);
            altHudRef.current.innerText = `${altKm} km`;
          }
        }, C.ScreenSpaceEventType.MOUSE_MOVE);

        // Click Listener for Vessel & Feature Picking
        handler.setInputAction((click: { position: Cesium.Cartesian2 }) => {
          if (!mounted) return;
          const picked = viewer.scene.pick(click.position);
          if (C.defined(picked) && picked.id && picked.id.properties) {
            const props = picked.id.properties;
            if (props.hasProperty('vesselData')) {
              setSelectedVessel(props.getValue(viewer.clock.currentTime).vesselData);
              return;
            }
            if (props.hasProperty('incidentKey')) {
              const incKey = props.getValue(viewer.clock.currentTime).incidentKey;
              if (incKey) {
                shouldFlyToScenarioRef.current = true;
                setActiveScenarioKey(incKey);
                return;
              }
            }
          }
          setSelectedVessel(null);
        }, C.ScreenSpaceEventType.LEFT_CLICK);

        viewerRef.current = viewer;

        // Set initial camera view to Regional Indian EEZ Maritime Overview (National Macro Perspective)
        // NOT pre-origin focused on an individual oil spill
        viewer.camera.setView({
          destination: C.Cartesian3.fromDegrees(78.5, 12.0, 3_800_000),
          orientation: {
            heading: C.Math.toRadians(0),
            pitch: C.Math.toRadians(-75),
            roll: 0,
          },
        });

        setCesiumReady(true);
        setStatus('ready');
      } catch (err) {
        console.error('[GlobeView] Initialization error:', err);
        if (mounted) setStatus('error');
      }
    })();

    return () => {
      mounted = false;
      if (orbitRAFRef.current !== null) {
        cancelAnimationFrame(orbitRAFRef.current);
        orbitRAFRef.current = null;
      }
      if (viewerRef.current && !viewerRef.current.isDestroyed()) {
        viewerRef.current.destroy();
        viewerRef.current = null;
      }
    };
  }, []);

  // ── 2. Handle Basemap Switching ─────────────────────────────────────────────
  useEffect(() => {
    if (!cesiumReady || !viewerRef.current) return;
    if (currentBasemapRef.current === basemap) return;
    applyBasemap(basemap);
  }, [basemap, cesiumReady, applyBasemap]);

  // ── 3. Toggle Visibility Effects (0ms, zero entity destruction, zero flicker) ──
  useEffect(() => {
    if (eezDSRef.current) eezDSRef.current.show = showEez;
  }, [showEez]);

  useEffect(() => {
    if (layerSourcesRef.current?.slickDS) layerSourcesRef.current.slickDS.show = showSlick;
  }, [showSlick]);

  useEffect(() => {
    if (layerSourcesRef.current?.driftDS) layerSourcesRef.current.driftDS.show = showDrift;
  }, [showDrift]);

  useEffect(() => {
    if (layerSourcesRef.current?.vesselDS) layerSourcesRef.current.vesselDS.show = showVessels;
  }, [showVessels]);

  useEffect(() => {
    if (layerSourcesRef.current?.liveVesselDS) layerSourcesRef.current.liveVesselDS.show = showLiveAis;
  }, [showLiveAis]);

  useEffect(() => {
    if (layerSourcesRef.current?.swathDS) layerSourcesRef.current.swathDS.show = showSarSwath;
  }, [showSarSwath]);

  useEffect(() => {
    if (layerSourcesRef.current?.sarDS) layerSourcesRef.current.sarDS.show = showSarDrape;
  }, [showSarDrape]);

  // ── 4. Rebuild All Tactical Entities (when activeScenarioKey changes) ──
  useEffect(() => {
    if (!cesiumReady || !viewerRef.current || !layerSourcesRef.current) return;
    const viewer = viewerRef.current;
    const C = Cesium;
    const { slickDS, driftDS, vesselDS, swathDS, sarDS, incidentDS } = layerSourcesRef.current;

    slickDS.entities.removeAll();
    driftDS.entities.removeAll();
    vesselDS.entities.removeAll();
    swathDS.entities.removeAll();
    sarDS.entities.removeAll();
    incidentDS.entities.removeAll();

    if (!activeScenario) {
      driftGeomRef.current = null;
      // When no scenario is pre-selected, populate all incident pins with glassmorphic cards
      incidents.forEach((inc) => {
        if (!inc.lat || !inc.lng) return;
        const incCard = getGlassmorphicIncidentCard(
          inc.title,
          `${inc.lat.toFixed(2)}°N, ${inc.lng.toFixed(2)}°E · Marine Incident`
        );

        incidentDS.entities.add({
          name: inc.title,
          position: C.Cartesian3.fromDegrees(inc.lng, inc.lat, 35),
          point: {
            pixelSize: 10,
            color: C.Color.fromCssColorString('#f97316'),
            outlineColor: C.Color.WHITE,
            outlineWidth: 2,
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
          },
          properties: new C.PropertyBag({
            incidentKey: inc.id,
          }),
        });

        incidentDS.entities.add({
          name: `[INCIDENT CARD] ${inc.title}`,
          position: C.Cartesian3.fromDegrees(inc.lng, inc.lat, 35),
          billboard: {
            image: incCard.dataUrl,
            width: incCard.width,
            height: incCard.height,
            verticalOrigin: C.VerticalOrigin.BOTTOM,
            pixelOffset: new C.Cartesian2(0, -16),
            eyeOffset: C.Cartesian3.ZERO,
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
            distanceDisplayCondition: new C.DistanceDisplayCondition(0, 15_000_000),
          },
          properties: new C.PropertyBag({
            incidentKey: inc.id,
          }),
        });
      });
      return;
    }

    const { lat, lng } = activeScenario;
    const scenarioKey = activeScenarioKey;
    const normalizedKey = scenarioKey.replace('2026-', '');

    // ── A. Hydrodynamic Lagrangian Drift Calculations ─────────────────────────
    let driftGeom: DriftGeometry | null = null;
    try {
      driftGeom = computeBackwardDriftGeometry(scenarioKey, [lat, lng]);
      driftGeomRef.current = driftGeom;
    } catch (e) {
      console.warn('[GlobeView] Drift calculation notice:', e);
      driftGeomRef.current = null;
    }

    // ── B. SENTINEL-1 SAR IMAGERY DRAPE ───────────────────────────────────────
    try {
      const aoi = createAoiForScenario(lat, lng);
      const coords = aoi.coordinates[0] as Array<[number, number]>;
      const minLng = Math.min(...coords.map((c) => c[0]));
      const maxLng = Math.max(...coords.map((c) => c[0]));
      const minLat = Math.min(...coords.map((c) => c[1]));
      const maxLat = Math.max(...coords.map((c) => c[1]));

      sarDS.entities.add({
        name: `Sentinel-1 C-SAR IW Radar Image (${scenarioKey})`,
        rectangle: {
          coordinates: C.Rectangle.fromDegrees(minLng, minLat, maxLng, maxLat),
          material: new C.ImageMaterialProperty({
            image: `/imagery/sar_${normalizedKey}.webp`,
            transparent: true,
            color: C.Color.WHITE.withAlpha(0.85),
          }),
          height: 2,
        },
      });
    } catch (e) {
      console.warn('[GlobeView] SAR drape notice:', e);
    }

    // ── C. OIL SLICK POLYGON & CENTROID ───────────────────────────────────────
    const coreRadius = 0.055;
    const corePositions: number[] = [];
    const sides = 14;
    for (let i = 0; i <= sides; i++) {
      const rad = (i * 2 * Math.PI) / sides;
      const jitter = i % 3 === 0 ? 0.78 : i % 2 === 0 ? 1.15 : 0.95;
      corePositions.push(lng + Math.cos(rad) * coreRadius * jitter * 1.3);
      corePositions.push(lat + Math.sin(rad) * coreRadius * jitter * 0.85);
    }

    slickDS.entities.add({
      name: 'Oil Slick Core (Weathered Emulsion)',
      polygon: {
        hierarchy: new C.PolygonHierarchy(C.Cartesian3.fromDegreesArray(corePositions)),
        material: C.Color.fromCssColorString('#09090b').withAlpha(0.88),
        outline: true,
        outlineColor: C.Color.fromCssColorString('#ef4444'),
        outlineWidth: 2,
        height: 10,
      },
    });

    const slickCard = getGlassmorphicBadgeCard(
      '🚨',
      activeScenario.title,
      `Area: ${activeScenario.area || '4.8 km²'} · ${activeScenario.oilType}`,
      '#ef4444'
    );

    slickDS.entities.add({
      name: `Slick Centroid — ${activeScenario.title}`,
      position: C.Cartesian3.fromDegrees(lng, lat, 40),
      point: {
        pixelSize: 12,
        color: C.Color.fromCssColorString('#ef4444'),
        outlineColor: C.Color.WHITE,
        outlineWidth: 3,
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });

    slickDS.entities.add({
      name: `[SLICK CARD] ${activeScenario.title}`,
      position: C.Cartesian3.fromDegrees(lng, lat, 40),
      billboard: {
        image: slickCard.dataUrl,
        width: slickCard.width,
        height: slickCard.height,
        verticalOrigin: C.VerticalOrigin.BOTTOM,
        pixelOffset: new C.Cartesian2(0, -20),
        eyeOffset: C.Cartesian3.ZERO,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
        distanceDisplayCondition: new C.DistanceDisplayCondition(0, 15_000_000),
      },
      properties: new C.PropertyBag({
        incidentKey: activeScenarioKey,
      }),
    });

    // ── D. LAGRANGIAN DRIFT & MONTE CARLO PROBABILITY ENVELOPES ──────────────
    if (driftGeom) {
      if (driftGeom.envelope90 && driftGeom.envelope90.length > 2) {
        driftDS.entities.add({
          name: '90% Origin Probability Envelope',
          polygon: {
            hierarchy: new C.PolygonHierarchy(
              C.Cartesian3.fromDegreesArray(
                driftGeom.envelope90.flatMap((c) => [c[1], c[0]])
              )
            ),
            material: C.Color.fromCssColorString('#f59e0b').withAlpha(0.12),
            outline: true,
            outlineColor: C.Color.fromCssColorString('#f59e0b').withAlpha(0.65),
            outlineWidth: 1,
            height: 4,
          },
        });
      }

      if (driftGeom.envelope75 && driftGeom.envelope75.length > 2) {
        driftDS.entities.add({
          name: '75% Origin Probability Envelope',
          polygon: {
            hierarchy: new C.PolygonHierarchy(
              C.Cartesian3.fromDegreesArray(
                driftGeom.envelope75.flatMap((c) => [c[1], c[0]])
              )
            ),
            material: C.Color.fromCssColorString('#f97316').withAlpha(0.18),
            outline: true,
            outlineColor: C.Color.fromCssColorString('#f97316').withAlpha(0.75),
            outlineWidth: 1.5,
            height: 6,
          },
        });
      }

      if (driftGeom.envelope50 && driftGeom.envelope50.length > 2) {
        driftDS.entities.add({
          name: '50% Ground Zero Release Envelope',
          polygon: {
            hierarchy: new C.PolygonHierarchy(
              C.Cartesian3.fromDegreesArray(
                driftGeom.envelope50.flatMap((c) => [c[1], c[0]])
              )
            ),
            material: C.Color.fromCssColorString('#ef4444').withAlpha(0.28),
            outline: true,
            outlineColor: C.Color.fromCssColorString('#ef4444'),
            outlineWidth: 2,
            height: 8,
          },
        });
      }

      if (driftGeom.driftPath && driftGeom.driftPath.length > 1) {
        const pathPositions = driftGeom.driftPath.map((c) =>
          C.Cartesian3.fromDegrees(c[1], c[0], 25)
        );
        driftDS.entities.add({
          name: 'Lagrangian Backward Drift Path',
          polyline: {
            positions: pathPositions,
            width: 3.5,
            material: new C.PolylineDashMaterialProperty({
              color: C.Color.fromCssColorString('#fb923c'),
              dashLength: 16.0,
            }),
          },
        });
      }

      const origin = driftGeom.originCoord;
      const originCard = getGlassmorphicBadgeCard(
        '⚡',
        'DISCHARGE ORIGIN [T - 4.5h]',
        `${origin[0].toFixed(3)}°N, ${origin[1].toFixed(3)}°E · Reverse Drift`,
        '#f59e0b'
      );

      driftDS.entities.add({
        name: 'Ground Zero: Suspected Discharge Origin',
        position: C.Cartesian3.fromDegrees(origin[1], origin[0], 35),
        point: {
          pixelSize: 12,
          color: C.Color.fromCssColorString('#ef4444'),
          outlineColor: C.Color.fromCssColorString('#fef08a'),
          outlineWidth: 3,
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      });

      driftDS.entities.add({
        name: '[ORIGIN CARD] Discharge Origin',
        position: C.Cartesian3.fromDegrees(origin[1], origin[0], 35),
        billboard: {
          image: originCard.dataUrl,
          width: originCard.width,
          height: originCard.height,
          verticalOrigin: C.VerticalOrigin.BOTTOM,
          pixelOffset: new C.Cartesian2(0, -20),
          eyeOffset: C.Cartesian3.ZERO,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      });

      if (driftGeom.forwardDriftPath && driftGeom.forwardDriftPath.length > 1) {
        const fwdPositions = driftGeom.forwardDriftPath.map((c) =>
          C.Cartesian3.fromDegrees(c[1], c[0], 20)
        );
        driftDS.entities.add({
          name: 'Forward Drift Trajectory Forecast (+24h)',
          polyline: {
            positions: fwdPositions,
            width: 3,
            material: new C.PolylineDashMaterialProperty({
              color: C.Color.fromCssColorString('#06b6d4'),
              dashLength: 12.0,
            }),
          },
        });

        const fwdEnd = driftGeom.forwardDriftPath[driftGeom.forwardDriftPath.length - 1];
        const forecastCard = getGlassmorphicBadgeCard(
          '⏱',
          'T+24h DRIFT FORECAST',
          `${fwdEnd[0].toFixed(3)}°N, ${fwdEnd[1].toFixed(3)}°E · Hydro Forecast`,
          '#06b6d4'
        );

        driftDS.entities.add({
          name: 'Forecast Landfall Vector (+24h)',
          position: C.Cartesian3.fromDegrees(fwdEnd[1], fwdEnd[0], 30),
          point: {
            pixelSize: 9,
            color: C.Color.fromCssColorString('#06b6d4'),
            outlineColor: C.Color.WHITE,
            outlineWidth: 2,
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
          },
        });

        driftDS.entities.add({
          name: '[FORECAST CARD] T+24h Landfall',
          position: C.Cartesian3.fromDegrees(fwdEnd[1], fwdEnd[0], 30),
          billboard: {
            image: forecastCard.dataUrl,
            width: forecastCard.width,
            height: forecastCard.height,
            verticalOrigin: C.VerticalOrigin.BOTTOM,
            pixelOffset: new C.Cartesian2(0, -18),
            eyeOffset: C.Cartesian3.ZERO,
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
          },
        });
      }
    }

    // ── E. AIS CANDIDATE SUSPECT FLEET ─────────────────────────────────────────
    const candidates =
      DEFAULT_SCENARIO_VESSELS[incidentKeyNormalized] ||
      DEFAULT_SCENARIO_VESSELS['INC-2026-001'] ||
      [];

    candidates.forEach((vessel) => {

      const card = getGlassmorphicVesselCard(
        vessel.name,
        `${(vessel.attribution_score * 100).toFixed(0)}% Attrib · ${vessel.flag}`,
        vessel.sog,
        vessel.cog,
        vessel.risk,
        vessel.risk === 'CRITICAL'
      );

      vesselDS.entities.add({
        name: `${vessel.name} [MMSI: ${vessel.mmsi}]`,
        position: C.Cartesian3.fromDegrees(vessel.lng, vessel.lat, 35),
        billboard: {
          image: createVesselSvgPin(vessel.risk, vessel.cog),
          width: 28,
          height: 28,
          verticalOrigin: C.VerticalOrigin.CENTER,
          eyeOffset: C.Cartesian3.ZERO,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
        properties: new C.PropertyBag({
          vesselData: vessel,
        }),
      });

      // Floating Glassmorphic Suspect Callout Card
      vesselDS.entities.add({
        name: `[SUSPECT CARD] ${vessel.name}`,
        position: C.Cartesian3.fromDegrees(vessel.lng, vessel.lat, 35),
        billboard: {
          image: card.dataUrl,
          width: card.width,
          height: card.height,
          verticalOrigin: C.VerticalOrigin.BOTTOM,
          pixelOffset: new C.Cartesian2(0, -18),
          eyeOffset: C.Cartesian3.ZERO,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
          distanceDisplayCondition: new C.DistanceDisplayCondition(0, 2_200_000),
        },
        properties: new C.PropertyBag({
          vesselData: vessel,
        }),
      });

      // Transponder-Silent / Dark Corridor (if AIS gap exists)
      if (vessel.ais_gap_hours && vessel.ais_gap_hours > 0.5 && driftGeom?.originCoord) {
        vesselDS.entities.add({
          name: `${vessel.name} AIS Silent Corridor (${vessel.ais_gap_hours.toFixed(1)}h Gap)`,
          polyline: {
            positions: [
              C.Cartesian3.fromDegrees(vessel.lng, vessel.lat, 20),
              C.Cartesian3.fromDegrees(driftGeom.originCoord[1], driftGeom.originCoord[0], 20),
            ],
            width: 2.5,
            material: new C.PolylineDashMaterialProperty({
              color: C.Color.fromCssColorString('#dc2626'),
              dashLength: 10.0,
            }),
          },
        });
      }
    });

    // ── F. SENTINEL-1 ORBITAL SWATH ──────────────────────────────────────────
    const { track, swathLeft, swathRight } = generateSentinelSwath(lat, lng);

    const trackPositions = track.map(([tLat, tLng]) =>
      C.Cartesian3.fromDegrees(tLng, tLat, 693000)
    );
    swathDS.entities.add({
      name: 'Sentinel-1 Orbital Path (693 km Sun-Sync)',
      polyline: {
        positions: trackPositions,
        width: 2,
        material: C.Color.fromCssColorString('#8b5cf6').withAlpha(0.65),
      },
    });

    const swathPolygonCoords: Cesium.Cartesian3[] = [];
    swathLeft.forEach(([sLat, sLng]) =>
      swathPolygonCoords.push(C.Cartesian3.fromDegrees(sLng, sLat, 0))
    );
    [...swathRight].reverse().forEach(([sLat, sLng]) =>
      swathPolygonCoords.push(C.Cartesian3.fromDegrees(sLng, sLat, 0))
    );

    swathDS.entities.add({
      name: 'Sentinel-1 C-SAR IW Mode 250 km Radar Swath',
      polygon: {
        hierarchy: new C.PolygonHierarchy(swathPolygonCoords),
        material: C.Color.fromCssColorString('#8b5cf6').withAlpha(0.06),
        outline: true,
        outlineColor: C.Color.fromCssColorString('#8b5cf6').withAlpha(0.4),
        outlineWidth: 1.5,
        height: 1,
      },
    });

    const midPoint = track[Math.floor(track.length / 2)];
    swathDS.entities.add({
      name: 'Sentinel-1C Spacecraft Platform',
      position: C.Cartesian3.fromDegrees(midPoint[1], midPoint[0], 693000),
      point: {
        pixelSize: 12,
        color: C.Color.fromCssColorString('#c4b5fd'),
        outlineColor: C.Color.fromCssColorString('#8b5cf6'),
        outlineWidth: 3,
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
      label: {
        text: '🛰 SENTINEL-1C SAR (IW MODE)\n693 km Polar Orbit · 250 km Swath',
        font: 'bold 11px monospace',
        showBackground: true,
        backgroundColor: C.Color.fromCssColorString('rgba(24, 16, 50, 0.95)'),
        backgroundPadding: new C.Cartesian2(8, 5),
        fillColor: C.Color.fromCssColorString('#e9d5ff'),
        outlineColor: C.Color.fromCssColorString('#7c3aed'),
        outlineWidth: 2,
        style: C.LabelStyle.FILL_AND_OUTLINE,
        verticalOrigin: C.VerticalOrigin.BOTTOM,
        pixelOffset: new C.Cartesian2(0, -22),
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });

    // ── G. Other National Incident Overview Markers ───────────────────────────
    incidents.forEach((inc) => {
      if (!inc.lat || !inc.lng || (inc.lat === lat && inc.lng === lng)) return;
      const incCard = getGlassmorphicIncidentCard(
        inc.title,
        `${inc.lat.toFixed(2)}°N, ${inc.lng.toFixed(2)}°E · Marine Incident`
      );

      incidentDS.entities.add({
        name: inc.title,
        position: C.Cartesian3.fromDegrees(inc.lng, inc.lat, 35),
        properties: new C.PropertyBag({
          incidentKey: inc.id,
        }),
        point: {
          pixelSize: 8,
          color: C.Color.fromCssColorString('#f97316'),
          outlineColor: C.Color.WHITE,
          outlineWidth: 2,
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      });

      incidentDS.entities.add({
        name: `[INCIDENT CARD] ${inc.title}`,
        position: C.Cartesian3.fromDegrees(inc.lng, inc.lat, 35),
        billboard: {
          image: incCard.dataUrl,
          width: incCard.width,
          height: incCard.height,
          verticalOrigin: C.VerticalOrigin.BOTTOM,
          pixelOffset: new C.Cartesian2(0, -14),
          eyeOffset: C.Cartesian3.ZERO,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
          distanceDisplayCondition: new C.DistanceDisplayCondition(0, 15_000_000),
        },
        properties: new C.PropertyBag({
          incidentKey: inc.id,
        }),
      });
    });

    // ── H. Google Earth Oblique Camera Fly-To Incident ────────────────────────
    // Only fly if user explicitly selected this scenario (DO NOT pre-origin on initial load)
    if (shouldFlyToScenarioRef.current) {
      viewer.camera.flyTo({
        destination: C.Cartesian3.fromDegrees(lng, lat - 0.14, 20000),
        orientation: {
          heading: C.Math.toRadians(0),
          pitch: C.Math.toRadians(-38),
          roll: 0,
        },
        duration: 2.0,
      });
      shouldFlyToScenarioRef.current = false;
    }
  }, [cesiumReady, activeScenarioKey, activeScenario, incidents]);

  // ── 5. Live AISStream Fleet Polling (God's Eye View live WebSocket feed) ──
  useEffect(() => {
    if (!cesiumReady || !layerSourcesRef.current) return;
    let mounted = true;

    const pollLiveAis = async () => {
      try {
        const res = await fetch('/api/v1/ais/stream-status');
        if (!res.ok) return;
        const data = await res.json();
        if (!mounted || !layerSourcesRef.current) return;

        const vessels = data.vessels || [];
        setLiveAisCount(data.total_live_vessels_tracked || vessels.length);

        const { liveVesselDS } = layerSourcesRef.current;
        liveVesselDS.entities.removeAll();
        const C = Cesium;

        vessels.forEach((v: any) => {
          if (!v.lat || !v.lng) return;
          const isTanker = v.is_tanker || (v.type && /tanker|oil|crude|gas|lng|chemical/i.test(v.type));
          const riskLevel = isTanker ? 'CRITICAL' : v.type && /cargo|container/i.test(v.type) ? 'MEDIUM' : 'LOW';

          // Compute CPA to the reconstructed discharge origin if active
          const originCoord = driftGeomRef.current?.originCoord;
          const liveCpaNm = originCoord
            ? computeDistanceKm(v.lat, v.lng, originCoord[0], originCoord[1]) / 1.852
            : (v.cpa_nm || 0);

          // Convert to CandidateVesselItem structure for the C2 dossier
          const vesselItem: CandidateVesselItem = {
            mmsi: v.mmsi,
            imo: v.imo || 'N/A',
            name: v.name || `MMSI: ${v.mmsi}`,
            type: v.type || 'Commercial Vessel',
            flag: v.flag || 'International',
            sog: v.sog || 0.0,
            cog: v.cog || 0.0,
            lat: v.lat,
            lng: v.lng,
            risk: isTanker ? 'HIGH' : 'LOW',
            attribution_score: isTanker ? 0.75 : 0.20,
            cpa_nm: liveCpaNm,
            ais_gap_hours: 0,
          };

          // 1. Vessel Directional Navigation Arrow Pin
          liveVesselDS.entities.add({
            name: `[LIVE AIS PIN] ${vesselItem.name} (MMSI: ${v.mmsi})`,
            position: C.Cartesian3.fromDegrees(v.lng, v.lat, 35),
            billboard: {
              image: createVesselSvgPin(riskLevel, v.cog || 0),
              width: 28,
              height: 28,
              verticalOrigin: C.VerticalOrigin.CENTER,
              eyeOffset: C.Cartesian3.ZERO,
              disableDepthTestDistance: Number.POSITIVE_INFINITY,
            },
            properties: new C.PropertyBag({
              vesselData: vesselItem,
            }),
          });

          // 2. High-DPI Glassmorphic Callout Card Frame
          const card = getGlassmorphicVesselCard(
            vesselItem.name,
            v.type || 'Commercial Vessel',
            v.sog || 0.0,
            v.cog || 0.0,
            riskLevel,
            isTanker
          );

          liveVesselDS.entities.add({
            name: `[LIVE AIS CARD] ${vesselItem.name}`,
            position: C.Cartesian3.fromDegrees(v.lng, v.lat, 35),
            billboard: {
              image: card.dataUrl,
              width: card.width,
              height: card.height,
              verticalOrigin: C.VerticalOrigin.BOTTOM,
              pixelOffset: new C.Cartesian2(0, -18),
              eyeOffset: C.Cartesian3.ZERO,
              disableDepthTestDistance: Number.POSITIVE_INFINITY,
              distanceDisplayCondition: new C.DistanceDisplayCondition(0, 2_200_000), // Declutters when zoomed out!
            },
            properties: new C.PropertyBag({
              vesselData: vesselItem,
            }),
          });
        });
      } catch (e) {
        // quiet network poll catch
      }
    };

    pollLiveAis();
    const interval = setInterval(pollLiveAis, 7000);

    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, [cesiumReady]);

  // ── 6. Google Earth 360° Turntable Orbit Animation ────────────────────────
  useEffect(() => {
    if (!cesiumReady || !viewerRef.current || !activeScenario) return;
    const viewer = viewerRef.current;
    const C = Cesium;

    if (orbitRAFRef.current !== null) {
      cancelAnimationFrame(orbitRAFRef.current);
      orbitRAFRef.current = null;
    }

    if (!isOrbiting) return;

    // Turntable rotation: rotate around the surface normal at the incident location or Indian domain center
    const targetLng = activeScenario ? activeScenario.lng : 78.5;
    const targetLat = activeScenario ? activeScenario.lat : 15.0;
    const targetCartesian = C.Cartesian3.fromDegrees(targetLng, targetLat, 0);
    const surfaceNormal = C.Ellipsoid.WGS84.geodeticSurfaceNormal(targetCartesian);
    const DEG_PER_SEC = 5.0; // 5 degrees per second
    lastFrameTimeRef.current = performance.now();

    const tick = (now: number) => {
      const dt = (now - lastFrameTimeRef.current) / 1000;
      lastFrameTimeRef.current = now;

      if (!viewer.isDestroyed()) {
        viewer.camera.rotate(surfaceNormal, C.Math.toRadians(DEG_PER_SEC * dt));
      }
      orbitRAFRef.current = requestAnimationFrame(tick);
    };

    orbitRAFRef.current = requestAnimationFrame(tick);

    return () => {
      if (orbitRAFRef.current !== null) {
        cancelAnimationFrame(orbitRAFRef.current);
        orbitRAFRef.current = null;
      }
    };
  }, [isOrbiting, cesiumReady, activeScenario]);

  // ── Camera Presets (Google Earth Perspectives) ─────────────────────────────
  const flyToPreset = useCallback(
    (preset: 'slick' | 'origin' | 'eez') => {
      if (!viewerRef.current || !cesiumReady) return;
      const viewer = viewerRef.current;
      const C = Cesium;

      if (preset === 'eez') {
        viewer.camera.flyTo({
          destination: C.Cartesian3.fromDegrees(78.5, 12.0, 3_800_000),
          orientation: {
            heading: C.Math.toRadians(0),
            pitch: C.Math.toRadians(-75),
            roll: 0,
          },
          duration: 2.2,
        });
        return;
      }

      const targetScenario = activeScenario || Object.values(scenarios)[0];
      if (!targetScenario) return;
      const { lat, lng } = targetScenario;

      if (preset === 'slick') {
        viewer.camera.flyTo({
          destination: C.Cartesian3.fromDegrees(lng, lat - 0.10, 14000),
          orientation: {
            heading: C.Math.toRadians(0),
            pitch: C.Math.toRadians(-36),
            roll: 0,
          },
          duration: 1.8,
        });
      } else if (preset === 'origin') {
        let originLat = lat + 0.06;
        let originLng = lng - 0.07;
        try {
          const geom = computeBackwardDriftGeometry(activeScenarioKey || Object.keys(scenarios)[0], [lat, lng]);
          originLat = geom.originCoord[0];
          originLng = geom.originCoord[1];
        } catch {}

        viewer.camera.flyTo({
          destination: C.Cartesian3.fromDegrees(originLng, originLat - 0.08, 12000),
          orientation: {
            heading: C.Math.toRadians(350),
            pitch: C.Math.toRadians(-40),
            roll: 0,
          },
          duration: 1.8,
        });
      }
    },
    [cesiumReady, activeScenario, activeScenarioKey, scenarios]
  );

  return (
    <div
      id="tab-globe"
      className="tab-content visible"
      style={{
        position: 'relative',
        width: '100%',
        height: '100%',
        background: '#030712',
        overflow: 'hidden',
      }}
    >
      {/* ── FULLSCREEN 3D CESIUM VIEWPORT ───────────────────────────────────── */}
      <div
        ref={containerRef}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          width: '100%',
          height: '100%',
          background: '#020617',
        }}
      />

      {/* ── FLOATING GLASS COMMAND ISLAND (TOP) ──────────────────────────────── */}
      <div
        style={{
          position: 'absolute',
          top: 14,
          left: 16,
          right: 16,
          zIndex: 30,
          display: 'flex',
          flexDirection: 'column',
          gap: 10,
          padding: '10px 16px',
          background: 'rgba(10, 16, 30, 0.72)',
          backdropFilter: 'blur(24px) saturate(190%)',
          WebkitBackdropFilter: 'blur(24px) saturate(190%)',
          border: '1px solid rgba(255, 255, 255, 0.12)',
          borderRadius: 14,
          boxShadow: '0 16px 40px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.15)',
        }}
      >
        {/* ROW 1: PRIMARY IDENTITY, WORKSPACE ROUTING, INCIDENT DROPDOWN & BASEMAPS */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: 10,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            {/* Brand Glass Badge */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                padding: '5px 12px',
                background: 'rgba(139, 92, 246, 0.18)',
                border: '1px solid rgba(168, 85, 247, 0.35)',
                borderRadius: 8,
                boxShadow: 'inset 0 1px 0 rgba(255, 255, 255, 0.15)',
              }}
            >
              <span
                style={{
                  display: 'inline-block',
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  background: '#a855f7',
                  boxShadow: '0 0 10px #a855f7',
                }}
              />
              <span
                style={{
                  fontWeight: 700,
                  fontSize: 12,
                  letterSpacing: '0.05em',
                  color: '#f1f5f9',
                  textTransform: 'uppercase',
                }}
              >
                3D Tactical Globe
              </span>
              <span
                style={{
                  fontSize: 10,
                  padding: '1px 6px',
                  background: 'rgba(56, 189, 248, 0.2)',
                  color: '#38bdf8',
                  borderRadius: 4,
                  fontWeight: 600,
                }}
              >
                C2 LIVE
              </span>
            </div>

            {onSelectTab && (
              <button
                onClick={() => onSelectTab('dashboard')}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  padding: '5px 12px',
                  background: 'rgba(255, 255, 255, 0.05)',
                  border: '1px solid rgba(255, 255, 255, 0.12)',
                  borderRadius: 8,
                  color: '#cbd5e1',
                  fontSize: 12,
                  fontWeight: 600,
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                }}
                title="Return to 2D Map Dashboard"
              >
                <span>←</span>
                <span>2D Map</span>
              </button>
            )}

            {/* Custom Styled Glass Dropdown */}
            <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
              <select
                value={activeScenarioKey}
                onChange={(e) => {
                  const k = e.target.value;
                  if (k) {
                    shouldFlyToScenarioRef.current = true;
                    setActiveScenarioKey(k);
                    if (viewerRef.current && scenarios[k]) {
                      const sc = scenarios[k];
                      viewerRef.current.camera.flyTo({
                        destination: Cesium.Cartesian3.fromDegrees(sc.lng, sc.lat - 0.12, 16000),
                        orientation: {
                          heading: Cesium.Math.toRadians(0),
                          pitch: Cesium.Math.toRadians(-38),
                          roll: 0,
                        },
                        duration: 2.0,
                      });
                    }
                  } else {
                    shouldFlyToScenarioRef.current = false;
                    setActiveScenarioKey('');
                    if (viewerRef.current) {
                      viewerRef.current.camera.flyTo({
                        destination: Cesium.Cartesian3.fromDegrees(78.5, 12.0, 3_800_000),
                        orientation: {
                          heading: Cesium.Math.toRadians(0),
                          pitch: Cesium.Math.toRadians(-75),
                          roll: 0,
                        },
                        duration: 2.0,
                      });
                    }
                  }
                  setSelectedVessel(null);
                }}
                style={{
                  appearance: 'none',
                  WebkitAppearance: 'none',
                  background: 'rgba(15, 23, 42, 0.75)',
                  color: '#f8fafc',
                  border: '1px solid rgba(139, 92, 246, 0.35)',
                  borderRadius: 8,
                  padding: '6px 32px 6px 12px',
                  fontSize: 12,
                  fontWeight: 600,
                  cursor: 'pointer',
                  outline: 'none',
                  boxShadow: '0 4px 12px rgba(0, 0, 0, 0.25)',
                }}
              >
                <option value="">🌐 Full Maritime Surveillance (Broad Indian EEZ · All Vessels)</option>
                {Object.keys(scenarios).map((k) => (
                  <option key={k} value={k}>
                    🚨 {scenarios[k].title} ({scenarios[k].oilType})
                  </option>
                ))}
              </select>
              <span
                style={{
                  position: 'absolute',
                  right: 12,
                  pointerEvents: 'none',
                  color: '#94a3b8',
                  fontSize: 10,
                }}
              >
                ▼
              </span>
            </div>
          </div>

          {/* Frosted Basemap Segmented Pill Control */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              background: 'rgba(15, 23, 42, 0.65)',
              padding: 3,
              borderRadius: 9,
              border: '1px solid rgba(255, 255, 255, 0.10)',
              gap: 2,
            }}
          >
            {[
              { id: 'satellite', label: '🛰 Satellite', title: 'Photorealistic Esri World Imagery' },
              { id: 'ion', label: '🌐 Cesium Ion', title: 'Cesium Ion Photorealistic Aerial Basemap' },
              { id: 'dark', label: '🕶 Dark C2', title: 'CartoDB Dark Matter Tactical Night Basemap' },
              { id: 'marine', label: '🗺 Marine', title: 'OpenSeaMap Nautical Navigation Chart' },
            ].map((bm) => {
              const isActive = basemap === bm.id;
              return (
                <button
                  key={bm.id}
                  onClick={() => setBasemap(bm.id as BasemapId)}
                  style={{
                    background: isActive ? 'rgba(139, 92, 246, 0.45)' : 'transparent',
                    color: isActive ? '#ffffff' : '#94a3b8',
                    border: isActive ? '1px solid rgba(168, 85, 247, 0.55)' : '1px solid transparent',
                    borderRadius: 7,
                    padding: '4px 10px',
                    fontSize: 11,
                    fontWeight: isActive ? 600 : 500,
                    cursor: 'pointer',
                    transition: 'all 0.15s ease',
                    boxShadow: isActive ? '0 2px 8px rgba(139, 92, 246, 0.35)' : 'none',
                  }}
                  title={bm.title}
                >
                  {bm.label}
                </button>
              );
            })}
          </div>
        </div>

        {/* ROW 2: TACTICAL LAYER FILTERS & QUICK ACTION PILLS */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: 10,
          }}
        >
          {/* Layer Chips */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
            <span
              style={{
                fontSize: 10,
                fontWeight: 700,
                color: '#64748b',
                letterSpacing: '0.06em',
                textTransform: 'uppercase',
                marginRight: 2,
              }}
            >
              LAYERS:
            </span>
            {[
              { label: 'India EEZ', active: showEez, toggle: () => setShowEez((v) => !v), color: '#06b6d4' },
              { label: 'SAR Drape', active: showSarDrape, toggle: () => setShowSarDrape((v) => !v), color: '#a855f7' },
              { label: 'Drift', active: showDrift, toggle: () => setShowDrift((v) => !v), color: '#f97316' },
              { label: 'Suspects', active: showVessels, toggle: () => setShowVessels((v) => !v), color: '#ef4444' },
              {
                label: `Live AIS (${liveAisCount})`,
                active: showLiveAis,
                toggle: () => setShowLiveAis((v) => !v),
                color: '#22c55e',
              },
              { label: 'Swath', active: showSarSwath, toggle: () => setShowSarSwath((v) => !v), color: '#8b5cf6' },
              { label: 'Slick', active: showSlick, toggle: () => setShowSlick((v) => !v), color: '#ef4444' },
            ].map(({ label, active, toggle, color }) => (
              <button
                key={label}
                onClick={toggle}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 5,
                  background: active ? `${color}22` : 'rgba(255, 255, 255, 0.03)',
                  color: active ? '#f1f5f9' : '#64748b',
                  border: `1px solid ${active ? `${color}66` : 'rgba(255, 255, 255, 0.08)'}`,
                  borderRadius: 6,
                  padding: '4px 9px',
                  fontSize: 11,
                  fontWeight: 600,
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                  boxShadow: active ? `0 2px 8px ${color}22` : 'none',
                }}
              >
                <span
                  style={{
                    width: 6,
                    height: 6,
                    borderRadius: '50%',
                    background: active ? color : '#475569',
                    boxShadow: active ? `0 0 6px ${color}` : 'none',
                  }}
                />
                <span>{label}</span>
              </button>
            ))}
          </div>

          {/* Perspective & Actions */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{ width: 1, height: 18, background: 'rgba(255, 255, 255, 0.12)', margin: '0 4px' }} />

            <button
              onClick={() => flyToPreset('eez')}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 5,
                background: 'rgba(56, 189, 248, 0.15)',
                color: '#38bdf8',
                border: '1px solid rgba(56, 189, 248, 0.45)',
                borderRadius: 6,
                padding: '4px 10px',
                fontSize: 11,
                fontWeight: 600,
                cursor: 'pointer',
              }}
              title="Overview: Indian Exclusive Economic Zone & Maritime Surveillance"
            >
              <span>🇮🇳</span>
              <span>EEZ Overview</span>
            </button>

            <button
              onClick={() => flyToPreset('slick')}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 5,
                background: activeScenario ? 'rgba(239, 68, 68, 0.15)' : 'rgba(255, 255, 255, 0.04)',
                color: activeScenario ? '#f87171' : '#64748b',
                border: `1px solid ${activeScenario ? 'rgba(239, 68, 68, 0.45)' : 'rgba(255, 255, 255, 0.08)'}`,
                borderRadius: 6,
                padding: '4px 10px',
                fontSize: 11,
                fontWeight: 600,
                cursor: activeScenario ? 'pointer' : 'not-allowed',
              }}
              title={activeScenario ? 'Focus camera directly on oil slick polygon' : 'Select an incident to view slick'}
            >
              <span>🎯</span>
              <span>Slick</span>
            </button>

            <button
              onClick={() => flyToPreset('origin')}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 5,
                background: activeScenario ? 'rgba(234, 179, 8, 0.15)' : 'rgba(255, 255, 255, 0.04)',
                color: activeScenario ? '#fde047' : '#64748b',
                border: `1px solid ${activeScenario ? 'rgba(234, 179, 8, 0.45)' : 'rgba(255, 255, 255, 0.08)'}`,
                borderRadius: 6,
                padding: '4px 10px',
                fontSize: 11,
                fontWeight: 600,
                cursor: activeScenario ? 'pointer' : 'not-allowed',
              }}
              title="Focus camera on suspected discharge release origin"
            >
              <span>⚡</span>
              <span>Origin</span>
            </button>

            <button
              onClick={() => setIsOrbiting((v) => !v)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 5,
                background: isOrbiting ? 'rgba(139, 92, 246, 0.35)' : 'rgba(255, 255, 255, 0.04)',
                color: isOrbiting ? '#c084fc' : '#cbd5e1',
                border: `1px solid ${isOrbiting ? 'rgba(168, 85, 247, 0.6)' : 'rgba(255, 255, 255, 0.1)'}`,
                borderRadius: 6,
                padding: '4px 10px',
                fontSize: 11,
                fontWeight: 600,
                cursor: 'pointer',
              }}
              title="Toggle continuous 360° turntable orbit"
            >
              <span>{isOrbiting ? '⏹' : '🔄'}</span>
              <span>Orbit</span>
            </button>

            <button
              onClick={() => setIsApiKeyModalOpen(true)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                background: 'rgba(245, 158, 11, 0.15)',
                color: '#fbbf24',
                border: '1px solid rgba(245, 158, 11, 0.45)',
                borderRadius: 6,
                padding: '4px 10px',
                fontSize: 11,
                fontWeight: 600,
                cursor: 'pointer',
              }}
              title="Configure C2 API Keys & Live Feeds (AISStream, Cesium Ion, CDSE)"
            >
              <span
                style={{
                  width: 6,
                  height: 6,
                  borderRadius: '50%',
                  background: '#22c55e',
                  boxShadow: '0 0 6px #22c55e',
                }}
              />
              <span>API Keys</span>
            </button>
          </div>
        </div>
      </div>

      {/* ── TACTICAL C2 VESSEL DOSSIER INSPECTION DRAWER ─────────────────────── */}
      {selectedVessel && (
        <div
          style={{
            position: 'absolute',
            top: 130,
            right: 16,
            width: 330,
            background: 'rgba(10, 16, 30, 0.82)',
            backdropFilter: 'blur(24px) saturate(190%)',
            WebkitBackdropFilter: 'blur(24px) saturate(190%)',
            border: `1px solid ${
              selectedVessel.risk === 'CRITICAL'
                ? 'rgba(239, 68, 68, 0.65)'
                : selectedVessel.risk === 'HIGH'
                ? 'rgba(249, 115, 22, 0.65)'
                : 'rgba(56, 189, 248, 0.45)'
            }`,
            borderRadius: 14,
            boxShadow: '0 20px 50px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.15)',
            zIndex: 35,
            padding: 16,
            color: '#e2e8f0',
            fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
          }}
        >
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'flex-start',
              marginBottom: 12,
            }}
          >
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ fontSize: 16 }}>⛵</span>
                <span style={{ fontWeight: 700, fontSize: 14, color: '#ffffff' }}>
                  {selectedVessel.name}
                </span>
              </div>
              <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2, fontFamily: 'monospace' }}>
                MMSI: {selectedVessel.mmsi} · IMO: {selectedVessel.imo}
              </div>
            </div>
            <button
              onClick={() => setSelectedVessel(null)}
              style={{
                background: 'rgba(255, 255, 255, 0.08)',
                border: 'none',
                color: '#94a3b8',
                fontSize: 14,
                width: 26,
                height: 26,
                borderRadius: '50%',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              ✕
            </button>
          </div>

          {/* Risk Classification Badge */}
          <div
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 6,
              padding: '4px 10px',
              borderRadius: 6,
              fontSize: 11,
              fontWeight: 700,
              marginBottom: 12,
              background:
                selectedVessel.risk === 'CRITICAL'
                  ? 'rgba(239, 68, 68, 0.18)'
                  : selectedVessel.risk === 'HIGH'
                  ? 'rgba(249, 115, 22, 0.18)'
                  : 'rgba(56, 189, 248, 0.18)',
              color:
                selectedVessel.risk === 'CRITICAL'
                  ? '#f87171'
                  : selectedVessel.risk === 'HIGH'
                  ? '#fb923c'
                  : '#38bdf8',
              border: `1px solid ${
                selectedVessel.risk === 'CRITICAL'
                  ? 'rgba(239, 68, 68, 0.45)'
                  : selectedVessel.risk === 'HIGH'
                  ? 'rgba(249, 115, 22, 0.45)'
                  : 'rgba(56, 189, 248, 0.45)'
              }`,
            }}
          >
            <span
              style={{
                width: 6,
                height: 6,
                borderRadius: '50%',
                background:
                  selectedVessel.risk === 'CRITICAL'
                    ? '#ef4444'
                    : selectedVessel.risk === 'HIGH'
                    ? '#f97316'
                    : '#38bdf8',
              }}
            />
            <span>
              CLASSIFICATION: {selectedVessel.risk} RISK ({(selectedVessel.attribution_score * 100).toFixed(0)}%)
            </span>
          </div>

          {/* Telemetry Metrics Grid */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: 8,
              fontSize: 11,
              background: 'rgba(0, 0, 0, 0.35)',
              padding: 10,
              borderRadius: 8,
              marginBottom: 12,
              border: '1px solid rgba(255, 255, 255, 0.06)',
            }}
          >
            <div>
              <span style={{ color: '#64748b', fontSize: 10, fontWeight: 700 }}>TYPE:</span>
              <div style={{ color: '#e2e8f0', fontWeight: 600 }}>{selectedVessel.type}</div>
            </div>
            <div>
              <span style={{ color: '#64748b', fontSize: 10, fontWeight: 700 }}>FLAG:</span>
              <div style={{ color: '#e2e8f0', fontWeight: 600 }}>{selectedVessel.flag}</div>
            </div>
            <div>
              <span style={{ color: '#64748b', fontSize: 10, fontWeight: 700 }}>SPEED (SOG):</span>
              <div style={{ color: '#38bdf8', fontWeight: 700, fontFamily: 'monospace' }}>
                {selectedVessel.sog} kn
              </div>
            </div>
            <div>
              <span style={{ color: '#64748b', fontSize: 10, fontWeight: 700 }}>COURSE (COG):</span>
              <div style={{ color: '#38bdf8', fontWeight: 700, fontFamily: 'monospace' }}>
                {selectedVessel.cog}°
              </div>
            </div>
            <div>
              <span style={{ color: '#64748b', fontSize: 10, fontWeight: 700 }}>CPA TO DISCHARGE ORIGIN:</span>
              <div style={{ color: '#f87171', fontWeight: 700, fontFamily: 'monospace' }}>
                {selectedVessel.cpa_nm > 0 ? `${selectedVessel.cpa_nm.toFixed(1)} nm` : '---'}
              </div>
            </div>
            <div>
              <span style={{ color: '#64748b', fontSize: 10, fontWeight: 700 }}>DIST TO OBSERVED SLICK:</span>
              <div style={{ color: '#38bdf8', fontWeight: 700, fontFamily: 'monospace' }}>
                {activeScenario && selectedVessel.lat && selectedVessel.lng
                  ? `${(computeDistanceKm(selectedVessel.lat, selectedVessel.lng, activeScenario.lat, activeScenario.lng) / 1.852).toFixed(1)} nm`
                  : '---'}
              </div>
            </div>
            <div style={{ gridColumn: 'span 2' }}>
              <span style={{ color: '#64748b', fontSize: 10, fontWeight: 700 }}>AIS CONTINUITY:</span>
              <div
                style={{
                  color: selectedVessel.ais_gap_hours > 1.0 ? '#ef4444' : '#22c55e',
                  fontWeight: 600,
                  fontSize: 11,
                }}
              >
                {selectedVessel.ais_gap_hours ? `${selectedVessel.ais_gap_hours.toFixed(1)} hrs silent` : 'Continuous Broadcast (Live Feed)'}
              </div>
            </div>
          </div>

          {/* Track Vessel Button */}
          <button
            onClick={() => {
              if (!viewerRef.current) return;
              viewerRef.current.camera.flyTo({
                destination: Cesium.Cartesian3.fromDegrees(
                  selectedVessel.lng,
                  selectedVessel.lat - 0.04,
                  6000
                ),
                orientation: {
                  heading: Cesium.Math.toRadians(selectedVessel.cog),
                  pitch: Cesium.Math.toRadians(-35),
                  roll: 0,
                },
                duration: 1.5,
              });
            }}
            style={{
              width: '100%',
              padding: '9px',
              background: 'linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%)',
              color: '#ffffff',
              border: 'none',
              borderRadius: 8,
              cursor: 'pointer',
              fontWeight: 700,
              fontSize: 12,
              letterSpacing: '0.04em',
              boxShadow: '0 4px 14px rgba(139, 92, 246, 0.45)',
              transition: 'all 0.15s ease',
            }}
          >
            🎯 TRACK VESSEL IN 3D
          </button>
        </div>
      )}

      {/* ── FLOATING GLASS TELEMETRY DOCK (BOTTOM) ───────────────────────────── */}
      <div
        style={{
          position: 'absolute',
          bottom: 14,
          left: 16,
          right: 16,
          zIndex: 30,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '7px 18px',
          background: 'rgba(10, 16, 30, 0.72)',
          backdropFilter: 'blur(24px) saturate(190%)',
          WebkitBackdropFilter: 'blur(24px) saturate(190%)',
          border: '1px solid rgba(255, 255, 255, 0.10)',
          borderRadius: 10,
          boxShadow: '0 12px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.12)',
          fontSize: 11,
          fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
          color: '#94a3b8',
          pointerEvents: 'none',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 18 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: 9.5, fontWeight: 700, color: '#64748b', letterSpacing: '0.05em' }}>SENSOR:</span>
            <span
              style={{
                background: 'rgba(168, 85, 247, 0.18)',
                color: '#c084fc',
                padding: '2px 7px',
                borderRadius: 4,
                fontWeight: 700,
                fontSize: 10.5,
                border: '1px solid rgba(168, 85, 247, 0.3)',
              }}
            >
              SENTINEL-1 C-SAR IW
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: 9.5, fontWeight: 700, color: '#64748b', letterSpacing: '0.05em' }}>CURSOR:</span>
            <span ref={cursorHudRef} style={{ color: '#38bdf8', fontFamily: 'monospace', fontWeight: 600 }}>
              ---
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: 9.5, fontWeight: 700, color: '#64748b', letterSpacing: '0.05em' }}>ALTITUDE:</span>
            <span ref={altHudRef} style={{ color: '#38bdf8', fontFamily: 'monospace', fontWeight: 600 }}>
              ---
            </span>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 18 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: 9.5, fontWeight: 700, color: '#64748b', letterSpacing: '0.05em' }}>LIVE AIS:</span>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 5,
                background: 'rgba(34, 197, 94, 0.15)',
                padding: '2px 8px',
                borderRadius: 4,
                border: '1px solid rgba(34, 197, 94, 0.35)',
              }}
            >
              <span
                style={{
                  width: 6,
                  height: 6,
                  borderRadius: '50%',
                  background: '#22c55e',
                  boxShadow: '0 0 6px #22c55e',
                }}
              />
              <span style={{ color: '#4ade80', fontWeight: 700, fontSize: 10.5 }}>
                {liveAisCount} ACTIVE VESSELS
              </span>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: 9.5, fontWeight: 700, color: '#64748b', letterSpacing: '0.05em' }}>THEATER:</span>
            <span style={{ color: '#f1f5f9', fontWeight: 600, fontSize: 10.5 }}>
              {!activeScenarioKey
                ? 'INDIAN OCEAN REGION'
                : activeScenarioKey.includes('001') || activeScenarioKey.includes('004') || activeScenarioKey.includes('006')
                ? 'ARABIAN SEA'
                : 'BAY OF BENGAL'}
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: 9.5, fontWeight: 700, color: '#64748b', letterSpacing: '0.05em' }}>BOUNDARY:</span>
            <span
              style={{
                background: 'rgba(6, 182, 212, 0.18)',
                color: '#22d3ee',
                padding: '2px 7px',
                borderRadius: 4,
                fontWeight: 700,
                fontSize: 10.5,
                border: '1px solid rgba(6, 182, 212, 0.3)',
              }}
            >
              INDIAN 200 NM EEZ ACTIVE
            </span>
          </div>
        </div>
      </div>

      {/* ── LOADING OVERLAY ──────────────────────────────────────────────────── */}
      {status === 'loading' && (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'rgba(2, 6, 23, 0.85)',
            backdropFilter: 'blur(20px)',
            color: '#c4b5fd',
            gap: 16,
            zIndex: 50,
          }}
        >
          <div className="w-12 h-12 border-4 border-violet-600 border-t-violet-300 rounded-full animate-spin" />
          <div style={{ textAlign: 'center' }}>
            <p style={{ fontFamily: 'Inter, sans-serif', fontWeight: 700, fontSize: 14, margin: '0 0 4px 0', letterSpacing: '0.05em' }}>
              INITIALIZING 3D MARITIME COMMON OPERATING PICTURE
            </p>
            <p style={{ fontFamily: 'Inter, sans-serif', fontSize: 12, color: '#a78bfa', margin: 0 }}>
              Loading Cesium 3D Globe · Live AISStream Vessels · Indian EEZ Boundaries
            </p>
          </div>
        </div>
      )}

      {/* ── ERROR OVERLAY ────────────────────────────────────────────────────── */}
      {status === 'error' && (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'rgba(2, 6, 23, 0.95)',
            backdropFilter: 'blur(20px)',
            color: '#ef4444',
            gap: 12,
            zIndex: 50,
          }}
        >
          <span style={{ fontSize: 40 }}>⚠</span>
          <p style={{ fontFamily: 'Inter, sans-serif', fontWeight: 700, fontSize: 14, margin: 0 }}>
            3D Globe WebGL Initialization Failed
          </p>
          <p style={{ fontFamily: 'Inter, sans-serif', fontSize: 12, color: '#94a3b8', margin: 0 }}>
            Ensure WebGL 2 hardware acceleration is enabled in your browser settings.
          </p>
        </div>
      )}

      {/* ── C2 API KEYS & LIVE FEEDS SETUP MODAL ─────────────────────────────── */}
      <ApiKeySetupModal
        isOpen={isApiKeyModalOpen}
        onClose={() => setIsApiKeyModalOpen(false)}
        onKeysUpdated={() => {
          if (basemap === 'ion') {
            applyBasemap('ion');
          }
        }}
      />
    </div>
  );
};

export default GlobeView;
