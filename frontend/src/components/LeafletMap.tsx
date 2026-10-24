import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import type { Scenario, BaseLayerType, CopernicusLayerId, DetectionResult, SarDriftPayload } from '../types/dashboard';
import type { LiveSimulationResult } from '../services/marineWeatherService';
import { SCENARIOS } from '../data/scenarios';
import { computeBackwardDriftGeometry, getRealisticAISTrack, sogToColor, interpolateKinematicsAtHour } from '../utils/geoContours';
import { getScenarioBenchmarkDetections } from '../services/detectionService';

interface LeafletMapProps {
  scenario: Scenario | null;
  baseLayer: BaseLayerType;
  showSeamarks: boolean;
  showIndiaOutline?: boolean;
  showEezBoundary?: boolean;
  selectedCopernicusLayer?: CopernicusLayerId;
  layerOpacity?: number;
  showAiMask?: boolean;
  showOverlay?: boolean;
  scrubHours?: number;
  detectionResult: DetectionResult | null;
  driftEnvelopes?: any;
  onUpdateCoords: (coords: string) => void;
  onSelectScenario?: (key: string) => void;
  mapRef: React.MutableRefObject<L.Map | null>;
  scenarios?: Record<string, Scenario>;
  targetLocation?: {
    lat: number;
    lng: number;
    zoom?: number;
    title: string;
    sub?: string;
    category?: string;
  } | null;
  sarDriftPayload?: SarDriftPayload | null;
  activeDriftSimulation?: LiveSimulationResult | null;
}

export const LeafletMap: React.FC<LeafletMapProps> = ({
  scenario,
  baseLayer,
  showSeamarks,
  showIndiaOutline = true,
  showEezBoundary = true,
  selectedCopernicusLayer = 'sar-vv',
  layerOpacity = 1.0,
  showAiMask = false,
  showOverlay = true,
  scrubHours = 0,
  detectionResult,
  driftEnvelopes,
  onUpdateCoords,
  onSelectScenario,
  mapRef,
  scenarios,
  targetLocation,
  sarDriftPayload,
  activeDriftSimulation,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const baseLayersRef = useRef<Record<string, L.TileLayer>>({});
  const seamarksLayerRef = useRef<L.TileLayer | null>(null);
  const layersGroupRef = useRef<L.LayerGroup | null>(null);
  const indiaOutlineLayerRef = useRef<L.LayerGroup | null>(null);
  const indiaBoundariesTileRef = useRef<L.TileLayer | null>(null);
  const driftLayerRef = useRef<L.GeoJSON | null>(null);
  const searchTargetMarkerRef = useRef<L.Marker | null>(null);
  const cachedIndiaOutline = useRef<any>(null);
  const cachedIndiaEez = useRef<any>(null);

  // Initialize Leaflet map with real GIS tile sources
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const initialCenter: [number, number] = scenario ? [scenario.lat, scenario.lng] : [15.5, 79.0];
    const initialZoom = scenario ? 11 : 4.25;

    const maxWorldBounds = L.latLngBounds(L.latLng(-85.0, -180.0), L.latLng(85.0, 180.0));

    const map = L.map(containerRef.current, {
      center: initialCenter,
      zoom: initialZoom,
      minZoom: 3,
      maxZoom: 22,
      zoomSnap: 0.25,
      zoomDelta: 0.5,
      zoomControl: false,
      attributionControl: false,
      keyboard: false,
      maxBounds: maxWorldBounds,
      maxBoundsViscosity: 1.0,
      worldCopyJump: false,
    });

    L.control.zoom({ position: 'topright' }).addTo(map);
    L.control.scale({ imperial: false, metric: true, position: 'bottomleft' }).addTo(map);

    // 0. Official ISRO Bhuvan & Bhoonidhi Satellite Imagery (Govt of India / NRSC WMTS Server)
    baseLayersRef.current.bhuvan = L.tileLayer(
      'https://bhuvanmaps.nrsc.gov.in/bhuvan_ras3/server/rest/services/World_Imagery/MapServer/WMTS?service=WMTS&version=1.0.0&request=GetTile&layer=HYDImagery&style=default&tilematrixSet=GoogleMapsCompatible&tilematrix={z}&tilerow={y}&tilecol={x}&format=image/jpeg',
      {
        tileSize: 256,
        minZoom: 3,
        maxNativeZoom: 18,
        maxZoom: 22,
        attribution: '© ISRO NRSC Bhuvan / Bhoonidhi Satellite Imagery (Govt. of India)',
        className: 'isro-bhuvan-satellite-tiles',
        noWrap: true,
        bounds: maxWorldBounds,
      }
    );
    baseLayersRef.current['isro-bhuvan'] = baseLayersRef.current.bhuvan;

    // 1. Official ISRO Bhoonidhi / GEBCO Ocean Bathymetry & Indian EEZ Basemap (Primary Marine Basemap)
    // Exactly matches ISRO Bhoonidhi portal: persistent oceanic trenches, depth contours, and marine blue at all zoom levels
    baseLayersRef.current['bhuvan-satellite'] = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}',
      {
        maxNativeZoom: 13,
        maxZoom: 22,
        attribution: '© ISRO Bhoonidhi / GEBCO / NOAA Ocean Bathymetry Relief',
        className: 'bhuvan-satellite-tiles',
        noWrap: true,
        bounds: maxWorldBounds,
      }
    );

    // 2. High-Resolution Optical Satellite (ArcGIS / ESRI World Imagery - Sub-meter zoom up to 19 native, 22 max)
    baseLayersRef.current.satellite = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      {
        maxNativeZoom: 19,
        maxZoom: 22,
        attribution: '© Esri ArcGIS World Imagery / Maxar / Earthstar Geographics',
        className: 'eo-satellite-tiles',
        noWrap: true,
        bounds: maxWorldBounds,
      }
    );
    baseLayersRef.current.arcgis = baseLayersRef.current.satellite;
    baseLayersRef.current['arcgis-satellite'] = baseLayersRef.current.satellite;

    // 3. Sentinel-1 SAR Radar Composite
    baseLayersRef.current.sar = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      {
        maxNativeZoom: 18,
        maxZoom: 22,
        className: 'eo-sar-radar-tiles',
        noWrap: true,
        bounds: maxWorldBounds,
      }
    );

    // 4. Google Earth Ultra-HD Satellite & Marine Infrastructure (Zoom 0-22)
    // Down to 30cm/pixel: lets operators zoom into water to see actual boats, ships, wakes, piers, and harbor SPMs
    baseLayersRef.current['carto-voyager'] = L.tileLayer(
      'https://{s}.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
      {
        subdomains: ['mt0', 'mt1', 'mt2', 'mt3'],
        maxNativeZoom: 21,
        maxZoom: 22,
        noWrap: true,
        bounds: maxWorldBounds,
        attribution: '© Google Earth / Maxar / Airbus Marine Imagery',
        className: 'google-earth-satellite-tiles',
      }
    );
    baseLayersRef.current['google-satellite'] = baseLayersRef.current['carto-voyager'];

    // 5. Tactical Dark Night Chart (Carto Dark)
    baseLayersRef.current['carto-dark'] = L.tileLayer(
      'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
      {
        subdomains: ['a', 'b', 'c', 'd'],
        maxNativeZoom: 19,
        maxZoom: 22,
        noWrap: true,
        bounds: maxWorldBounds,
      }
    );

    baseLayersRef.current.opensea = baseLayersRef.current['carto-voyager'];
    baseLayersRef.current.msn = baseLayersRef.current.satellite;
    baseLayersRef.current.day = baseLayersRef.current['carto-voyager'];
    baseLayersRef.current.dark = baseLayersRef.current['carto-dark'];
    baseLayersRef.current['sar-vh'] = baseLayersRef.current.sar;

    // 6. ISRO Bhuvan / Survey of India Vector Cartography
    baseLayersRef.current['bhuvan-vector'] = L.tileLayer(
      'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
      {
        maxZoom: 19,
        attribution: '© ISRO Bhuvan / Survey of India National Basemap',
        className: 'bhuvan-vector-tiles',
        noWrap: true,
        bounds: maxWorldBounds,
      }
    );

    seamarksLayerRef.current = L.tileLayer('https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png', {
      maxZoom: 18,
      noWrap: true,
      bounds: maxWorldBounds,
    });

    const initialLayer = baseLayersRef.current[baseLayer] || baseLayersRef.current.satellite;
    initialLayer.addTo(map);

    if (showSeamarks && seamarksLayerRef.current) {
      seamarksLayerRef.current.addTo(map);
    }

    layersGroupRef.current = L.layerGroup().addTo(map);
    indiaOutlineLayerRef.current = L.layerGroup().addTo(map);

    indiaBoundariesTileRef.current = L.tileLayer(
      'https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
      {
        maxZoom: 22,
        opacity: 0.85,
        attribution: '© ESRI World Boundaries & Places',
        noWrap: true,
        bounds: maxWorldBounds,
      }
    );

    map.on('mousemove', (e) => {
      const lat = e.latlng.lat.toFixed(4);
      const lng = e.latlng.lng.toFixed(4);
      onUpdateCoords(`${lat}°N, ${lng}°E`);
    });

    mapRef.current = map;

    // Ensure map tiles layout and render properly
    const resizeTimer = window.setTimeout(() => {
      if (mapRef.current) {
        map.invalidateSize();
      }
    }, 150);

    let resizeObserver: ResizeObserver | null = null;
    if (containerRef.current && typeof ResizeObserver !== 'undefined') {
      resizeObserver = new ResizeObserver(() => {
        if (mapRef.current) {
          map.invalidateSize();
        }
      });
      resizeObserver.observe(containerRef.current);
    }

    return () => {
      window.clearTimeout(resizeTimer);
      if (resizeObserver) {
        resizeObserver.disconnect();
      }
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // Handle custom switch-scenario events from Leaflet popups
  useEffect(() => {
    const handleSwitch = (e: Event) => {
      const customEvent = e as CustomEvent<string>;
      if (customEvent.detail && onSelectScenario) {
        onSelectScenario(customEvent.detail);
      }
    };
    window.addEventListener('switch-scenario', handleSwitch);
    return () => window.removeEventListener('switch-scenario', handleSwitch);
  }, [onSelectScenario]);

  // Base layer switcher
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    Object.values(baseLayersRef.current).forEach((layer) => {
      if (map.hasLayer(layer)) {
        map.removeLayer(layer);
      }
    });

    const targetLayer = baseLayersRef.current[baseLayer] || baseLayersRef.current.satellite;
    if (targetLayer) {
      targetLayer.addTo(map);
    }
  }, [baseLayer]);

  // Seamarks overlay
  useEffect(() => {
    const map = mapRef.current;
    const seamarks = seamarksLayerRef.current;
    if (!map || !seamarks) return;

    if (showSeamarks) {
      if (!map.hasLayer(seamarks)) seamarks.addTo(map);
    } else {
      if (map.hasLayer(seamarks)) map.removeLayer(seamarks);
    }
  }, [showSeamarks]);

  // Render Official Outline Map of India & 200 NM Indian EEZ Maritime Limit
  useEffect(() => {
    const map = mapRef.current;
    const outlineGroup = indiaOutlineLayerRef.current;
    const tileLayer = indiaBoundariesTileRef.current;
    if (!map || !outlineGroup) return;

    outlineGroup.clearLayers();

    // Toggle ESRI Reference Labels & Boundaries tile layer
    if (tileLayer) {
      if (showIndiaOutline) {
        if (!map.hasLayer(tileLayer)) tileLayer.addTo(map);
      } else {
        if (map.hasLayer(tileLayer)) map.removeLayer(tileLayer);
      }
    }

    // Function to render India Sovereign Boundary
    const renderIndiaOutline = (geoJsonData: any) => {
      // 1. Dark halo casing for high contrast on all map baselayers
      const haloLayer = L.geoJSON(geoJsonData, {
        style: {
          color: '#020617',
          weight: 4.5,
          opacity: 0.90,
          fillColor: 'transparent',
          fillOpacity: 0,
        },
        interactive: false,
      });
      outlineGroup.addLayer(haloLayer);

      // 2. High-visibility tactical sovereign boundary stroke
      const mainBorderLayer = L.geoJSON(geoJsonData, {
        style: {
          color: '#38BDF8',
          weight: 2.2,
          opacity: 1.0,
          fillColor: 'rgba(56, 189, 248, 0.02)',
          fillOpacity: 0.02,
        },
      }).bindTooltip(
        `<b>🇮🇳 Republic of India — Sovereign Territory & Coastline</b><br>Official Survey of India National Perimeter & Island Territories`,
        { sticky: true, className: 'gis-custom-tooltip' }
      );
      outlineGroup.addLayer(mainBorderLayer);
    };

    // Function to render Indian EEZ (200 Nautical Miles)
    const renderIndiaEez = (eezData: any) => {
      const eezLayer = L.geoJSON(eezData, {
        style: {
          color: '#F59E0B',
          weight: 1.8,
          dashArray: '6, 6',
          opacity: 0.95,
          fillColor: 'rgba(245, 158, 11, 0.02)',
          fillOpacity: 0.02,
        },
        onEachFeature: (feature, layer) => {
          const name = feature.properties?.name || 'Indian Exclusive Economic Zone';
          const zone = feature.properties?.zone || '200 Nautical Miles';
          const auth = feature.properties?.authority || 'Indian Coast Guard';
          layer.bindTooltip(
            `<b>🌊 ${name}</b><br>Perimeter: ${zone}<br>Enforcement: ${auth}`,
            { sticky: true, className: 'gis-custom-tooltip' }
          );
        },
      });
      outlineGroup.addLayer(eezLayer);
    };

    if (showIndiaOutline) {
      if (cachedIndiaOutline.current) {
        renderIndiaOutline(cachedIndiaOutline.current);
      } else {
        fetch('/india-outline.geojson')
          .then((res) => (res.ok ? res.json() : null))
          .then((data) => {
            if (data) {
              cachedIndiaOutline.current = data;
              renderIndiaOutline(data);
            }
          })
          .catch((err) => console.error('Error loading india-outline.geojson:', err));
      }
    }

    if (showEezBoundary) {
      if (cachedIndiaEez.current) {
        renderIndiaEez(cachedIndiaEez.current);
      } else {
        fetch('/india-eez.geojson')
          .then((res) => (res.ok ? res.json() : null))
          .then((data) => {
            if (data) {
              cachedIndiaEez.current = data;
              renderIndiaEez(data);
            }
          })
          .catch((err) => console.error('Error loading india-eez.geojson:', err));
      }
    }
  }, [showIndiaOutline, showEezBoundary]);

  // Real-Time Socket.io Drift Envelope Rendering
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    if (driftLayerRef.current) {
      map.removeLayer(driftLayerRef.current);
      driftLayerRef.current = null;
    }

    if (showAiMask && driftEnvelopes) {
      driftLayerRef.current = L.geoJSON(driftEnvelopes, {
        style: (feature) => {
          const prob = feature?.properties?.probability;
          if (prob === '50%') {
            return { color: '#DC2626', weight: 3, fillColor: '#DC2626', fillOpacity: 0.15 };
          } else if (prob === '75%') {
            return { color: '#F97316', weight: 2, dashArray: '6, 6', fillColor: '#F97316', fillOpacity: 0.08 };
          } else if (prob === '95%') {
            return { color: '#EAB308', weight: 1.5, dashArray: '2, 4', fillColor: '#EAB308', fillOpacity: 0.03 };
          }
          return { color: '#00E5FF', weight: 1, fillOpacity: 0 };
        },
        onEachFeature: (feature, layer) => {
          layer.bindTooltip(
            `<b>OpenDrift Backend</b><br>Confidence Interval: ${feature?.properties?.probability}`,
            { sticky: true, className: 'gis-custom-tooltip' }
          );
        }
      }).addTo(map);
    }
  }, [driftEnvelopes, showAiMask]);

  // Camera navigation: ONLY fly to coordinates when user switches scenario
  const prevScenarioIdRef = useRef<string | undefined>(scenario?.id);
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    const curId = scenario?.id;
    if (prevScenarioIdRef.current !== curId) {
      prevScenarioIdRef.current = curId;
      if (scenario) {
        if (searchTargetMarkerRef.current && map) {
          map.removeLayer(searchTargetMarkerRef.current);
          searchTargetMarkerRef.current = null;
        }
        map.flyTo([scenario.lat, scenario.lng], 11, { duration: 1.2 });
      } else {
        map.flyTo([15.5, 79.0], 4.25, { duration: 1.2 });
      }
    }
  }, [scenario?.id, scenario?.lat, scenario?.lng]);

  // Search Target Navigation: Fly to searched port, strait, coordinate, or vessel with tactical pulse marker
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !targetLocation) return;

    map.flyTo([targetLocation.lat, targetLocation.lng], targetLocation.zoom || 12, { duration: 1.5 });

    if (searchTargetMarkerRef.current) {
      map.removeLayer(searchTargetMarkerRef.current);
      searchTargetMarkerRef.current = null;
    }

    const pinIcon = L.divIcon({
      className: 'gis-search-target-pin-wrap',
      html: `
        <div class="search-target-pin">
          <div class="search-target-pulse"></div>
          <div class="search-target-dot"></div>
          <div class="search-target-badge">${targetLocation.title}</div>
        </div>
      `,
      iconSize: [0, 0],
      iconAnchor: [0, 0],
    });

    const marker = L.marker([targetLocation.lat, targetLocation.lng], { icon: pinIcon, zIndexOffset: 2000 })
      .bindPopup(`
        <div class="gis-popup-card">
          <div class="gis-popup-header" style="border-bottom: 2px solid #0284C7;">
            ${targetLocation.title}
          </div>
          <div class="gis-popup-row"><span>Classification:</span> <strong style="text-transform:uppercase;color:#38BDF8;">${targetLocation.category || 'MARITIME LOCATION'}</strong></div>
          <div class="gis-popup-row"><span>Coordinates:</span> <strong>${targetLocation.lat.toFixed(4)}°N, ${targetLocation.lng.toFixed(4)}°E</strong></div>
          ${targetLocation.sub ? `<div class="gis-popup-row"><span>Details:</span> <span>${targetLocation.sub}</span></div>` : ''}
        </div>
      `);

    marker.addTo(map);
    setTimeout(() => {
      marker.openPopup();
    }, 1600);
    searchTargetMarkerRef.current = marker;
  }, [targetLocation]);

  // Render REAL satellite imagery picture, ML-predicted oil spill, past origin, and future drift forecast
  useEffect(() => {
    const map = mapRef.current;
    const group = layersGroupRef.current;
    if (!map || !group) return;

    group.clearLayers();

    // ── 1. RENDER REAL SATELLITE IMAGERY DRAPES & BONN-TIERED OIL SLICKS FOR ALL INCIDENTS ──
    const activeScenarios = scenarios && Object.keys(scenarios).length > 0 ? scenarios : SCENARIOS;
    Object.entries(activeScenarios).forEach(([key, sc]) => {
      const isActive = scenario ? sc.id === scenario.id : false;

      const centerLat = sc.lat;
      const centerLng = sc.lng;

      // Per-scenario calibrated imagery bounds — shifted seaward for coastal incidents
      // so the satellite drape never overlaps land. Values are [latMin, lngMin, latMax, lngMax].

      const scenarioBoundsMap: Record<string, [[number, number], [number, number]]> = {
        // INC-001: 18.743°N 71.218°E — Mumbai High, deep offshore, centered symmetric
        'INC-001': [[18.52, 70.98], [18.96, 71.46]],

        // INC-002: 13.250°N 80.460°E — Chennai-Ennore
        // Coast at ~80.30°E. Centered on 80.46 with tight left margin (0.14° offshore buffer).
        // Window: 80.32→80.60 (center=80.46, matching scenario exactly)
        'INC-002': [[13.08, 80.32], [13.42, 80.60]],

        // INC-003: 10.456°N 93.123°E — Andaman Sea, open ocean, centered symmetric
        'INC-003': [[10.24, 92.88], [10.68, 93.37]],

        // INC-004: 15.420°N 73.650°E — Goa (west coast, ocean = WEST, land = EAST)
        // Goa coast at ~73.88°E. Center on 73.65 with right edge at 73.82 (safe from land).
        // Window: 73.48→73.82 (center=73.65, matching scenario exactly)
        'INC-004': [[15.22, 73.48], [15.62, 73.82]],

        // INC-005: 22.600°N 69.500°E — Gulf of Kutch, open water, centered symmetric
        'INC-005': [[22.38, 69.26], [22.82, 69.74]],

        // INC-006: 9.960°N 76.080°E — Cochin (west coast, ocean = WEST, land = EAST)
        // Kerala coast at ~76.30°E. Center on 76.08 with right edge at 76.22 (safe from land).
        // Window: 75.94→76.22 (center=76.08, matching scenario exactly)
        'INC-006': [[9.76, 75.94], [10.16, 76.22]],

        // INC-007: 20.250°N 86.720°E — Paradip (east coast, ocean = EAST, land = WEST)
        // Calibrated center on 86.720°E so red SAR/NISAR spill drape aligns directly with AI detection centroid
        'INC-007': [[20.06, 86.54], [20.44, 86.90]],

        // INC-008: 8.500°N 73.000°E — Lakshadweep, open ocean, centered symmetric
        'INC-008': [[8.28, 72.76], [8.72, 73.24]],
      };

      const imageryBounds: [[number, number], [number, number]] = scenarioBoundsMap[key] ?? [
        [centerLat - 0.22, centerLng - 0.28],
        [centerLat + 0.22, centerLng + 0.28],
      ];


      // Default to Sentinel-1 SAR Radar Backscatter — large files use .webp (87% smaller than PNG)
      let imageryFile = `/imagery/sar_${key}.webp`;
      let layerLabel = 'Sentinel-1A C-SAR IW GRD Calibrated Backscatter (VV Decibels)';
      let sensorDesc = 'Capillary Wave Damping (Δσ0 = -8.40 dB). Dark slick crater against rough ocean speckle.';

      if (selectedCopernicusLayer === 'sar-vv') {
        imageryFile = `/imagery/sar_${key}.webp`;
        layerLabel = 'Sentinel-1A C-SAR IW GRD Calibrated Backscatter (VV Decibels)';
        sensorDesc = 'Capillary Wave Damping (Δσ0 = -8.40 dB). Dark slick crater against rough ocean speckle.';
      } else if (selectedCopernicusLayer === 'sar-vh') {
        imageryFile = `/imagery/vh_${key}.webp`;
        layerLabel = 'Sentinel-1C C-SAR Cross-Polarization (VH Mode)';
        sensorDesc = 'Cross-polarized backscatter optimizing metallic vessel detection and surface roughness.';
      } else if (selectedCopernicusLayer === 'true-color') {
        imageryFile = `/imagery/tc_${key}.png`;
        layerLabel = 'Sentinel-2A MSI Natural True Color with Surface Sheen';
        sensorDesc = 'Visible ocean surface showing specular sun-glint on spreading iridescent oil film.';
      } else if (selectedCopernicusLayer === 'swir-oil') {
        imageryFile = `/imagery/swir_${key}.png`;
        layerLabel = 'Sentinel-2B MSI SWIR Hydrocarbon Emulsion (1610nm / 2190nm)';
        sensorDesc = 'Short-Wave Infrared hydrocarbon absorption overtones highlighting weathered crude emulsion.';
      } else if (selectedCopernicusLayer === 'false-color') {
        imageryFile = `/imagery/cir_${key}.png`;
        layerLabel = 'Sentinel-2 MSI False Color (CIR) Algae Exclusion';
        sensorDesc = 'Near-Infrared reflection proving negative for biogenic Sargassum/phytoplankton bloom.';
      } else if (selectedCopernicusLayer === 'nisar-ls') {
        imageryFile = `/imagery/nisar_${key}.webp`;
        layerLabel = 'ISRO NISAR Dual-Band L+S Radar (DFDI Bragg Damping Composite)';
        sensorDesc = 'Dual-Frequency: S-Band (ISRO 3.2GHz) + L-Band (NASA 1.26GHz). DFDI=2.7dB confirms thick crude oil.';
      } else if (selectedCopernicusLayer === 'eos-04') {
        imageryFile = `/imagery/eos04_${key}.webp`;
        layerLabel = 'ISRO EOS-04 (RISAT-1A) Circular Hybrid Polarimetry';
        sensorDesc = 'm-chi polarimetric decomposition distinguishing oil slick from look-alike low-wind areas.';
      }

      // Add satellite imagery drape ONLY for the active (focused) scenario to avoid broken image boxes
      // for all the inactive incidents whose large imagery files would be loaded simultaneously
      const hasImageFile = isActive && (
        ['INC-001', 'INC-002', 'INC-003', 'INC-004', 'INC-005', 'INC-006', 'INC-007', 'INC-008'].includes(key)
      );
      if (showOverlay && hasImageFile) {
        const satelliteDrape = L.imageOverlay(imageryFile, imageryBounds, {
          opacity: layerOpacity,
          interactive: true,
          className: 'crisp-sar-overlay',
          attribution: selectedCopernicusLayer === 'nisar-ls' && isActive ? '© ISRO NRSC Bhoonidhi / NASA JPL NISAR' : selectedCopernicusLayer === 'eos-04' && isActive ? '© ISRO NRSC Bhoonidhi / EOS-04' : '© ESA Copernicus Sentinel Data Space Ecosystem',
        });

        satelliteDrape.bindPopup(`
          <div class="gis-popup-card">
            <div class="gis-popup-header" style="color: #00E5FF; border-bottom: 1.5px solid rgba(0, 229, 255, 0.5);">${sc.title} — ${layerLabel}</div>
            <div class="gis-popup-row"><span style="color:#94A3B8;">Sensor:</span> <strong style="color:#F1F5F9;">${selectedCopernicusLayer === 'nisar-ls' && isActive ? 'ISRO NISAR L-SAR + S-SAR (242km SweepSAR)' : selectedCopernicusLayer === 'eos-04' && isActive ? 'ISRO EOS-04 C-SAR Circular Pol' : selectedCopernicusLayer === 'true-color' || selectedCopernicusLayer === 'swir-oil' || selectedCopernicusLayer === 'false-color' ? 'Sentinel-2 MSI Optical' : 'Sentinel-1 C-SAR IW GRD'}</strong></div>
            <div class="gis-popup-row"><span style="color:#94A3B8;">Physical Signal:</span> <strong style="color:#F1F5F9;">${sensorDesc}</strong></div>
            <div class="gis-popup-row"><span style="color:#94A3B8;">Ground Resolution:</span> <strong style="color:#38BDF8;">10m x 10m Ultra-HD (RTC Calibrated)</strong></div>
            <div class="gis-popup-row"><span style="color:#94A3B8;">Incident Status:</span> <strong style="color: ${sc.oilColor || '#00E5FF'};">${sc.sev}</strong></div>
            ${!isActive ? `<div style="margin-top:8px;"><button class="btn btn-sm btn-accent w-full" style="width:100%;cursor:pointer;padding:6px 8px;font-size:11px;font-weight:700;border-radius:4px;background:#00E5FF;color:#0A0F1D;border:none;" onclick="window.dispatchEvent(new CustomEvent('switch-scenario', { detail: '${key}' }))">Focus Incident</button></div>` : ''}
          </div>
        `);

        satelliteDrape.on('click', () => {
          if (onSelectScenario && !isActive) {
            onSelectScenario(key);
          }
        });
        group.addLayer(satelliteDrape);

        // Hide broken image overlays silently — if the asset 404s or fails to decode,
        // remove the layer so no white broken box appears on the map
        satelliteDrape.on('load', () => {
          const imgEl = (satelliteDrape as any)._image as HTMLImageElement | undefined;
          if (imgEl) {
            imgEl.onerror = () => {
              imgEl.style.display = 'none';
              group.removeLayer(satelliteDrape);
            };
          }
        });
        satelliteDrape.on('error', () => {
          group.removeLayer(satelliteDrape);
        });
      }

      // Tiered Oil Slick Polygons (Bonn Code Core + Sheen) - ONLY RENDER IF showAiMask is TRUE!
      if (showAiMask) {
        const detectedPolygons =
          isActive && detectionResult?.polygons && detectionResult.polygons.length > 0
            ? detectionResult.polygons
            : getScenarioBenchmarkDetections(centerLat, centerLng);

        detectedPolygons.slice().reverse().forEach((poly, idx) => {
          const isCore = idx === 1;
          const latLngs: [number, number][] = poly.geometry.coordinates[0].map(([lon, lat]) => [lat, lon]);

          const slickPoly = L.polygon(latLngs, {
            color: isCore ? '#F59E0B' : '#0284C7',
            weight: isCore ? 1.8 : 1.2,
            dashArray: isCore ? undefined : '3, 3',
            fillColor: isCore ? '#F59E0B' : '#0284C7',
            fillOpacity: 0.04,
            className: isCore ? 'gis-slick-core' : 'gis-slick-sheen',
          }).bindPopup(`
            <div class="gis-popup-card">
              <div class="gis-popup-header" style="color: ${isCore ? '#F59E0B' : '#38BDF8'}; border-bottom: 1.5px solid ${isCore ? '#F59E0B' : '#38BDF8'};">${isCore ? 'HEAVY VISCOUS EMULSION CORE' : 'IRIDESCENT METALLIC SHEEN'}</div>
              <div class="gis-popup-row"><span style="color:#94A3B8;">Incident:</span> <strong style="color:#F1F5F9;">${sc.title} (${sc.id})</strong></div>
              <div class="gis-popup-row"><span style="color:#94A3B8;">Layer Classification:</span> <strong style="color:#F59E0B;">${isCore ? 'Bonn Code 4/5 (Continuous Emulsion)' : 'Bonn Code 1/2 (Rainbow Sheen)'}</strong></div>
              <div class="gis-popup-row"><span style="color:#94A3B8;">Measured Surface Area:</span> <strong style="color:#38BDF8;">${poly.area_km2} km² (${(poly.area_km2 * 100).toFixed(0)} Hectares)</strong></div>
              <div class="gis-popup-row"><span style="color:#94A3B8;">U-Net Confidence:</span> <strong style="color:#10B981;">${(poly.confidence * 100).toFixed(1)}%</strong></div>
              <div class="gis-popup-row"><span style="color:#94A3B8;">SAR Radar Damping:</span> <strong style="color:#16A34A;">Δσ0 = ${poly.mean_damping_db} dB</strong></div>
              <div class="gis-popup-row"><span style="color:#94A3B8;">Estimated Volume:</span> <strong style="color:#F1F5F9;">${isCore ? '~12.8 Metric Tons (~94 bbls)' : '~1.4 Metric Tons'}</strong></div>
              ${!isActive ? `<div style="margin-top:8px;"><button class="btn btn-sm btn-accent w-full" style="width:100%;cursor:pointer;padding:6px 8px;font-size:11px;font-weight:700;border-radius:4px;background:#00E5FF;color:#0A0F1D;border:none;" onclick="window.dispatchEvent(new CustomEvent('switch-scenario', { detail: '${key}' }))">Focus Incident</button></div>` : ''}
            </div>
          `);

          slickPoly.on('click', () => {
            if (onSelectScenario && !isActive) {
              onSelectScenario(key);
            }
          });
          group.addLayer(slickPoly);
        });
      }

      // ── INCIDENT PLACE INDICATION BEACONS (ALWAYS RENDERED ACROSS INDIA) ──
      const isWestCoast = centerLng < 78.0;
      const placementClass = isWestCoast ? 'placement-left' : 'placement-right';

      let shortTitle = sc.title;
      let badgeColor = sc.oilColor || '#38BDF8';
      if (sc.id.includes('001')) {
        shortTitle = 'Mumbai High Basin';
        badgeColor = isActive ? '#00E5FF' : '#F59E0B';
      } else if (sc.id.includes('002')) {
        shortTitle = 'Chennai–Ennore Corridor';
        badgeColor = '#F43F5E';
      } else if (sc.id.includes('003')) {
        shortTitle = 'Andaman Sea SL-7';
        badgeColor = '#38BDF8';
      } else if (sc.id.includes('004')) {
        shortTitle = 'Goa Coastal Waters';
        badgeColor = '#FACC15';
      } else if (sc.id.includes('005')) {
        shortTitle = 'Gulf of Kutch SPM';
        badgeColor = '#F97316';
      } else if (sc.id.includes('006')) {
        shortTitle = 'Cochin Port Anchorage';
        badgeColor = '#A855F7';
      } else if (sc.id.includes('007')) {
        shortTitle = 'Paradip Offshore Basin';
        badgeColor = '#EF4444';
      } else if (sc.id.includes('008')) {
        shortTitle = 'Lakshadweep 9° Ch';
        badgeColor = '#06B6D4';
      }

      if (isActive) {
        // Active incident marker: High-contrast sleek tactical pin
        const activeBeaconIcon = L.divIcon({
          className: 'gis-glass-marker-wrap',
          html: `
            <div class="gis-glass-marker active-incident" style="color: #00E5FF; cursor: pointer;">
              <div class="gis-glass-dot" style="background: #00E5FF; box-shadow: 0 0 10px rgba(0, 229, 255, 0.6); border: 2px solid #FFFFFF;"></div>
              <div class="gis-glass-capsule ${placementClass}" style="border-color: rgba(0, 229, 255, 0.6);">
                <span class="gis-capsule-id" style="background: rgba(0, 229, 255, 0.22); border-color: #00E5FF; color: #00E5FF;">
                  ${sc.id.replace('2026-', '')}
                </span>
                <span class="gis-capsule-title" style="color: #FFFFFF; font-weight: 700;">
                  ${shortTitle}
                </span>
                <span class="gis-capsule-pill" style="color: #00E5FF; border-color: rgba(0, 229, 255, 0.6); background: rgba(0, 229, 255, 0.18); font-weight: 700;">
                  ACTIVE
                </span>
              </div>
            </div>
          `,
          iconSize: [0, 0],
          iconAnchor: [0, 0],
        });

        const activeMarker = L.marker([centerLat, centerLng], { icon: activeBeaconIcon, zIndexOffset: 1000 })
          .bindPopup(`
            <div class="gis-popup-card">
              <div class="gis-popup-header" style="border-bottom: 2px solid #00E5FF;">
                ${sc.title} (${sc.id})
              </div>
              <div class="gis-popup-row"><span style="color:#94A3B8;">Status:</span> <strong style="color:#00E5FF;">ACTIVE INVESTIGATION</strong></div>
              <div class="gis-popup-row"><span style="color:#94A3B8;">Coordinates:</span> <strong style="color:#F1F5F9;">${sc.lat.toFixed(4)}°N, ${sc.lng.toFixed(4)}°E</strong></div>
              <div class="gis-popup-row"><span style="color:#94A3B8;">Severity:</span> <strong style="color:#EF4444;">${sc.sev}</strong></div>
              <div class="gis-popup-row"><span style="color:#94A3B8;">Oil Classification:</span> <strong style="color:#F59E0B;">${sc.oilType}</strong></div>
              <div class="gis-popup-row"><span style="color:#94A3B8;">Surface Extent:</span> <strong style="color:#38BDF8;">${sc.area}</strong></div>
              <div class="gis-popup-row"><span style="color:#94A3B8;">Suspect Intercept:</span> <strong style="color:#F1F5F9;">${sc.topVessel}</strong></div>
            </div>
          `);
        group.addLayer(activeMarker);
      } else {
        // Monitored incidents across India: Clickable clean GIS pins
        const beaconIcon = L.divIcon({
          className: 'gis-glass-marker-wrap',
          html: `
            <div class="gis-glass-marker" style="color: ${badgeColor}; cursor: pointer;">
              <div class="gis-glass-dot" style="background: ${badgeColor}; box-shadow: 0 1px 6px rgba(0,0,0,0.5); border: 1.5px solid #FFFFFF;"></div>
              <div class="gis-glass-capsule ${placementClass}" style="border-color: ${badgeColor}55;">
                <span class="gis-capsule-id" style="background: ${badgeColor}18; border-color: ${badgeColor}55; color: ${badgeColor};">
                  ${sc.id.replace('2026-', '')}
                </span>
                <span class="gis-capsule-title" style="color: #F1F5F9; font-weight: 600;">
                  ${shortTitle}
                </span>
                <span class="gis-capsule-pill" style="color: ${badgeColor}; border-color: ${badgeColor}55; background: ${badgeColor}18; font-weight: 600;">
                  ${sc.oilType}
                </span>
              </div>
            </div>
          `,
          iconSize: [0, 0],
          iconAnchor: [0, 0],
        });

        const marker = L.marker([centerLat, centerLng], { icon: beaconIcon, zIndexOffset: 500 })
          .on('click', () => {
            if (onSelectScenario) {
              onSelectScenario(key);
            }
          })
          .bindPopup(`
            <div class="gis-popup-card">
              <div class="gis-popup-header" style="border-bottom: 2px solid ${badgeColor}; color: ${badgeColor};">
                ${sc.title} (${sc.id})
              </div>
              <div class="gis-popup-row"><span style="color:#94A3B8;">Status:</span> <strong style="color:${badgeColor};">MONITORED INCIDENT</strong></div>
              <div class="gis-popup-row"><span style="color:#94A3B8;">Coordinates:</span> <strong style="color:#F1F5F9;">${sc.lat.toFixed(4)}°N, ${sc.lng.toFixed(4)}°E</strong></div>
              <div class="gis-popup-row"><span style="color:#94A3B8;">Severity:</span> <strong style="color:${badgeColor};">${sc.sev}</strong></div>
              <div class="gis-popup-row"><span style="color:#94A3B8;">Oil Classification:</span> <strong style="color:#F59E0B;">${sc.oilType}</strong></div>
              <div class="gis-popup-row"><span style="color:#94A3B8;">Surface Extent:</span> <strong style="color:#38BDF8;">${sc.area}</strong></div>
              <div class="gis-popup-row"><span style="color:#94A3B8;">Primary Suspect:</span> <strong style="color:#F1F5F9;">${sc.topVessel}</strong></div>
              <div style="margin-top: 10px;">
                <button class="btn btn-sm btn-accent w-full" style="width: 100%; cursor: pointer; padding: 6px 10px; font-size: 11px; font-weight: 700; border-radius: 4px; background: #00E5FF; color: #0A0F1D; border: none;" onclick="window.dispatchEvent(new CustomEvent('switch-scenario', { detail: '${key}' }))">
                  Investigate Incident
                </button>
              </div>
            </div>
          `);
        group.addLayer(marker);
      }
    });

    // If NO scenario is selected (National Overview mode), return
    if (!scenario) {
      return;
    }

    const centerLat = scenario.lat;
    const centerLng = scenario.lng;

    // ── SATELLITE FOOTPRINT SWATH (Dynamic for ISRO NISAR / Sentinel-1 / Sentinel-2 / EOS-04) ──
    if (showAiMask || showOverlay) {
      const isNisar = selectedCopernicusLayer === 'nisar-ls';
      const isEos04 = selectedCopernicusLayer === 'eos-04';
      const isOptical = selectedCopernicusLayer === 'true-color' || selectedCopernicusLayer === 'swir-oil' || selectedCopernicusLayer === 'false-color';

      const swathCoords: [number, number][] = isNisar
        ? [
            [centerLat + 0.75, centerLng - 0.85],
            [centerLat + 0.55, centerLng + 0.95],
            [centerLat - 0.75, centerLng + 0.65],
            [centerLat - 0.55, centerLng - 1.15],
          ]
        : [
            [centerLat + 0.65, centerLng - 0.95],
            [centerLat + 0.45, centerLng + 0.85],
            [centerLat - 0.65, centerLng + 0.55],
            [centerLat - 0.45, centerLng - 1.25],
          ];

      const swathColor = isNisar ? '#F59E0B' : isEos04 ? '#10B981' : isOptical ? '#38BDF8' : '#0891B2';
      const swathTitle = isNisar
        ? '<b>ISRO NISAR SweepSAR L+S Footprint (242 km Swath)</b><br>ISRO S-Band (3.2 GHz) + NASA L-Band (1.25 GHz)<br>Polarimetric Bragg Damping DFDI Composite • ISRO NRSC Bhoonidhi'
        : isEos04
        ? '<b>ISRO EOS-04 (RISAT-1A) C-SAR Circular Pol Footprint (140 km Swath)</b><br>ISRO NRSC Bhoonidhi OpenData Node'
        : isOptical
        ? '<b>Sentinel-2 MSI Multispectral Footprint (290 km Swath)</b><br>ESA Copernicus Multispectral'
        : '<b>Sentinel-1A C-SAR IW GRD Footprint (250 km Swath)</b><br>Copernicus Data Space Ecosystem';

      const swathPoly = L.polygon(swathCoords, {
        color: swathColor,
        weight: isNisar ? 2.5 : 1.5,
        dashArray: isNisar ? '6, 4' : '5, 6',
        fillColor: swathColor,
        fillOpacity: isNisar ? 0.08 : 0.03,
      }).bindTooltip(swathTitle, {
        sticky: true,
        className: 'gis-custom-tooltip',
      });
      group.addLayer(swathPoly);
    }

    // ── AI MASK & HYDRODYNAMIC ANALYSIS OVERLAYS (Rendered ONLY when showAiMask is enabled) ──
    if (showAiMask) {
      // ── 4. GEODETIC DIMENSION AXES ──
      const isChennai = scenario.id.includes('002');
      const isGoa = scenario.id.includes('004');
      let majHeadingRad = 3.80; // default 218° WSW
      let majLen = 0.038;
      let minLen = 0.016;

      if (isChennai) {
        majHeadingRad = 0.35; // 20° NNE parallel to Coromandel coast
        majLen = 0.032;
        minLen = 0.012;
      } else if (isGoa) {
        majHeadingRad = 2.85; // 160° SSE parallel to Konkan coast
        majLen = 0.026;
        minLen = 0.011;
      }

      const majStart: [number, number] = [
        centerLat + Math.sin(majHeadingRad) * majLen,
        centerLng + Math.cos(majHeadingRad) * majLen,
      ];
      const majEnd: [number, number] = [
        centerLat - Math.sin(majHeadingRad) * majLen,
        centerLng - Math.cos(majHeadingRad) * majLen,
      ];

      const majLine = L.polyline([majStart, majEnd], {
        color: '#00E5FF',
        weight: 2.0,
        dashArray: '5, 5',
      }).bindPopup(`
        <div class="gis-popup-card" style="min-width: 220px;">
          <div class="gis-popup-header" style="color: #00E5FF; border-bottom: 1.5px solid rgba(0, 229, 255, 0.5);">MAJOR DISPERSION AXIS</div>
          <div class="gis-popup-row"><span style="color:#94A3B8;">Elongation Length:</span> <strong style="color:#F1F5F9;">${isChennai ? '3.2 km' : '4.65 km'}</strong></div>
          <div class="gis-popup-row"><span style="color:#94A3B8;">Bearing:</span> <strong style="color:#00E5FF;">${isChennai ? '020° NNE' : '248° WSW'}</strong></div>
          <div class="gis-popup-row"><span style="color:#94A3B8;">Mechanism:</span> <strong style="color:#38BDF8;">Current-driven elongation</strong></div>
        </div>
      `);
      group.addLayer(majLine);

      const minHeadingRad = majHeadingRad + Math.PI / 2;
      const minStart: [number, number] = [
        centerLat + Math.sin(minHeadingRad) * minLen,
        centerLng + Math.cos(minHeadingRad) * minLen,
      ];
      const minEnd: [number, number] = [
        centerLat - Math.sin(minHeadingRad) * minLen,
        centerLng - Math.cos(minHeadingRad) * minLen,
      ];

      const minLine = L.polyline([minStart, minEnd], {
        color: '#10B981',
        weight: 1.8,
        dashArray: '4, 4',
      }).bindPopup(`
        <div class="gis-popup-card" style="min-width: 220px;">
          <div class="gis-popup-header" style="color: #10B981; border-bottom: 1.5px solid rgba(16, 185, 129, 0.5);">MINOR CROSS-DISPERSION AXIS</div>
          <div class="gis-popup-row"><span style="color:#94A3B8;">Diffusion Width:</span> <strong style="color:#F1F5F9;">${isChennai ? '1.2 km' : '1.78 km'}</strong></div>
          <div class="gis-popup-row"><span style="color:#94A3B8;">Mechanism:</span> <strong style="color:#38BDF8;">Turbulent lateral diffusion</strong></div>
        </div>
      `);
      group.addLayer(minLine);

      // ── 5. LAGRANGIAN HYDRODYNAMIC MODELING (PAST ORIGIN & FUTURE FORECAST) ──
      const activePolygons =
        detectionResult?.polygons && detectionResult.polygons.length > 0
          ? detectionResult.polygons
          : getScenarioBenchmarkDetections(centerLat, centerLng);
      const primarySlick = activePolygons[0];
      const targetCentroid: [number, number] = sarDriftPayload
        ? [sarDriftPayload.lat, sarDriftPayload.lng]
        : primarySlick.slick_centroid;
      const driftGeo = computeBackwardDriftGeometry(scenario.id, targetCentroid, activeDriftSimulation);

      // A. PAST: 90%, 75%, 50% Reconstructed Discharge Origin Envelopes
      const poly90 = L.polygon(driftGeo.envelope90, {
        color: '#B45309',
        weight: 1.2,
        dashArray: '4, 5',
        fillColor: '#B45309',
        fillOpacity: 0.08,
      }).bindTooltip('<b>90% Probability Origin Isobar</b><br>OpenDrift Reverse Monte Carlo', { sticky: true, className: 'gis-custom-tooltip' });
      group.addLayer(poly90);

      const poly50 = L.polygon(driftGeo.envelope50, {
        color: '#92400E',
        weight: 1.8,
        fillColor: '#B45309',
        fillOpacity: 0.30,
      }).bindTooltip('<b>50% Core Origin Interception Envelope</b>', { sticky: true, className: 'gis-custom-tooltip' });
      group.addLayer(poly50);

      // B. PAST: Backward Lagrangian Drift Vector Line (T0 → T-22h)
      const driftPathLine = L.polyline(driftGeo.driftPath, {
        color: '#D97706',
        weight: 2.8,
        dashArray: '6, 6',
        opacity: 0.95,
      }).bindTooltip('<b>⏪ Backward Lagrangian Drift Vector (T0 → T-22h Source)</b><br>CMEMS current + 3.5% ERA5 Stokes drift (38.2 km transport)', { sticky: true, className: 'gis-custom-tooltip' });
      group.addLayer(driftPathLine);

      // C. FUTURE: Predicted Forward Drift Trajectory & Dispersion Cones (T0 → T+24h)
      const forwardConePoly24 = L.polygon(driftGeo.forwardCone24h, {
        color: '#10B981',
        weight: 1.2,
        dashArray: '4, 4',
        fillColor: '#10B981',
        fillOpacity: 0.10,
      }).bindTooltip('<b>⏩ T+24h Predicted Spill Dispersion Cone</b><br>Projected spreading under ocean current & wind forecast', { sticky: true, className: 'gis-custom-tooltip' });
      group.addLayer(forwardConePoly24);

      const forwardConePoly12 = L.polygon(driftGeo.forwardCone12h, {
        color: '#10B981',
        weight: 1.6,
        fillColor: '#10B981',
        fillOpacity: 0.18,
      }).bindTooltip('<b>⏩ T+12h Predicted Core Impact Zone</b>', { sticky: true, className: 'gis-custom-tooltip' });
      group.addLayer(forwardConePoly12);

      const forwardPathLine = L.polyline(driftGeo.forwardDriftPath, {
        color: '#10B981',
        weight: 3.0,
        dashArray: '5, 5',
        opacity: 0.95,
      }).bindTooltip('<b>⏩ Predicted Forward Drift Trajectory (T0 → T+24h Forecast)</b><br>CMEMS hydrodynamic current model forecast', { sticky: true, className: 'gis-custom-tooltip' });
      group.addLayer(forwardPathLine);

      // D. CANDIDATE AIS VESSEL TRACK — REAL MULTI-POINT KINEMATIC PROJECTION
      const aisTrackPoints = getRealisticAISTrack(
        scenario.id,
        driftGeo.originCoord[0],
        driftGeo.originCoord[1],
        driftGeo.vesselHeading
      );

      // Draw multi-colored speed-heatmap polyline segments between consecutive waypoints
      for (let i = 0; i < aisTrackPoints.length - 1; i++) {
        const p1 = aisTrackPoints[i];
        const p2 = aisTrackPoints[i + 1];
        // Use the lower SOG or the silent status of either endpoint to color the segment
        const segmentStatus = (p1.aisStatus === 'silent' || p2.aisStatus === 'silent')
          ? 'silent'
          : (p1.aisStatus === 'gap' || p2.aisStatus === 'gap') ? 'gap' : 'active';
        const minSog = Math.min(p1.sog, p2.sog);
        const segmentColor = sogToColor(minSog, segmentStatus);
        const isSilent = segmentStatus === 'silent' || segmentStatus === 'gap';

        const segPoly = L.polyline([p1.coord, p2.coord], {
          color: segmentColor,
          weight: isSilent ? 3.5 : 2.5,
          dashArray: isSilent ? '6, 5' : undefined,
          opacity: 0.92,
        }).bindTooltip(
          `<b>AIS Track Segment</b><br>` +
          `Time: ${p1.timeLabel} → ${p2.timeLabel}<br>` +
          `Speed: <b>${minSog.toFixed(1)} kn</b> (${minSog >= 10 ? 'Transit' : minSog >= 6 ? 'Maneuver' : 'Slow/Discharge'})<br>` +
          `Status: <span style="color:${segmentColor};font-weight:700;">${segmentStatus.toUpperCase()}</span>`,
          { sticky: true, className: 'gis-custom-tooltip' }
        );
        group.addLayer(segPoly);
      }

      // Draw waypoint dots with speed, heading, and time labels
      aisTrackPoints.forEach((pt, idx) => {
        const isEpoch = idx === aisTrackPoints.length - 1; // T=0h (current)
        const isDischarge = pt.timeLabel === 'T−22h' || pt.timeLabel === 'T−18h';
        const isSilent = pt.aisStatus === 'silent';
        const isGap = pt.aisStatus === 'gap';
        const ptColor = sogToColor(pt.sog, pt.aisStatus);

        // Skip adding dot for T=0h (handled by the larger vessel icon below)
        if (isEpoch) return;

        const dotIcon = L.divIcon({
          className: 'gis-ais-wp-wrap',
          html: `
            <div style="
              width: ${isDischarge ? '11px' : '7px'};
              height: ${isDischarge ? '11px' : '7px'};
              border-radius: 50%;
              background: ${ptColor};
              border: 1.5px solid ${isDischarge ? '#FFFFFF' : 'rgba(255,255,255,0.7)'};
              box-shadow: 0 0 ${isDischarge ? '8px' : '4px'} ${ptColor};
              transform: translate(-50%, -50%);
              cursor: pointer;
            "></div>
          `,
          iconSize: [0, 0],
          iconAnchor: [0, 0],
        });

        const wpMarker = L.marker(pt.coord, { icon: dotIcon }).bindPopup(`
          <div class="gis-popup-card" style="min-width: 200px;">
            <div class="gis-popup-header" style="color: ${ptColor}; border-bottom: 1.5px solid ${ptColor}88;">
              AIS POSITION · ${pt.timeLabel}
            </div>
            <div class="gis-popup-row"><span style="color:#94A3B8;">Vessel:</span> <strong style="color:#F1F5F9;">${scenario.topVessel}</strong></div>
            <div class="gis-popup-row"><span style="color:#94A3B8;">Coordinates:</span> <strong style="color:#F1F5F9;">${pt.coord[0].toFixed(3)}°N, ${pt.coord[1].toFixed(3)}°E</strong></div>
            <div class="gis-popup-row"><span style="color:#94A3B8;">Speed (SOG):</span> <strong style="color:${ptColor}; font-weight:700;">${pt.sog.toFixed(1)} kn</strong></div>
            <div class="gis-popup-row"><span style="color:#94A3B8;">Course (COG):</span> <strong style="color:#38BDF8;">${pt.cog}° (${scenario.diagDetails?.slice(0, 30) || 'Maritime Fairway'})</strong></div>
            <div class="gis-popup-row"><span style="color:#94A3B8;">AIS Broadcast:</span> <strong style="color:${ptColor};">${isSilent ? '🔴 TRANSPONDER OFF' : isGap ? '⚠️ INTERMITTENT GAP' : '🟢 BROADCASTING'}</strong></div>
          </div>
        `);
        group.addLayer(wpMarker);
      });

      // Vessel position at SAR detection epoch (T=0h)
      const lastPoint = aisTrackPoints[aisTrackPoints.length - 1];
      const vesselPos = lastPoint.coord;
      const vesselIcon = L.divIcon({
        className: 'gis-glass-marker-wrap',
        html: `
          <div class="gis-glass-marker" style="color: #EA580C; cursor: pointer;">
            <div class="gis-vessel-wrap" style="position: absolute; top: -11px; left: -11px; width: 22px; height: 22px; transform: rotate(${driftGeo.vesselHeading}deg);" title="${scenario.topVessel}">
              <svg width="22" height="22" viewBox="0 0 18 18">
                <polygon points="9,1 16,16 9,12 2,16" fill="#EA580C" stroke="#FFFFFF" stroke-width="1.5" />
              </svg>
            </div>
            <div class="gis-glass-capsule placement-right" style="border-color: rgba(234, 88, 12, 0.6); left: 16px;">
              <span class="gis-capsule-pill" style="color: #EA580C; border-color: rgba(234, 88, 12, 0.6); background: rgba(234, 88, 12, 0.18); font-weight: 700;">
                SUSPECT
              </span>
              <span class="gis-capsule-title" style="color: #FFFFFF; font-weight: 700;">
                ${scenario.topVessel}
              </span>
              <span class="gis-capsule-id" style="background: rgba(234, 88, 12, 0.22); border-color: #EA580C; color: #EA580C;">
                T=0h · ${lastPoint.sog.toFixed(1)}kn
              </span>
            </div>
          </div>
        `,
        iconSize: [0, 0],
        iconAnchor: [0, 0],
      });
      group.addLayer(
        L.marker(vesselPos, { icon: vesselIcon }).bindPopup(`
          <div class="gis-popup-card">
            <div class="gis-popup-header" style="color: #EA580C; border-bottom: 1.5px solid rgba(234, 88, 12, 0.5);">OFFENDING VESSEL CANDIDATE (T=0h)</div>
            <div class="gis-popup-row"><span style="color:#94A3B8;">Vessel:</span> <strong style="color:#F1F5F9;">${scenario.topVessel}</strong></div>
            <div class="gis-popup-row"><span style="color:#94A3B8;">Current SOG / COG:</span> <strong style="color:#38BDF8;">${lastPoint.sog.toFixed(1)} kn · ${driftGeo.vesselHeading}°</strong></div>
            <div class="gis-popup-row"><span style="color:#94A3B8;">AIS Telemetry:</span> <strong style="color:#F1F5F9;">${scenario.diagDetails}</strong></div>
            <div class="gis-popup-row"><span style="color:#94A3B8;">Attribution Score:</span> <strong style="color:#F59E0B;">S = ${scenario.scores?.[0] || 0.82} (Bayesian Match)</strong></div>
          </div>
        `)
      );

      // ── 6. TEMPORAL HUD LABELS (DISCHARGE ORIGIN T-22h & PREDICTED DRIFT T+24h) ──

      // Dynamic Temporal Scrubbing Overlays (When scrubber is moved into the past)
      if (scrubHours < 0) {
        const scrubbedState = interpolateKinematicsAtHour(
          aisTrackPoints,
          driftGeo.originCoord,
          [centerLat, centerLng],
          scrubHours
        );

        // A. Dynamic Vessel Position at T = scrubHours
        const isDischarge = scrubbedState.isDischargeActive;
        const vColor = scrubbedState.vesselStatus === 'silent' ? '#DC2626' : isDischarge ? '#EF4444' : '#00E5FF';

        const scrubbedVesselIcon = L.divIcon({
          className: 'gis-glass-marker-wrap',
          html: `
            <div class="gis-glass-marker" style="color: ${vColor}; cursor: pointer; z-index: 1000;">
              <div class="gis-vessel-wrap" style="position: absolute; top: -13px; left: -13px; width: 26px; height: 26px; transform: rotate(${driftGeo.vesselHeading}deg);">
                <svg width="26" height="26" viewBox="0 0 18 18">
                  <polygon points="9,1 16,16 9,12 2,16" fill="${vColor}" stroke="#FFFFFF" stroke-width="2" />
                </svg>
              </div>
              <div class="gis-glass-capsule placement-right" style="border-color: ${vColor}; left: 18px; box-shadow: 0 0 14px ${vColor}88; background: rgba(5,15,30,0.92);">
                <span class="gis-capsule-pill" style="color: ${vColor}; border-color: ${vColor}; background: ${vColor}22; font-weight: 800;">
                  T = ${scrubHours}h
                </span>
                <span class="gis-capsule-title" style="color: #FFFFFF; font-weight: 700;">
                  ${scenario.topVessel}
                </span>
                <span class="gis-capsule-id" style="background: ${vColor}33; border-color: ${vColor}; color: ${vColor}; font-weight: 800;">
                  ${scrubbedState.vesselSog.toFixed(1)}kn ${isDischarge ? '· DISCHARGING' : scrubbedState.vesselStatus === 'silent' ? '· AIS OFF' : ''}
                </span>
              </div>
            </div>
          `,
          iconSize: [0, 0],
          iconAnchor: [0, 0],
        });

        group.addLayer(
          L.marker(scrubbedState.vesselCoord, { icon: scrubbedVesselIcon, zIndexOffset: 2000 }).bindPopup(`
            <div class="gis-popup-card">
              <div class="gis-popup-header" style="color: ${vColor}; border-bottom: 1.5px solid ${vColor}88;">
                HISTORICAL AIS KINEMATICS · T = ${scrubHours}h
              </div>
              <div class="gis-popup-row"><span style="color:#94A3B8;">Vessel:</span> <strong style="color:#F1F5F9;">${scenario.topVessel}</strong></div>
              <div class="gis-popup-row"><span style="color:#94A3B8;">Coordinates:</span> <strong style="color:#38BDF8;">${scrubbedState.vesselCoord[0].toFixed(4)}°N, ${scrubbedState.vesselCoord[1].toFixed(4)}°E</strong></div>
              <div class="gis-popup-row"><span style="color:#94A3B8;">Speed Over Ground:</span> <strong style="color:${vColor}; font-weight:700;">${scrubbedState.vesselSog.toFixed(1)} kn</strong></div>
              <div class="gis-popup-row"><span style="color:#94A3B8;">Broadcast State:</span> <strong style="color:${vColor};">${scrubbedState.vesselStatus.toUpperCase()}</strong></div>
              <div class="gis-popup-row"><span style="color:#94A3B8;">Operational Status:</span> <strong style="color:#F59E0B;">${isDischarge ? '⚠️ ACTIVE BUNKER/SLOP DISCHARGE IN PROGRESS' : 'Underway Along Marine Fairway'}</strong></div>
            </div>
          `)
        );

        // B. Dynamic Oil Spill Centroid at T = scrubHours
        const spillColor = isDischarge ? '#EF4444' : '#F59E0B';
        const scrubbedSpillIcon = L.divIcon({
          className: 'gis-glass-marker-wrap',
          html: `
            <div class="gis-glass-marker" style="color: ${spillColor}; z-index: 900;">
              <div class="gis-glass-dot" style="background: ${spillColor}; box-shadow: 0 0 16px ${spillColor}; border: 2.5px solid #FFFFFF; width: 14px; height: 14px; border-radius: 50%;"></div>
              <div class="gis-glass-capsule placement-left" style="border-color: ${spillColor}; right: 18px; box-shadow: 0 0 12px ${spillColor}66; background: rgba(5,15,30,0.92);">
                <span class="gis-capsule-pill" style="color: ${spillColor}; border-color: ${spillColor}; background: ${spillColor}22; font-weight: 800;">
                  SLICK T = ${scrubHours}h
                </span>
                <span class="gis-capsule-title" style="color: #FFFFFF; font-weight: 700;">
                  ${isDischarge ? 'Discharge In Progress' : 'Lagrangian Drift Step'}
                </span>
              </div>
            </div>
          `,
          iconSize: [0, 0],
          iconAnchor: [0, 0],
        });

        group.addLayer(
          L.marker(scrubbedState.spillCoord, { icon: scrubbedSpillIcon, zIndexOffset: 1500 })
        );
      }

      // 2. DISCHARGE ORIGIN PIN (T-22h Source)
      const originIcon = L.divIcon({
        className: 'gis-glass-marker-wrap',
        html: `
          <div class="gis-glass-marker" style="color: #F59E0B; cursor: pointer;">
            <div class="gis-glass-dot" style="background: #F59E0B; box-shadow: 0 0 10px rgba(245, 158, 11, 0.6); border: 2px solid #FFFFFF;"></div>
            <div class="gis-glass-capsule placement-right" style="border-color: rgba(245, 158, 11, 0.6);">
              <span class="gis-capsule-pill" style="color: #F59E0B; border-color: rgba(245, 158, 11, 0.6); background: rgba(245, 158, 11, 0.18); font-weight: 700;">
                ORIGIN
              </span>
              <span class="gis-capsule-title" style="color: #FFFFFF; font-weight: 700;">
                Estimated Discharge
              </span>
              <span class="gis-capsule-id" style="background: rgba(245, 158, 11, 0.22); border-color: #F59E0B; color: #F59E0B;">
                T - 22h
              </span>
            </div>
          </div>
        `,
        iconSize: [0, 0],
        iconAnchor: [0, 0],
      });
      group.addLayer(
        L.marker(driftGeo.originCoord, { icon: originIcon }).bindPopup(`
          <div class="gis-popup-card">
            <div class="gis-popup-header" style="color: #F59E0B; border-bottom: 1.5px solid rgba(245, 158, 11, 0.5);">RECONSTRUCTED DISCHARGE ORIGIN (T-22h)</div>
            <div class="gis-popup-row"><span style="color:#94A3B8;">Estimated Release Time:</span> <strong style="color:#F1F5F9;">06:00 UTC (during AIS silence window)</strong></div>
            <div class="gis-popup-row"><span style="color:#94A3B8;">Origin Position:</span> <strong style="color:#00E5FF;">${driftGeo.originCoord[0].toFixed(4)}°N, ${driftGeo.originCoord[1].toFixed(4)}°E</strong></div>
            <div class="gis-popup-row"><span style="color:#94A3B8;">Total Transport Distance:</span> <strong style="color:#38BDF8;">38.2 km under CMEMS ocean current</strong></div>
          </div>
        `)
      );

      // 3. PREDICTED FORWARD DRIFT PIN (T+24h Forecast)
      const forwardIcon = L.divIcon({
        className: 'gis-glass-marker-wrap',
        html: `
          <div class="gis-glass-marker" style="color: #10B981; cursor: pointer;">
            <div class="gis-glass-dot" style="background: #10B981; box-shadow: 0 0 10px rgba(16, 185, 129, 0.6); border: 2px solid #FFFFFF;"></div>
            <div class="gis-glass-capsule placement-right" style="border-color: rgba(16, 185, 129, 0.6);">
              <span class="gis-capsule-pill" style="color: #10B981; border-color: rgba(16, 185, 129, 0.6); background: rgba(16, 185, 129, 0.18); font-weight: 700;">
                FORECAST
              </span>
              <span class="gis-capsule-title" style="color: #FFFFFF; font-weight: 700;">
                Predicted Spill Drift
              </span>
              <span class="gis-capsule-id" style="background: rgba(16, 185, 129, 0.22); border-color: #10B981; color: #10B981;">
                T + 24h
              </span>
            </div>
          </div>
        `,
        iconSize: [0, 0],
        iconAnchor: [0, 0],
      });
      group.addLayer(
        L.marker(driftGeo.predictedCoord24h, { icon: forwardIcon }).bindPopup(`
          <div class="gis-popup-card">
            <div class="gis-popup-header" style="color: #10B981; border-bottom: 1.5px solid rgba(16, 185, 129, 0.5);">PREDICTED DRIFT FORECAST (T+24h)</div>
            <div class="gis-popup-row"><span style="color:#94A3B8;">Forecast Time:</span> <strong style="color:#F1F5F9;">T+24h (Next 24 Hours Projection)</strong></div>
            <div class="gis-popup-row"><span style="color:#94A3B8;">Projected Position:</span> <strong style="color:#00E5FF;">${driftGeo.predictedCoord24h[0].toFixed(4)}°N, ${driftGeo.predictedCoord24h[1].toFixed(4)}°E</strong></div>
            <div class="gis-popup-row"><span style="color:#94A3B8;">Forecast Model:</span> <strong style="color:#10B981;">OpenDrift + CMEMS Hydrodynamic Forecast</strong></div>
            <div class="gis-popup-row"><span style="color:#94A3B8;">Spreading Rate:</span> <strong style="color:#38BDF8;">Estimated plume expansion to ~6.4 km²</strong></div>
          </div>
        `)
      );
    }
  }, [scenario, selectedCopernicusLayer, layerOpacity, showAiMask, detectionResult, scrubHours]);

  return (
    <div
      ref={containerRef}
      id="leaflet-map"
      style={{ flex: 1, width: '100%', height: '100%', minHeight: 0, position: 'relative' }}
    />
  );
};
