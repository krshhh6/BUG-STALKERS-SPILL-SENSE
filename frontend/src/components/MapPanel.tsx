import React, { useState, useEffect, useRef } from 'react';
import type { Scenario, BaseLayerType, CopernicusLayerId, CopernicusLayerInfo, DetectionResult, SarDriftPayload } from '../types/dashboard';
import type { LiveSimulationResult } from '../services/marineWeatherService';
import { LeafletMap } from './LeafletMap';
import { TimeScrubber } from './TimeScrubber';
import { CleanupModal } from './modals/CleanupModal';
import { runSentinel1Detection } from '../services/detectionService';
import { createAoiForScenario } from '../utils/geoContours';
import { SCENARIOS } from '../data/scenarios';
import { socket, connectSocket } from '../services/socket';
import L from 'leaflet';

interface MapPanelProps {
  scenario: Scenario | null;
  onUpdateCoords: (coords: string) => void;
  onSelectScenario?: (key: string) => void;
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

const COPERNICUS_LAYERS: CopernicusLayerInfo[] = [
  {
    id: 'sar-vv',
    name: 'SAR Decibel (VV Damping)',
    bands: 'Sentinel-1 C-SAR IW GRD',
    desc: 'Capillary wave depression (Δσ0 ≤ -8.4 dB). High-contrast oil slick boundary mask.',
    badge: 'RADAR P0',
    badgeColor: '#0891B2',
    thumbBg: '#0f2b38',
  },
  {
    id: 'sar-vh',
    name: 'SAR Cross-Pol (VH Mode)',
    bands: 'Sentinel-1C C-SAR IW GRD',
    desc: 'Cross-polarized volume scattering and metallic vessel discrimination.',
    badge: 'RADAR VH',
    badgeColor: '#0EA5E9',
    thumbBg: '#0f2738',
  },
  {
    id: 'true-color',
    name: 'True color',
    bands: 'Based on bands B4, B3, B2',
    desc: 'Photorealistic natural ocean satellite imagery (Copernicus Browser standard).',
    badge: 'OPTICAL',
    badgeColor: '#0284C7',
    thumbBg: '#1e3a5f',
  },
  {
    id: 'swir-oil',
    name: 'Highlight Optimized (SWIR)',
    bands: 'SWIR B11 (1610nm), B12 (2190nm)',
    desc: 'Shortwave infrared absorption overtones. Highlights weathered crude emulsions.',
    badge: 'HYDROCARBON',
    badgeColor: '#B45309',
    thumbBg: '#3d2508',
  },

  {
    id: 'false-color',
    name: 'False color (CIR)',
    bands: 'Based on bands B8, B4, B3',
    desc: 'Near-Infrared CIR. Distinguishes biogenic Sargassum/algae blooms from mineral oil.',
    badge: 'ALGAE REJECT',
    badgeColor: '#DC2626',
    thumbBg: '#3b1212',
  },
  {
    id: 'ndwi',
    name: 'NDWI (Water Mask)',
    bands: 'Based on combination (B3 - B8)/(B3 + B8)',
    desc: 'Normalized Difference Water Index. Sharp boundary between open water and petroleum.',
    badge: 'WATER MASK',
    badgeColor: '#2563EB',
    thumbBg: '#14254b',
  },
  {
    id: 'thermal',
    name: 'Sea Surface Thermal Anomaly',
    bands: 'Copernicus SLSTR 1km Thermal IR',
    desc: 'Sea surface temperature gradient caused by evaporative cooling of volatile fractions.',
    badge: 'THERMAL',
    badgeColor: '#7C3AED',
    thumbBg: '#2a144b',
  },
  {
    id: 'nisar-ls',
    name: 'ISRO NISAR L+S Dual-Band (DFDI)',
    bands: 'S-band (3.2GHz) + L-band (1.26GHz)',
    desc: 'Dual-frequency Bragg scattering damping index. Eliminates biogenic algal false positives.',
    badge: 'ISRO DUAL-BAND',
    badgeColor: '#F59E0B',
    thumbBg: '#3d2508',
  },
  {
    id: 'eos-04',
    name: 'ISRO EOS-04 Hybrid Pol (CP)',
    bands: 'C-Band Circular Polarization (RH/RV)',
    desc: 'm-chi polarimetric decomposition distinguishing mineral crude from look-alikes.',
    badge: 'ISRO HYBRID',
    badgeColor: '#10B981',
    thumbBg: '#0f382a',
  },
];

export const MapPanel: React.FC<MapPanelProps> = ({
  scenario,
  onUpdateCoords,
  onSelectScenario,
  scenarios,
  targetLocation,
  sarDriftPayload,
  activeDriftSimulation,
}) => {
  const [baseLayer2D, setBaseLayer2D] = useState<BaseLayerType>('arcgis');
  const showSeamarks = false;
  const [selectedCopernicusLayer, setSelectedCopernicusLayer] = useState<CopernicusLayerId>('nisar-ls');
  const [activeSatellite, setActiveSatellite] = useState<string>('ISRO NISAR');
  const [showSpillOverlay, setShowSpillOverlay] = useState<boolean>(true);
  const [showIndiaOutline, setShowIndiaOutline] = useState<boolean>(true);
  const [showEezBoundary, setShowEezBoundary] = useState<boolean>(true);
  const [satelliteToast, setSatelliteToast] = useState<string | null>(null);
  const [isSideLayersOpen, setIsSideLayersOpen] = useState<boolean>(false);

  const selectedMission =
    activeSatellite === 'Sentinel-1A' ? 'Sentinel-1A C-SAR (VV)' :
    activeSatellite === 'Sentinel-1C' ? 'Sentinel-1C C-SAR (VH)' :
    activeSatellite === 'Sentinel-2A' ? 'Sentinel-2A MSI (Optical)' :
    activeSatellite === 'Sentinel-2B' ? 'Sentinel-2B MSI (SWIR)' :
    activeSatellite === 'ISRO NISAR' ? 'ISRO NISAR (L+S)' :
    activeSatellite.includes('EOS-04') ? 'ISRO EOS-04 (Hybrid)' :
    'Sentinel-1A C-SAR (VV)';

  const [layerOpacity, setLayerOpacity] = useState<number>(0.92);
  const [showAiMask, setShowAiMask] = useState<boolean>(true);
  const [isCleanupModalOpen, setIsCleanupModalOpen] = useState<boolean>(false);
  const [showIncidentDropdownMenu, setShowIncidentDropdownMenu] = useState<boolean>(false);
  const pendingCleanupRef = useRef<boolean>(false);
  const prevScenarioRef = useRef<Scenario | null>(scenario);
  const cleanupButtonContainerRef = useRef<HTMLDivElement | null>(null);

  const activeScenarios = scenarios && Object.keys(scenarios).length > 0 ? scenarios : SCENARIOS;

  // Auto-close cleanup modal if scenario becomes null (e.g. user returns to national overview)
  useEffect(() => {
    if (!scenario) {
      setIsCleanupModalOpen(false);
    }
  }, [scenario]);

  // When scenario changes, if user clicked Clean-Up while none was selected, open the modal for this selected scenario
  useEffect(() => {
    if (!prevScenarioRef.current && scenario && pendingCleanupRef.current) {
      pendingCleanupRef.current = false;
      setIsCleanupModalOpen(true);
      setShowIncidentDropdownMenu(false);
      const pill = document.querySelector('.scenario-selector-pill');
      pill?.classList.remove('highlight-cleanup-prompt');
    }
    prevScenarioRef.current = scenario;
  }, [scenario]);

  // Close fallback dropdown menu when clicking outside
  useEffect(() => {
    const handleOutsideClick = (e: MouseEvent) => {
      if (cleanupButtonContainerRef.current && !cleanupButtonContainerRef.current.contains(e.target as Node)) {
        setShowIncidentDropdownMenu(false);
      }
    };
    if (showIncidentDropdownMenu) {
      document.addEventListener('mousedown', handleOutsideClick);
    }
    return () => document.removeEventListener('mousedown', handleOutsideClick);
  }, [showIncidentDropdownMenu]);

  const [scrubHours, setScrubHours] = useState<number>(0);
  const [showRawImageModal, setShowRawImageModal] = useState<boolean>(false);
  const [modalZoom, setModalZoom] = useState<number>(1.0);
  const [selectedDate, setSelectedDate] = useState<string>(() => new Date().toISOString().split('T')[0]);
  const [cloudCoverMax] = useState<number>(30);
  const [detectionResult, setDetectionResult] = useState<DetectionResult | null>(null);
  const [driftEnvelopes, setDriftEnvelopes] = useState<any>(null);
  const [, setIsDetecting] = useState<boolean>(false);
  const leafletMapRef = useRef<L.Map | null>(null);

  useEffect(() => {
    connectSocket();

    const handleDriftCompleted = (data: any) => {
      if (data && data.envelopes_geojson) {
        setDriftEnvelopes(data.envelopes_geojson);
      }
    };

    socket.on('drift_completed', handleDriftCompleted);
    socket.on('system_status', (data: any) => console.log(data.message));

    return () => {
      socket.off('drift_completed', handleDriftCompleted);
      socket.off('system_status');
      // disconnectSocket(); // keeping it open for other views
    };
  }, []);

  const scenarioKey = scenario
    ? scenario.id.includes('001') ? 'INC-001'
    : scenario.id.includes('002') ? 'INC-002'
    : scenario.id.includes('003') ? 'INC-003'
    : scenario.id.includes('004') ? 'INC-004'
    : scenario.id.includes('005') ? 'INC-005'
    : scenario.id.includes('006') ? 'INC-006'
    : scenario.id.includes('007') ? 'INC-007'
    : scenario.id.includes('008') ? 'INC-008'
    : 'INC-001'
    : 'INC-001';

  let rawImageSrc = `/imagery/sar_${scenarioKey}.webp`;
  if (selectedCopernicusLayer === 'sar-vv') rawImageSrc = `/imagery/sar_${scenarioKey}.webp`;
  else if (selectedCopernicusLayer === 'sar-vh') rawImageSrc = `/imagery/vh_${scenarioKey}.webp`;
  else if (selectedCopernicusLayer === 'true-color') rawImageSrc = `/imagery/tc_${scenarioKey}.png`;
  else if (selectedCopernicusLayer === 'swir-oil') rawImageSrc = `/imagery/swir_${scenarioKey}.png`;
  else if (selectedCopernicusLayer === 'false-color') rawImageSrc = `/imagery/cir_${scenarioKey}.png`;
  else if (selectedCopernicusLayer === 'nisar-ls') rawImageSrc = `/imagery/nisar_${scenarioKey}.webp`;
  else if (selectedCopernicusLayer === 'eos-04') rawImageSrc = `/imagery/eos04_${scenarioKey}.webp`;

  const executeDetection = async () => {
    if (!scenario) {
      setDetectionResult(null);
      setIsDetecting(false);
      return;
    }
    setIsDetecting(true);
    const aoi = createAoiForScenario(scenario.lat, scenario.lng);
    const result = await runSentinel1Detection(aoi, undefined, 0.35);
    setDetectionResult(result);
    setIsDetecting(false);
  };

  useEffect(() => {
    executeDetection();
  }, [scenario?.id]);

  const handleSelectBasemap = (layer: BaseLayerType) => {
    setBaseLayer2D(layer);
    if (layer === 'bhuvan') {
      // Auto-apply ISRO Bhuvan preset: NISAR dual-band, AI Analysis, EEZ, Overlay at 76%
      setActiveSatellite('ISRO NISAR');
      setSelectedCopernicusLayer('nisar-ls');
      setShowAiMask(true);
      setShowEezBoundary(true);
      setShowIndiaOutline(true);
      setShowSpillOverlay(true);
      setLayerOpacity(0.76);
      setSatelliteToast('🇮🇳 ISRO Bhuvan + NISAR SweepSAR L+S • Auto-preset applied');
      setTimeout(() => setSatelliteToast(null), 3500);
    } else if (layer === 'satellite' || layer === 'arcgis') {
      handleSelectCopernicusLayer('sar-vv');
      setSatelliteToast('🛰️ ArcGIS World Imagery • High-Res Sub-Meter Satellite (Zoom 19+)');
      setTimeout(() => setSatelliteToast(null), 3500);
    } else if (layer === 'sar') {
      handleSelectCopernicusLayer('sar-vv');
    } else if (layer === 'bhuvan-satellite') {
      setSatelliteToast('🌊 ArcGIS World Ocean Base • Bathymetry & Seabed Relief');
      setTimeout(() => setSatelliteToast(null), 3500);
    }
  };

  const handleSelectSatellite = (name: string) => {
    setActiveSatellite(name);
    let newLayer: CopernicusLayerId = 'sar-vv';
    let sensorDesc = 'Sentinel-1A C-SAR IW GRD VV (10m Radar)';

    if (name === 'Sentinel-1A') {
      newLayer = 'sar-vv';
      sensorDesc = 'Sentinel-1A C-SAR IW GRD VV (Decibels) • Radar Damping';
    } else if (name === 'Sentinel-1C') {
      newLayer = 'sar-vh';
      sensorDesc = 'Sentinel-1C C-SAR Cross-Polarization (VH) • Ship & Roughness';
    } else if (name === 'Sentinel-2A') {
      newLayer = 'true-color';
      sensorDesc = 'Sentinel-2A MSI Natural True Color Optical • Sun-Glint Sheen';
    } else if (name === 'Sentinel-2B') {
      newLayer = 'swir-oil';
      sensorDesc = 'Sentinel-2B MSI SWIR Hydrocarbon Absorption (1610nm)';
    } else if (name === 'ISRO NISAR') {
      newLayer = 'nisar-ls';
      sensorDesc = 'ISRO NISAR SweepSAR Dual-Band L+S Radar (DFDI Bragg Damping)';
    } else if (name.includes('EOS-04')) {
      newLayer = 'eos-04';
      sensorDesc = 'ISRO EOS-04 (RISAT-1A) Circular Hybrid Polarimetry (CP Mode)';
    }

    setSelectedCopernicusLayer(newLayer);
    setSatelliteToast(sensorDesc);
    setTimeout(() => setSatelliteToast(null), 4500);

    // Fly camera only when a specific incident is active
    if (leafletMapRef.current && scenario) {
      leafletMapRef.current.flyTo([scenario.lat, scenario.lng], 12, { duration: 0.8 });
    }
  };

  const handleSelectSensor = (sensor: string) => {
    if (sensor === 'C-SAR') {
      handleSelectSatellite('Sentinel-1A');
    } else if (sensor === 'MSI') {
      handleSelectSatellite('Sentinel-2A');
    } else if (sensor.includes('SweepSAR')) {
      handleSelectSatellite('ISRO NISAR');
    }
  };

  const handleSelectCopernicusLayer = (id: CopernicusLayerId) => {
    setSelectedCopernicusLayer(id);
    if (id === 'sar-vv') setActiveSatellite('Sentinel-1A');
    else if (id === 'sar-vh') setActiveSatellite('Sentinel-1C');
    else if (id === 'true-color') setActiveSatellite('Sentinel-2A');
    else if (id === 'swir-oil') setActiveSatellite('Sentinel-2B');
    else if (id === 'nisar-ls') setActiveSatellite('ISRO NISAR');
    else if (id === 'eos-04') setActiveSatellite('EOS-04 (RISAT-1A)');

    const layerName = COPERNICUS_LAYERS.find((l) => l.id === id)?.name || id;
    setSatelliteToast(`Spectral Drape: ${layerName}`);
    setTimeout(() => setSatelliteToast(null), 4500);

    if (leafletMapRef.current && scenario) {
      leafletMapRef.current.flyTo([scenario.lat, scenario.lng], 12, { duration: 0.8 });
    }
  };

  const [panelTab, setPanelTab] = useState<'satellites' | 'spectral'>('satellites');
  const availableSatellites = [
    'Sentinel-1A',
    'Sentinel-1C',
    'Sentinel-2A',
    'Sentinel-2B',
    'ISRO NISAR',
    'EOS-04 (RISAT-1A)',
  ];
  const availableSensors = [
    { name: 'C-SAR', mode: '5.4GHz C-Band', sat: 'Sentinel-1A' },
    { name: 'MSI', mode: '13 Bands Optical', sat: 'Sentinel-2A' },
    { name: 'SweepSAR (L+S)', mode: 'Dual Frequency', sat: 'ISRO NISAR' },
  ];
  const [selectedProducts, setSelectedProducts] = useState<Record<string, boolean>>({
    'Sentinel-1A_C-SAR_Level-1_GRD': true,
    'Sentinel-2A_MSI_Level-2A': true,
    'NISAR_L_S_L2_GCOV': true,
  });
  const [searchFeedback, setSearchFeedback] = useState<string | null>(null);

  const toggleProduct = (prod: string) => {
    setSelectedProducts((prev) => ({ ...prev, [prod]: !prev[prod] }));
  };

  const handleBhoonidhiSubmit = () => {
    setSearchFeedback('Searching Bhoonidhi NRSC OpenData Node...');
    setTimeout(() => {
      setSearchFeedback(`3 Valid Passes Ingested • AOI: ${scenario ? scenario.title : 'Indian EEZ'} (10m RTC Calibrated)`);
      setTimeout(() => setSearchFeedback(null), 5000);
    }, 600);
  };

  const handleFocusSpill = () => {
    if (leafletMapRef.current) {
      if (scenario) {
        leafletMapRef.current.flyTo([scenario.lat, scenario.lng], 12, { duration: 0.8 });
      } else {
        leafletMapRef.current.flyTo([15.5, 79.0], 4.25, { duration: 0.8 });
      }
    }
  };

  const handleCleanupClick = () => {
    // If an incident is already selected, open the Clean-Up modal directly
    if (scenario) {
      setIsCleanupModalOpen(true);
      setShowIncidentDropdownMenu(false);
      return;
    }

    // No incident selected: open the incident dropdown so user can select which one they want
    pendingCleanupRef.current = true;

    // Trigger incident picker in Topbar
    const selectEl = document.getElementById('scenario-dropdown') as HTMLSelectElement | null;
    let pickerOpened = false;

    if (selectEl) {
      const pill = selectEl.closest('.scenario-selector-pill');
      if (pill) {
        pill.classList.add('highlight-cleanup-prompt');
      }
      selectEl.focus();

      // Clear pending state after 25s timeout if user abandons selection
      setTimeout(() => {
        if (!scenario && pendingCleanupRef.current) {
          pendingCleanupRef.current = false;
          pill?.classList.remove('highlight-cleanup-prompt');
        }
      }, 25000);

      if (typeof selectEl.showPicker === 'function') {
        try {
          selectEl.showPicker();
          pickerOpened = true;
        } catch (err) {
          console.warn('showPicker on scenario-dropdown failed:', err);
          pickerOpened = false;
        }
      }
    }

    // Dispatch event to Topbar for custom UI enhancement
    window.dispatchEvent(
      new CustomEvent('spill-sense:request-incident-selection', {
        detail: { reason: 'cleanup' },
      })
    );

    // If native picker didn't open (or unsupported), toggle inline dropdown menu
    if (!pickerOpened) {
      setShowIncidentDropdownMenu((prev) => !prev);
    }
  };

  const handleSelectIncidentFromCleanupMenu = (key: string) => {
    pendingCleanupRef.current = true;
    setShowIncidentDropdownMenu(false);
    if (onSelectScenario) {
      onSelectScenario(key);
    }
  };

  const handleStepDate = (deltaDays: number) => {
    const d = new Date(selectedDate);
    d.setDate(d.getDate() + deltaDays);
    setSelectedDate(d.toISOString().split('T')[0]);
  };

  return (
    <div className="map-panel" style={{ flex: 1, width: '100%', height: '100%', minHeight: 0, display: 'flex', flexDirection: 'column' }}>
      {/* MINIMALIST MARITIME GIS TOOLBAR */}
      <div className="map-header" style={{ padding: '6px 12px', gap: 10, display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'nowrap', overflowX: 'auto', background: 'var(--bg-surface)', borderBottom: '1px solid var(--border-subtle)' }}>
        {/* LEFT: Basemap Switcher */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
          <div className="map-base-selector" style={{ display: 'flex', background: 'var(--bg-raised)', border: '1px solid var(--border-subtle)', borderRadius: 4, padding: 2 }}>
            <button
              className={`base-btn ${baseLayer2D === 'bhuvan' ? 'active' : ''}`}
              onClick={() => handleSelectBasemap('bhuvan')}
              style={{ fontSize: 11, padding: '3px 8px', borderRadius: 3 }}
              title="Official ISRO Bhuvan & Bhoonidhi Satellite Imagery (Govt. of India / NRSC)"
            >
              ISRO Bhuvan
            </button>
            <button
              className={`base-btn ${baseLayer2D === 'satellite' || baseLayer2D === 'arcgis' ? 'active' : ''}`}
              onClick={() => handleSelectBasemap('arcgis')}
              style={{ fontSize: 11, padding: '3px 8px', borderRadius: 3 }}
              title="ArcGIS World Imagery High-Resolution Sub-Meter Satellite (Zoom 19+)"
            >
              ArcGIS
            </button>
            <button
              className={`base-btn ${baseLayer2D === 'bhuvan-satellite' ? 'active' : ''}`}
              onClick={() => handleSelectBasemap('bhuvan-satellite')}
              style={{ fontSize: 11, padding: '3px 8px', borderRadius: 3 }}
              title="Bhoonidhi Ocean Depth Relief & Bathymetry"
            >
              Bathymetry
            </button>
            <button
              className={`base-btn ${baseLayer2D === 'carto-voyager' || baseLayer2D === 'google-satellite' || baseLayer2D === 'opensea' ? 'active' : ''}`}
              onClick={() => handleSelectBasemap('carto-voyager')}
              style={{ fontSize: 11, padding: '3px 8px', borderRadius: 3 }}
              title="Google Earth Ultra-HD Satellite (Zoom 21 - View real boats, ships, wakes, piers & harbor SPM)"
            >
              Google Earth
            </button>
          </div>
        </div>

        {/* RIGHT: Tactical Toggles & Actions */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexShrink: 0 }}>
          {/* AI Analysis Mask Toggle */}
          <button
            className={`btn ${showAiMask ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setShowAiMask(!showAiMask)}
            style={{ padding: '3px 9px', fontSize: 11, gap: 5 }}
            title="Toggle AI Segmentation Mask, Lagrangian Drift & Vessel Tracking"
          >
            <span className="material-symbols-outlined" style={{ fontSize: 14 }}>
              {showAiMask ? 'visibility' : 'visibility_off'}
            </span>
            AI Analysis
          </button>

          {/* India Sovereign Borders & EEZ Toggle */}
          <button
            className={`btn ${showIndiaOutline ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => {
              const next = !showIndiaOutline;
              setShowIndiaOutline(next);
              setShowEezBoundary(next);
            }}
            style={{ padding: '3px 9px', fontSize: 11, gap: 5 }}
            title="Toggle Survey of India Borders & 200 NM EEZ Maritime Limit"
          >
            <span className="material-symbols-outlined" style={{ fontSize: 14 }}>public</span>
            Borders &amp; EEZ
          </button>

          <span style={{ color: 'var(--border-subtle)', fontSize: 12, margin: '0 2px' }}>|</span>

          <div ref={cleanupButtonContainerRef} style={{ position: 'relative', display: 'inline-flex' }}>
            <button
              className="btn btn-secondary"
              onClick={handleCleanupClick}
              style={{
                padding: '3px 9px',
                fontSize: 11,
                gap: 5,
                borderColor: 'rgba(16, 185, 129, 0.4)',
                background: 'rgba(16, 185, 129, 0.08)',
              }}
              title={
                scenario
                  ? `NOS-DCP Clean-Up Logistics & Resource Allocation Plan (${scenario.id})`
                  : 'Select an incident to view NOS-DCP Clean-Up Logistics'
              }
            >
              <span className="material-symbols-outlined" style={{ fontSize: 14, color: '#10B981' }}>
                cleaning_services
              </span>
              <span style={{ color: '#10B981', fontWeight: 600 }}>Clean-Up</span>
              {!scenario && (
                <span className="material-symbols-outlined" style={{ fontSize: 13, color: '#10B981', marginLeft: -2 }}>
                  expand_more
                </span>
              )}
            </button>

            {showIncidentDropdownMenu && !scenario && (
              <div
                className="cleanup-incident-dropdown"
                style={{
                  position: 'absolute',
                  top: '100%',
                  right: 0,
                  marginTop: 6,
                  background: 'var(--bg-raised, #0f172a)',
                  border: '1px solid rgba(16, 185, 129, 0.4)',
                  borderRadius: 6,
                  boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
                  minWidth: 280,
                  zIndex: 1000,
                  padding: '6px 0',
                }}
              >
                <div
                  style={{
                    padding: '6px 12px',
                    fontSize: 10,
                    fontWeight: 700,
                    color: '#10B981',
                    textTransform: 'uppercase',
                    letterSpacing: '0.05em',
                    borderBottom: '1px solid var(--border-subtle, rgba(255,255,255,0.08))',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6,
                  }}
                >
                  <span className="material-symbols-outlined" style={{ fontSize: 13 }}>touch_app</span>
                  Select Incident For Clean-Up Plan
                </div>
                {Object.entries(activeScenarios).map(([key, s]) => (
                  <div
                    key={key}
                    onClick={() => handleSelectIncidentFromCleanupMenu(key)}
                    className="cleanup-dropdown-item"
                    style={{
                      padding: '8px 12px',
                      fontSize: 11,
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 8,
                      transition: 'background 0.15s ease',
                    }}
                  >
                    <span className="material-symbols-outlined" style={{ fontSize: 14, color: '#10B981' }}>
                      water_drop
                    </span>
                    <div>
                      <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                        {s.id}: {s.title}
                      </div>
                      <div style={{ fontSize: 9.5, color: 'var(--text-muted)' }}>
                        {s.oilType} · {s.sev}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <button
            className="btn btn-secondary"
            onClick={handleFocusSpill}
            style={{ padding: '3px 9px', fontSize: 11, gap: 5 }}
            title="Recenter Camera on Spill Coordinates"
          >
            <span className="material-symbols-outlined" style={{ fontSize: 14 }}>my_location</span>
            Recenter
          </button>

          <button
            className="btn btn-secondary"
            onClick={() => setShowRawImageModal(true)}
            style={{ padding: '3px 9px', fontSize: 11, gap: 5 }}
            title="Inspect Full 2048px Raw Satellite Image"
          >
            <span className="material-symbols-outlined" style={{ fontSize: 14 }}>fullscreen</span>
            Full Image
          </button>

          <button
            className={`btn ${isSideLayersOpen ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setIsSideLayersOpen(!isSideLayersOpen)}
            style={{ padding: '3px 9px', fontSize: 11, gap: 5 }}
            title="Toggle Satellite Pass & Spectral Channel Browser"
          >
            <span className="material-symbols-outlined" style={{ fontSize: 14 }}>tune</span>
            Sensors
          </button>
        </div>
      </div>

      {/* MAP CANVAS & COPERNICUS BROWSER PANEL */}
      <div style={{ position: 'relative', flex: 1, display: 'flex', minHeight: 0, height: '100%', overflow: 'hidden' }}>
        {/* COPERNICUS BROWSER SIDE LA        {/* BHOONIDHI & COPERNICUS SATELLITE & SENSOR SIDE PANEL */}
        <div className={`copernicus-side-panel ${isSideLayersOpen ? '' : 'collapsed'}`}>
          <div className="copernicus-header" style={{ flexShrink: 0 }}>
            <div className="copernicus-brand">
              <span className="material-symbols-outlined" style={{ fontSize: 18, color: '#38BDF8' }}>satellite_alt</span>
              <div>
                <div style={{ fontSize: 12, fontWeight: 700, letterSpacing: '0.02em', display: 'flex', alignItems: 'center', gap: 5 }}>
                  <span>ISRO Bhoonidhi &amp; Copernicus</span>
                  <span style={{ fontSize: 8, background: '#10B98122', color: '#10B981', padding: '1px 4px', borderRadius: 3, border: '1px solid #10B98144' }}>ONLINE</span>
                </div>
                <div style={{ fontSize: 9, color: 'var(--text-muted)' }}>
                  User: <strong style={{ color: '#F59E0B' }}>KRISHNA KANT</strong> • NRSC OpenData Node
                </div>
              </div>
            </div>
            <button
              className="copernicus-collapse-btn"
              onClick={() => setIsSideLayersOpen(false)}
              title="Collapse Satellite Panel"
            >
              <span className="material-symbols-outlined" style={{ fontSize: 18 }}>chevron_left</span>
            </button>
          </div>

          {/* Sub-Tabs: Satellites & Sensors VS Spectral Drapes */}
          <div style={{ display: 'flex', borderBottom: '1px solid var(--border-subtle)', background: 'var(--bg-raised)', flexShrink: 0 }}>
            <button
              onClick={() => setPanelTab('satellites')}
              style={{
                flex: 1,
                padding: '7px 4px',
                fontSize: 10.5,
                fontWeight: 700,
                border: 'none',
                background: panelTab === 'satellites' ? 'var(--bg-card)' : 'transparent',
                color: panelTab === 'satellites' ? 'var(--accent)' : 'var(--text-muted)',
                borderBottom: panelTab === 'satellites' ? '2px solid var(--accent)' : '2px solid transparent',
                cursor: 'pointer',
              }}
            >
              Satellites ({availableSatellites.length})
            </button>
            <button
              onClick={() => setPanelTab('spectral')}
              style={{
                flex: 1,
                padding: '7px 4px',
                fontSize: 10.5,
                fontWeight: 700,
                border: 'none',
                background: panelTab === 'spectral' ? 'var(--bg-card)' : 'transparent',
                color: panelTab === 'spectral' ? 'var(--accent)' : 'var(--text-muted)',
                borderBottom: panelTab === 'spectral' ? '2px solid var(--accent)' : '2px solid transparent',
                cursor: 'pointer',
              }}
            >
              Spectral ({COPERNICUS_LAYERS.length})
            </button>
          </div>

          {/* Date Strip */}
          <div style={{ padding: '6px 10px', background: '#F8FAFC', borderBottom: '1px solid #E2E8F0', display: 'flex', flexDirection: 'column', gap: 4, flexShrink: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <button className="btn btn-secondary" onClick={() => handleStepDate(-1)} style={{ padding: '1px 5px', fontSize: 10 }}>◀</button>
                <input
                  type="date"
                  value={selectedDate}
                  onChange={(e) => setSelectedDate(e.target.value)}
                  style={{
                    fontSize: 10.5,
                    fontFamily: 'JetBrains Mono',
                    padding: '2px 4px',
                    borderRadius: 4,
                    border: '1px solid #CBD5E1',
                    background: '#FFFFFF',
                    color: '#0F172A',
                  }}
                />
                <button className="btn btn-secondary" onClick={() => handleStepDate(1)} style={{ padding: '1px 5px', fontSize: 10 }}>▶</button>
              </div>
              <span className="chip" style={{ background: '#E0F2FE', color: '#0369A1', fontSize: 8.5, fontWeight: 700 }}>
                {cloudCoverMax}% Max Cloud
              </span>
            </div>
          </div>

          {panelTab === 'satellites' ? (
            /* TAB A: SATELLITES & SENSORS SELECTION (REACTIVE GIS SUITE) */
            <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: '8px 10px', display: 'flex', flexDirection: 'column', gap: 10 }}>
              {/* Mission scope notice */}
              <div style={{ background: 'rgba(56, 189, 248, 0.08)', border: '1px solid rgba(56, 189, 248, 0.25)', borderRadius: 4, padding: '5px 7px' }}>
                <div style={{ fontSize: 9.5, fontWeight: 700, color: '#38BDF8', marginBottom: 2 }}>
                  Marine Satellite Intelligence Suite
                </div>
                <div style={{ fontSize: 8.5, color: 'var(--text-secondary)', lineHeight: 1.3 }}>
                  Select any satellite below to stream calibrated imagery onto the map and focus the acquisition scene.
                </div>
              </div>

              {/* 1. SATELLITE MISSIONS (ACTIVE ONE-CLICK SELECTION) */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                  <span style={{ fontSize: 10.5, fontWeight: 700, color: 'var(--text-primary)' }}>
                    Satellite Missions ({availableSatellites.length})
                  </span>
                  <span style={{ fontSize: 8.5, color: '#F59E0B', fontFamily: 'monospace' }}>Bhoonidhi/CDSE</span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 4, background: 'var(--bg-card)', padding: '5px', borderRadius: 4, border: '1px solid var(--border-subtle)' }}>
                  {availableSatellites.map((satName) => {
                    const isRadar = satName.includes('Sentinel-1') || satName.includes('NISAR') || satName.includes('EOS-04');
                    const isActive = activeSatellite === satName;
                    return (
                      <div
                        key={satName}
                        onClick={() => handleSelectSatellite(satName)}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: 6,
                          fontSize: 10,
                          cursor: 'pointer',
                          padding: '5px 7px',
                          borderRadius: 4,
                          border: isActive ? '1px solid #00E5FF' : '1px solid transparent',
                          background: isActive ? 'rgba(0, 229, 255, 0.12)' : 'transparent',
                          transition: 'all 0.15s ease',
                        }}
                      >
                        <span
                          style={{
                            width: 8,
                            height: 8,
                            borderRadius: '50%',
                            background: isActive ? '#00E5FF' : '#64748B',
                            boxShadow: isActive ? '0 0 8px #00E5FF' : 'none',
                            flexShrink: 0,
                          }}
                        />
                        <span style={{ fontWeight: isActive ? 700 : 500, flex: 1, color: isActive ? '#00E5FF' : 'var(--text-primary)' }}>
                          {satName}
                        </span>
                        {isActive && (
                          <span style={{ fontSize: 7.5, padding: '1px 5px', borderRadius: 2, background: 'rgba(0, 229, 255, 0.25)', color: '#00E5FF', fontWeight: 800 }}>
                            ON MAP
                          </span>
                        )}
                        <span style={{ fontSize: 7.5, padding: '1px 4px', borderRadius: 2, background: isRadar ? 'rgba(8, 145, 178, 0.2)' : 'rgba(2, 132, 199, 0.2)', color: isRadar ? '#0891B2' : '#0284C7', fontWeight: 700 }}>
                          {isRadar ? 'RADAR' : 'OPTICAL'}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* 2. SENSOR TYPE SELECTION */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                  <span style={{ fontSize: 10.5, fontWeight: 700, color: 'var(--text-primary)' }}>
                    Sensor Modes ({availableSensors.length})
                  </span>
                  <span style={{ fontSize: 8.5, color: 'var(--text-muted)' }}>Physical Channels</span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 4, background: 'var(--bg-card)', padding: '5px', borderRadius: 4, border: '1px solid var(--border-subtle)' }}>
                  {availableSensors.map((sensor) => {
                    const isMatchingActive =
                      (sensor.name === 'C-SAR' && activeSatellite.includes('Sentinel-1')) ||
                      (sensor.name === 'MSI' && activeSatellite.includes('Sentinel-2')) ||
                      (sensor.name.includes('SweepSAR') && activeSatellite.includes('NISAR'));
                    return (
                      <div
                        key={sensor.name}
                        onClick={() => handleSelectSensor(sensor.name)}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: 6,
                          fontSize: 10,
                          cursor: 'pointer',
                          padding: '4px 6px',
                          borderRadius: 3,
                          background: isMatchingActive ? 'rgba(16, 185, 129, 0.12)' : 'transparent',
                          border: isMatchingActive ? '1px solid #10B981' : '1px solid transparent',
                        }}
                      >
                        <span
                          style={{
                            width: 6,
                            height: 6,
                            borderRadius: '50%',
                            background: isMatchingActive ? '#10B981' : '#64748B',
                            flexShrink: 0,
                          }}
                        />
                        <span style={{ fontWeight: isMatchingActive ? 700 : 400, flex: 1, color: isMatchingActive ? '#10B981' : 'var(--text-primary)' }}>
                          {sensor.name}
                        </span>
                        <span style={{ fontSize: 8, color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                          {sensor.mode}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* 3. PRODUCTS (OPENDATA DIRECT) */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                  <span style={{ fontSize: 10.5, fontWeight: 700, color: 'var(--text-primary)' }}>
                    Products (OpenData Direct)
                  </span>
                  <span style={{ fontSize: 8.5, color: '#16A34A', fontWeight: 700 }}>● FREE</span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 3, background: 'var(--bg-card)', padding: '5px 6px', borderRadius: 4, border: '1px solid var(--border-subtle)' }}>
                  {Object.entries(selectedProducts).map(([prodName, isChecked]) => (
                    <label key={prodName} style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 9.5, cursor: 'pointer' }}>
                      <input
                        type="checkbox"
                        checked={isChecked}
                        onChange={() => toggleProduct(prodName)}
                        style={{ accentColor: '#0284C7' }}
                      />
                      <span className="mono" style={{ fontSize: 8.5, color: isChecked ? 'var(--text-primary)' : 'var(--text-muted)' }}>
                        {prodName}
                      </span>
                    </label>
                  ))}
                </div>
              </div>

              {/* 4. DIRECT SATELLITE LAYER DRAPE */}
              <div>
                <div style={{ fontSize: 10.5, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 4 }}>
                  Direct Satellite Layer Drape
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
                  <button
                    className={`btn btn-sm ${selectedCopernicusLayer === 'sar-vv' ? 'btn-primary' : 'btn-secondary'}`}
                    onClick={() => handleSelectSatellite('Sentinel-1A')}
                    style={{ fontSize: 9.5, padding: '3px 5px', justifyContent: 'center' }}
                  >
                    C-SAR (VV)
                  </button>
                  <button
                    className={`btn btn-sm ${selectedCopernicusLayer === 'sar-vh' ? 'btn-primary' : 'btn-secondary'}`}
                    onClick={() => handleSelectSatellite('Sentinel-1C')}
                    style={{ fontSize: 9.5, padding: '3px 5px', justifyContent: 'center' }}
                  >
                    C-SAR (VH)
                  </button>
                  <button
                    className={`btn btn-sm ${selectedCopernicusLayer === 'true-color' ? 'btn-primary' : 'btn-secondary'}`}
                    onClick={() => handleSelectSatellite('Sentinel-2A')}
                    style={{ fontSize: 9.5, padding: '3px 5px', justifyContent: 'center' }}
                  >
                    True Color (MSI)
                  </button>
                  <button
                    className={`btn btn-sm ${selectedCopernicusLayer === 'swir-oil' ? 'btn-primary' : 'btn-secondary'}`}
                    onClick={() => handleSelectSatellite('Sentinel-2B')}
                    style={{ fontSize: 9.5, padding: '3px 5px', justifyContent: 'center' }}
                  >
                    MSI SWIR
                  </button>
                  <button
                    className={`btn btn-sm ${selectedCopernicusLayer === 'nisar-ls' ? 'btn-primary' : 'btn-secondary'}`}
                    onClick={() => handleSelectSatellite('ISRO NISAR')}
                    style={{ fontSize: 9.5, padding: '3px 5px', gridColumn: 'span 2', justifyContent: 'center', borderColor: selectedCopernicusLayer === 'nisar-ls' ? '#F59E0B' : undefined }}
                  >
                    ISRO NISAR Dual-Band (DFDI)
                  </button>
                  <button
                    className={`btn btn-sm ${selectedCopernicusLayer === 'eos-04' ? 'btn-primary' : 'btn-secondary'}`}
                    onClick={() => handleSelectSatellite('EOS-04 (RISAT-1A)')}
                    style={{ fontSize: 9.5, padding: '3px 5px', gridColumn: 'span 2', justifyContent: 'center', borderColor: selectedCopernicusLayer === 'eos-04' ? '#10B981' : undefined }}
                  >
                    ISRO EOS-04 Circular Pol (CP)
                  </button>
                </div>
              </div>

              {/* 5. SUBMIT QUERY */}
              <div style={{ marginTop: 'auto' }}>
                <button
                  className="btn btn-primary w-full"
                  onClick={handleBhoonidhiSubmit}
                  style={{ width: '100%', padding: '5px', fontSize: 10.5, fontWeight: 700, background: '#0284C7' }}
                >
                  Query Bhoonidhi &amp; Copernicus
                </button>
                {searchFeedback && (
                  <div style={{ marginTop: 5, fontSize: 9, color: '#10B981', fontWeight: 600, textAlign: 'center' }}>
                    {searchFeedback}
                  </div>
                )}
              </div>
            </div>
          ) : (
            /* TAB B: SPECTRAL DRAPES LIST */
            <div className="copernicus-layers-list" style={{ flex: 1, minHeight: 0, overflowY: 'auto' }}>
              {COPERNICUS_LAYERS.map((layer) => {
                const isActive = selectedCopernicusLayer === layer.id;
                return (
                  <div
                    key={layer.id}
                    className={`copernicus-layer-card ${isActive ? 'active' : ''}`}
                    onClick={() => handleSelectCopernicusLayer(layer.id)}
                  >
                    <div className="layer-thumb" style={{ background: layer.thumbBg }}>
                      <span className="material-symbols-outlined" style={{ fontSize: 16, color: '#FFFFFF' }}>
                        {layer.id === 'true-color'
                          ? 'photo_camera'
                          : layer.id === 'sar-vv'
                          ? 'radar'
                          : layer.id === 'swir-oil'
                          ? 'oil_barrel'
                          : layer.id === 'false-color'
                          ? 'nature'
                          : layer.id === 'ndwi'
                          ? 'waves'
                          : 'thermostat'}
                      </span>
                    </div>
                    <div className="layer-info">
                      <div className="layer-name">
                        <span>{layer.name}</span>
                        {isActive && (
                          <span style={{ fontSize: 9, color: '#0284C7', fontWeight: 800 }}>ACTIVE</span>
                        )}
                      </div>
                      <div className="layer-bands">{layer.bands}</div>
                      <div className="layer-desc">{layer.desc}</div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* Docked Layer Opacity Slider Footer */}
          <div
            className="copernicus-opacity-footer"
            style={{
              padding: '8px 12px',
              borderTop: '1px solid var(--border-default)',
              background: 'var(--bg-raised)',
              flexShrink: 0,
              marginTop: 'auto',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, fontWeight: 600, marginBottom: 4 }}>
              <span style={{ color: 'var(--text-muted)' }}>Layer Opacity:</span>
              <span className="mono" style={{ color: 'var(--accent)' }}>{Math.round(layerOpacity * 100)}%</span>
            </div>
            <input
              type="range"
              min="0.15"
              max="1.0"
              step="0.05"
              value={layerOpacity}
              onChange={(e) => setLayerOpacity(parseFloat(e.target.value))}
              style={{ width: '100%', accentColor: 'var(--accent)', cursor: 'pointer', display: 'block' }}
            />
          </div>
        </div>

        {/* Collapse / Expand Tab Button */}
        {!isSideLayersOpen && (
          <button
            className="panel-toggle-btn"
            onClick={() => setIsSideLayersOpen(true)}
            title="Open Copernicus Layers"
          >
            <span className="material-symbols-outlined" style={{ fontSize: 15 }}>layers</span>
            Layers
          </button>
        )}

        {/* Dynamic Satellite Ingestion Toast Banner */}
        {satelliteToast && (
          <div
            style={{
              position: 'absolute',
              top: 14,
              left: '50%',
              transform: 'translateX(-50%)',
              zIndex: 1000,
              background: 'rgba(15, 23, 42, 0.94)',
              backdropFilter: 'blur(10px)',
              border: '1px solid #00E5FF',
              boxShadow: '0 4px 20px rgba(0, 229, 255, 0.35)',
              borderRadius: 20,
              padding: '6px 18px',
              fontSize: 11.5,
              fontWeight: 700,
              color: '#00E5FF',
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              pointerEvents: 'none',
              animation: 'fadeIn 0.2s ease-out',
            }}
          >
            <span className="material-symbols-outlined" style={{ fontSize: 16, color: '#00E5FF' }}>satellite_alt</span>
            <span>{satelliteToast}</span>
          </div>
        )}

        {/* Dedicated 2D Leaflet Map Viewport */}
        <LeafletMap
          scenario={scenario}
          baseLayer={baseLayer2D}
          showSeamarks={showSeamarks}
          showIndiaOutline={showIndiaOutline}
          showEezBoundary={showEezBoundary}
          selectedCopernicusLayer={selectedCopernicusLayer}
          layerOpacity={layerOpacity}
          showAiMask={showAiMask}
          showOverlay={showSpillOverlay}
          scrubHours={scrubHours}
          detectionResult={detectionResult}
          driftEnvelopes={driftEnvelopes}
          onUpdateCoords={onUpdateCoords}
          onSelectScenario={onSelectScenario}
          mapRef={leafletMapRef}
          scenarios={scenarios}
          targetLocation={targetLocation}
          sarDriftPayload={sarDriftPayload}
          activeDriftSimulation={activeDriftSimulation}
        />

        {/* Cartographic Legend (Superhuman Glassmorphic HUD at Bottom-Right) */}
        <div className="map-overlay-hud">
          <div className="hud-header">
            <span className="material-symbols-outlined" style={{ fontSize: 13, color: '#38BDF8' }}>layers</span>
            <span className="hud-title">{showAiMask ? 'AI Detection & Drift Layers' : 'Raw Satellite Mode'}</span>
          </div>

          {showAiMask ? (
            <>
              <div className="hud-row">
                <div className="lswatch" style={{ background: '#00E5FF' }}></div>
                <span><strong>Current Spill (T0)</strong>: <span style={{ color: '#38BDF8' }}>{scenario?.area || 'Monitored'}</span></span>
              </div>

              <div className="hud-row">
                <div className="lswatch" style={{ background: '#10B981', border: '1px dashed #10B981' }}></div>
                <span><strong>Predicted Drift (T+24h)</strong>: Forecast</span>
              </div>

              <div className="hud-row">
                <div className="lswatch" style={{ background: '#F59E0B' }}></div>
                <span><strong>Discharge Origin (T-22h)</strong>: Source</span>
              </div>

              <div className="hud-row">
                <div className="lswatch" style={{ background: '#D97706', border: '1px dashed #D97706' }}></div>
                <span><strong>Reverse Drift Track</strong>: Past 22h</span>
              </div>

              <div className="hud-row">
                <div className="lswatch" style={{ background: '#3B82F6' }}></div>
                <span><strong>AIS Transit Speed</strong>: &gt;10 kn</span>
              </div>

              <div className="hud-row">
                <div className="lswatch" style={{ background: '#F59E0B' }}></div>
                <span><strong>AIS Slow / Maneuver</strong>: 2.5–6 kn</span>
              </div>

              <div className="hud-row">
                <div className="lswatch" style={{ background: '#DC2626', border: '1px dashed #FFFFFF' }}></div>
                <span><strong>AIS Silence Gap</strong>: Transponder Off</span>
              </div>
            </>
          ) : (
            <div style={{ padding: '4px 0', fontSize: 10, color: '#94A3B8', lineHeight: 1.4 }}>
              <div style={{ color: '#38BDF8', fontWeight: 700, marginBottom: 2 }}>Prism Raw Satellite View</div>
              <div>Solid 100% native unblurred satellite raster. Zero polygon masks covering the spill.</div>
              <div style={{ marginTop: 4, color: '#F59E0B', cursor: 'pointer' }} onClick={() => setShowAiMask(true)}>
                ➔ Click to toggle AI Detection &amp; Drift Mask
              </div>
            </div>
          )}

          {/* Sovereign & Maritime Boundaries */}
          {showIndiaOutline && (
            <div className="hud-row" style={{ borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 4, marginTop: 4 }}>
              <div className="lswatch" style={{ background: '#38BDF8', border: '1px solid #00E5FF' }}></div>
              <span><strong>India Sovereign Outline</strong>: <span style={{ color: '#38BDF8' }}>SOI Perimeter</span></span>
            </div>
          )}
          {showEezBoundary && (
            <div className="hud-row">
              <div className="lswatch" style={{ background: 'transparent', border: '1.5px dashed #F59E0B' }}></div>
              <span><strong>Indian EEZ (200 NM)</strong>: <span style={{ color: '#F59E0B' }}>Coast Guard Limit</span></span>
            </div>
          )}

          <div className="hud-footer">
            <span className="hud-status-dot"></span>
            <span className="hud-model-tag">{detectionResult?.model_version || 'unet-s1-sar-sos-v2.4-cdse'}</span>
            <span className="hud-model-sub">CDSE SAR Engine</span>
          </div>
        </div>
      </div>

      <TimeScrubber onTimeChange={setScrubHours} />

      {/* ── FULL-SCREEN 2048x2048 RAW SATELLITE SPILL IMAGE INSPECTOR MODAL ── */}
      {showRawImageModal && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 99999,
            backgroundColor: 'rgba(5, 10, 20, 0.90)',
            backdropFilter: 'blur(12px)',
            display: 'flex',
            flexDirection: 'column',
            animation: 'fadeIn 0.2s ease-out',
          }}
        >
          {/* Modal Header */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '12px 24px',
              borderBottom: '1px solid rgba(255,255,255,0.12)',
              background: 'rgba(15, 23, 42, 0.96)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <span className="material-symbols-outlined" style={{ fontSize: 24, color: '#00E5FF' }}>satellite_alt</span>
              <div>
                <div style={{ fontSize: 15, fontWeight: 800, color: '#FFFFFF', display: 'flex', alignItems: 'center', gap: 10 }}>
                  <span>Original Raw Satellite Capture (2048 × 2048 px)</span>
                  <span style={{ fontSize: 10, background: '#0284C722', color: '#38BDF8', border: '1px solid #0284C755', padding: '2px 8px', borderRadius: 4, fontFamily: 'monospace' }}>
                    100% UNCOMPRESSED NATIVE
                  </span>
                </div>
                <div style={{ fontSize: 11, color: '#94A3B8', marginTop: 2 }}>
                  Incident: <strong style={{ color: '#F8FAFC' }}>{scenario ? scenario.title : 'Mumbai High Basin'} ({scenarioKey})</strong> • Sensor: <strong style={{ color: '#00E5FF' }}>{selectedMission}</strong> • Resolution: <strong>10m/pixel (Calibrated RTC dB)</strong>
                </div>
              </div>
            </div>

            {/* Band selector pills inside modal */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, background: 'rgba(30,41,59,0.8)', padding: '3px 6px', borderRadius: 6, border: '1px solid rgba(255,255,255,0.1)' }}>
              <button
                onClick={() => handleSelectCopernicusLayer('sar-vv')}
                style={{
                  fontSize: 10.5,
                  padding: '4px 8px',
                  borderRadius: 4,
                  border: 'none',
                  background: selectedCopernicusLayer === 'sar-vv' ? '#0891B2' : 'transparent',
                  color: selectedCopernicusLayer === 'sar-vv' ? '#FFFFFF' : '#94A3B8',
                  fontWeight: selectedCopernicusLayer === 'sar-vv' ? 700 : 500,
                  cursor: 'pointer',
                }}
              >
                SAR VV (Damping)
              </button>
              <button
                onClick={() => handleSelectCopernicusLayer('sar-vh')}
                style={{
                  fontSize: 10.5,
                  padding: '4px 8px',
                  borderRadius: 4,
                  border: 'none',
                  background: selectedCopernicusLayer === 'sar-vh' ? '#0EA5E9' : 'transparent',
                  color: selectedCopernicusLayer === 'sar-vh' ? '#FFFFFF' : '#94A3B8',
                  fontWeight: selectedCopernicusLayer === 'sar-vh' ? 700 : 500,
                  cursor: 'pointer',
                }}
              >
                SAR VH (Cross-Pol)
              </button>
              <button
                onClick={() => handleSelectCopernicusLayer('true-color')}
                style={{
                  fontSize: 10.5,
                  padding: '4px 8px',
                  borderRadius: 4,
                  border: 'none',
                  background: selectedCopernicusLayer === 'true-color' ? '#0284C7' : 'transparent',
                  color: selectedCopernicusLayer === 'true-color' ? '#FFFFFF' : '#94A3B8',
                  fontWeight: selectedCopernicusLayer === 'true-color' ? 700 : 500,
                  cursor: 'pointer',
                }}
              >
                MSI True Color
              </button>
              <button
                onClick={() => handleSelectCopernicusLayer('swir-oil')}
                style={{
                  fontSize: 10.5,
                  padding: '4px 8px',
                  borderRadius: 4,
                  border: 'none',
                  background: selectedCopernicusLayer === 'swir-oil' ? '#B45309' : 'transparent',
                  color: selectedCopernicusLayer === 'swir-oil' ? '#FFFFFF' : '#94A3B8',
                  fontWeight: selectedCopernicusLayer === 'swir-oil' ? 700 : 500,
                  cursor: 'pointer',
                }}
              >
                SWIR Hydrocarbon
              </button>
              <button
                onClick={() => handleSelectCopernicusLayer('nisar-ls')}
                style={{
                  fontSize: 10.5,
                  padding: '4px 8px',
                  borderRadius: 4,
                  border: 'none',
                  background: selectedCopernicusLayer === 'nisar-ls' ? '#F59E0B' : 'transparent',
                  color: selectedCopernicusLayer === 'nisar-ls' ? '#000000' : '#94A3B8',
                  fontWeight: selectedCopernicusLayer === 'nisar-ls' ? 800 : 500,
                  cursor: 'pointer',
                }}
              >
                🇮🇳 NISAR L+S
              </button>
              <button
                onClick={() => handleSelectCopernicusLayer('eos-04')}
                style={{
                  fontSize: 10.5,
                  padding: '4px 8px',
                  borderRadius: 4,
                  border: 'none',
                  background: selectedCopernicusLayer === 'eos-04' ? '#10B981' : 'transparent',
                  color: selectedCopernicusLayer === 'eos-04' ? '#FFFFFF' : '#94A3B8',
                  fontWeight: selectedCopernicusLayer === 'eos-04' ? 700 : 500,
                  cursor: 'pointer',
                }}
              >
                🇮🇳 EOS-04 CP
              </button>
            </div>

            {/* Zoom Controls & Close */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <div style={{ display: 'flex', alignItems: 'center', background: 'rgba(30,41,59,0.8)', borderRadius: 4, border: '1px solid rgba(255,255,255,0.1)' }}>
                <button
                  onClick={() => setModalZoom((z) => Math.max(0.5, z - 0.25))}
                  style={{ background: 'transparent', border: 'none', color: '#FFFFFF', padding: '4px 8px', cursor: 'pointer', fontSize: 13 }}
                  title="Zoom Out"
                >
                  −
                </button>
                <span style={{ fontSize: 11, color: '#00E5FF', fontFamily: 'monospace', padding: '0 6px', fontWeight: 700 }}>
                  {Math.round(modalZoom * 100)}%
                </span>
                <button
                  onClick={() => setModalZoom((z) => Math.min(3.0, z + 0.25))}
                  style={{ background: 'transparent', border: 'none', color: '#FFFFFF', padding: '4px 8px', cursor: 'pointer', fontSize: 13 }}
                  title="Zoom In"
                >
                  +
                </button>
                <button
                  onClick={() => setModalZoom(1.0)}
                  style={{ background: 'transparent', border: 'none', color: '#94A3B8', padding: '4px 8px', cursor: 'pointer', fontSize: 10, borderLeft: '1px solid rgba(255,255,255,0.1)' }}
                  title="Reset 100%"
                >
                  100%
                </button>
              </div>

              <a
                href={rawImageSrc}
                download={`${scenarioKey}_${selectedCopernicusLayer}_raw_2048px.png`}
                className="btn btn-secondary"
                style={{ textDecoration: 'none', display: 'flex', alignItems: 'center', gap: 5, padding: '5px 10px', fontSize: 11, color: '#38BDF8', borderColor: '#38BDF8' }}
              >
                <span className="material-symbols-outlined" style={{ fontSize: 14 }}>download</span>
                Save PNG
              </a>

              <button
                onClick={() => setShowRawImageModal(false)}
                style={{
                  background: 'rgba(239,68,68,0.2)',
                  border: '1px solid rgba(239,68,68,0.4)',
                  color: '#EF4444',
                  borderRadius: 6,
                  width: 32,
                  height: 32,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  cursor: 'pointer',
                  fontWeight: 700,
                  fontSize: 16,
                }}
                title="Close Viewer"
              >
                ✕
              </button>
            </div>
          </div>

          {/* Modal Image Body (Scrollable / Pannable full resolution) */}
          <div
            style={{
              flex: 1,
              minHeight: 0,
              overflow: 'auto',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: '#070C18',
              padding: 24,
            }}
          >
            <div
              style={{
                position: 'relative',
                boxShadow: '0 8px 32px rgba(0, 0, 0, 0.9)',
                border: '1px solid rgba(0, 229, 255, 0.3)',
                borderRadius: 4,
                overflow: 'hidden',
                lineHeight: 0,
                transform: `scale(${modalZoom})`,
                transformOrigin: 'center center',
                transition: 'transform 0.15s ease-out',
              }}
            >
              <img
                src={rawImageSrc}
                alt={`Raw Satellite Spill Image ${scenarioKey}`}
                style={{
                  width: 860,
                  height: 860,
                  display: 'block',
                  imageRendering: 'crisp-edges',
                  objectFit: 'contain',
                }}
              />
              <div
                style={{
                  position: 'absolute',
                  bottom: 12,
                  left: 12,
                  background: 'rgba(10, 15, 29, 0.85)',
                  backdropFilter: 'blur(6px)',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                  borderRadius: 4,
                  padding: '4px 10px',
                  fontFamily: 'monospace',
                  fontSize: 10,
                  color: '#00E5FF',
                  pointerEvents: 'none',
                }}
              >
                2048 × 2048 px Native • No Vector Overlays • No Blur
              </div>
            </div>
          </div>

          {/* Modal Footer Telemetry */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '10px 24px',
              borderTop: '1px solid rgba(255,255,255,0.1)',
              background: 'rgba(15, 23, 42, 0.95)',
              fontSize: 11,
              color: '#94A3B8',
              fontFamily: 'JetBrains Mono, monospace',
            }}
          >
            <div style={{ display: 'flex', gap: 20 }}>
              <span>Center Coords: <strong style={{ color: '#F8FAFC' }}>{scenario ? `${scenario.lat.toFixed(3)}°N, ${scenario.lng.toFixed(3)}°E` : '18.743°N, 71.218°E'}</strong></span>
              <span>Observed Spill Area: <strong style={{ color: '#F59E0B' }}>{scenario?.area || '4.82 km²'}</strong></span>
              <span>Mean Radar Damping: <strong style={{ color: '#10B981' }}>Δσ0 = -8.40 dB</strong></span>
              <span>Optical Sheen: <strong style={{ color: '#00E5FF' }}>Specular Iridescence Present</strong></span>
            </div>
            <div>
              <span>ESA CDSE &amp; ISRO Bhoonidhi Level-1 GRD Processed</span>
            </div>
          </div>
        </div>
      )}
      {/* ICG NOS-DCP CLEAN-UP LOGISTICS & COST OPTIMIZER MODAL */}
      <CleanupModal
        isOpen={isCleanupModalOpen}
        onClose={() => setIsCleanupModalOpen(false)}
        scenario={scenario}
      />
    </div>
  );
};
