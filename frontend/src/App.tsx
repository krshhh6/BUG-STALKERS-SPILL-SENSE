import React, { useState, useEffect, useMemo } from 'react';
import type { TabType, SarDriftPayload } from './types/dashboard';
import type { LiveSimulationResult } from './services/marineWeatherService';
import { SCENARIOS } from './data/scenarios';
import { useIncidents, type LiveIncident } from './hooks/useIncidents';
import { Topbar } from './components/Topbar';
import { Sidebar } from './components/Sidebar';
import { DashboardView } from './components/views/DashboardView';
import { DriftView } from './components/views/DriftView';
import { AttributionView } from './components/views/AttributionView';
import { EvidenceView } from './components/views/EvidenceView';
import { AnalyticsView } from './components/views/AnalyticsView';
import { DetectionView } from './components/views/DetectionView';
import { ForensicModal } from './components/modals/ForensicModal';
import { SentinelHubModal } from './components/modals/SentinelHubModal';
import { BhoonidhiModal } from './components/modals/BhoonidhiModal';
import {
  searchMaritimeCatalog,
  parseGpsCoordinates,
  type MaritimeSearchResult,
} from './services/maritimeSearchService';
import { MobileNav } from './components/MobileNav';
import { MobileDrawer } from './components/MobileDrawer';
import { OpeningScreen } from './components/OpeningScreen';

export const App: React.FC = () => {
  const [showOpeningScreen, setShowOpeningScreen] = useState<boolean>(true);
  const [activeTab, setActiveTab] = useState<TabType>('dashboard');
  const [currentScenarioKey, setCurrentScenarioKey] = useState<string>('');
  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  const [coordinates, setCoordinates] = useState<string>('15.5000°N, 79.0000°E (Indian Ocean EEZ)');
  const [isForensicOpen, setIsForensicOpen] = useState<boolean>(false);
  const [isSentinelHubOpen, setIsSentinelHubOpen] = useState<boolean>(false);
  const [isBhoonidhiOpen, setIsBhoonidhiOpen] = useState<boolean>(false);
  const [isMapFullscreen, setIsMapFullscreen] = useState<boolean>(false);
  const [targetLocation, setTargetLocation] = useState<{
    lat: number;
    lng: number;
    zoom?: number;
    title: string;
    sub?: string;
    category?: string;
  } | null>(null);
  const [sarDriftPayload, setSarDriftPayload] = useState<SarDriftPayload | null>(null);
  const [activeDriftSimulation, setActiveDriftSimulation] = useState<LiveSimulationResult | null>(null);
  const [isMobileDrawerOpen, setIsMobileDrawerOpen] = useState<boolean>(false);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isMapFullscreen) {
        setIsMapFullscreen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isMapFullscreen]);

  // Live incident data — falls back to static SCENARIOS when backend is offline
  const { incidents, scenarios: liveScenarios } = useIncidents();
  const scenarios = { ...SCENARIOS, ...liveScenarios };

  // Incidents dynamically registered from SAR Detection Lab uploads & inferences
  const [labIncidents, setLabIncidents] = useState<LiveIncident[]>(() => {
    try {
      const stored = localStorage.getItem('SPILL_SENSE_LAB_INCIDENTS');
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  });

  const handleApplyLabDetection = (incident: LiveIncident) => {
    setLabIncidents((prev) => {
      const filtered = prev.filter((i) => i.id !== incident.id);
      const updated = [incident, ...filtered];
      try {
        localStorage.setItem('SPILL_SENSE_LAB_INCIDENTS', JSON.stringify(updated));
      } catch {}
      return updated;
    });
  };

  const allIncidents = useMemo(() => {
    return [...labIncidents, ...incidents];
  }, [labIncidents, incidents]);

  const scenario = currentScenarioKey ? scenarios[currentScenarioKey] || null : null;

  // Apply theme to document element
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === 'light' ? 'dark' : 'light'));
  };

  const handleSelectTab = (tab: TabType) => {
    setActiveTab(tab);
    if (isMapFullscreen) {
      setIsMapFullscreen(false);
    }
  };

  const handleFeedIntoDrift = (payload: SarDriftPayload) => {
    setSarDriftPayload(payload);
    setActiveTab('drift');
    if (isMapFullscreen) {
      setIsMapFullscreen(false);
    }
  };

  const handleSelectScenario = (key: string) => {
    setCurrentScenarioKey(key);
    if (key && scenarios[key]) {
      const s = scenarios[key];
      setCoordinates(`${s.lat.toFixed(4)}°N, ${s.lng.toFixed(4)}°E`);
      setTargetLocation(null);
    } else {
      setCoordinates('15.5000°N, 79.0000°E (Indian Ocean EEZ)');
      setTargetLocation(null);
    }
  };

  const handleSelectSearchResult = (result: MaritimeSearchResult) => {
    setActiveTab('dashboard');
    if (result.scenarioKey && scenarios[result.scenarioKey]) {
      handleSelectScenario(result.scenarioKey);
    } else {
      setCoordinates(`${result.lat.toFixed(4)}°N, ${result.lng.toFixed(4)}°E (${result.title})`);
      setTargetLocation({
        lat: result.lat,
        lng: result.lng,
        zoom: result.zoom || 12,
        title: result.title,
        sub: result.sub,
        category: result.category,
      });
    }
  };

  const handleSearchPlace = (query: string) => {
    const results = searchMaritimeCatalog(query, scenarios);
    if (results.length > 0) {
      handleSelectSearchResult(results[0]);
    } else {
      const coords = parseGpsCoordinates(query);
      if (coords) {
        handleSelectSearchResult({
          id: 'coord-custom',
          title: `GPS: ${coords.lat.toFixed(4)}°N, ${coords.lng.toFixed(4)}°E`,
          category: 'coordinate',
          lat: coords.lat,
          lng: coords.lng,
          zoom: 12,
          sub: 'Direct nautical coordinate inspection',
          badge: 'COORDINATES',
          badgeColor: '#0284C7',
          icon: 'pin_drop',
        });
      } else {
        fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}`)
          .then((res) => res.json())
          .then((data) => {
            if (data && data.length > 0) {
              const item = data[0];
              const lat = parseFloat(item.lat);
              const lng = parseFloat(item.lon);
              handleSelectSearchResult({
                id: `geo-${item.osm_id}`,
                title: item.name || query,
                category: 'external',
                lat,
                lng,
                zoom: 10,
                sub: item.display_name,
                badge: 'GEO SEARCH',
                badgeColor: '#10B981',
                icon: 'public',
              });
            } else {
              alert(`Maritime Intelligence Directory: No port, strait, or vessel found matching "${query}".`);
            }
          })
          .catch(() => {
            alert(`Maritime Intelligence Directory: Could not locate "${query}".`);
          });
      }
    }
  };

  // Keyboard navigation shortcuts (1-6, Escape)
  useEffect(() => {
    const tabs: TabType[] = ['dashboard', 'detection', 'drift', 'attribution', 'evidence', 'analytics'];
    const handleKeyDown = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLSelectElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      // Avoid switching tabs via number keys on mobile touch devices
      if (window.innerWidth >= 1024) {
        const k = parseInt(e.key, 10);
        if (k >= 1 && k <= 6) {
          setActiveTab(tabs[k - 1]);
        }
      }

      if (e.key === 'Escape') {
        setIsForensicOpen(false);
        setIsSentinelHubOpen(false);
        setIsBhoonidhiOpen(false);
        setIsMobileDrawerOpen(false);
      }
    };

    const handleReplayIntro = () => setShowOpeningScreen(true);
    window.addEventListener('spill-sense:replay-intro', handleReplayIntro);

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('spill-sense:replay-intro', handleReplayIntro);
    };
  }, []);

  return (
    <>
      {showOpeningScreen && (
        <OpeningScreen onComplete={() => setShowOpeningScreen(false)} />
      )}
      <div className="app-shell">
      {/* PRIMARY NAVIGATION DRAWER */}
      <Sidebar
        activeTab={activeTab}
        onSelectTab={handleSelectTab}
        currentScenarioKey={currentScenarioKey}
        onSelectScenario={handleSelectScenario}
        onOpenSettings={() => setIsForensicOpen(true)}
        incidents={allIncidents}
      />

      {/* MAIN WORKSPACE CANVAS */}
      <div className={`workspace-container ${isMapFullscreen ? 'map-fullscreen-active' : ''}`}>
        {/* WORKSPACE HEADER */}
        <Topbar
          activeTab={activeTab}
          onSelectTab={handleSelectTab}
          currentScenario={scenario}
          currentScenarioKey={currentScenarioKey}
          onSelectScenario={handleSelectScenario}
          coordinates={coordinates}
          theme={theme}
          onToggleTheme={toggleTheme}
          onOpenForensicModal={() => setIsForensicOpen(true)}
          onOpenSentinelHubModal={() => setIsSentinelHubOpen(true)}
          onOpenBhoonidhiModal={() => setIsBhoonidhiOpen(true)}
          onSearchPlace={handleSearchPlace}
          onSelectSearchResult={handleSelectSearchResult}
          scenarios={scenarios}
        />

        {/* WORKSPACE MAIN VIEW */}
        <main
          className="main"
          id="main-content"
          style={{
            overflow: activeTab === 'dashboard' ? 'hidden' : 'auto',
          }}
        >
          {activeTab === 'dashboard' && (
            <DashboardView
              currentScenario={scenario}
              onSelectTab={handleSelectTab}
              onOpenForensicModal={() => setIsForensicOpen(true)}
              onUpdateCoords={setCoordinates}
              onSelectScenario={handleSelectScenario}
              incidents={allIncidents}
              scenarios={scenarios}
              isFullscreen={isMapFullscreen}
              onToggleFullscreen={() => setIsMapFullscreen((prev) => !prev)}
              targetLocation={targetLocation}
              sarDriftPayload={sarDriftPayload}
              activeDriftSimulation={activeDriftSimulation}
            />
          )}

          {activeTab === 'drift' && (
            <DriftView
              onSelectTab={setActiveTab}
              currentScenario={scenario}
              onSelectScenario={handleSelectScenario}
              sarDriftPayload={sarDriftPayload}
              onDriftSimulationUpdated={setActiveDriftSimulation}
            />
          )}

          {activeTab === 'attribution' && (
            <AttributionView
              currentScenario={scenario}
              onSelectScenario={handleSelectScenario}
              onSelectTab={setActiveTab}
              sarDriftPayload={sarDriftPayload}
              activeDriftSimulation={activeDriftSimulation}
              onInspectVesselOnMap={(v) => {
                setTargetLocation({
                  lat: v.lat,
                  lng: v.lng,
                  zoom: 12,
                  title: v.name,
                  sub: `MMSI: ${v.mmsi} · AIS Suspect Track`,
                  category: 'vessel',
                });
                setActiveTab('dashboard');
              }}
            />
          )}

          {activeTab === 'evidence' && (
            <EvidenceView
              onOpenForensicModal={() => setIsForensicOpen(true)}
              currentScenario={scenario}
              onSelectScenario={handleSelectScenario}
              scenarios={scenarios}
            />
          )}

          {activeTab === 'analytics' && <AnalyticsView incidents={allIncidents} />}

          {activeTab === 'detection' && (
            <DetectionView
              onSelectTab={setActiveTab}
              currentScenario={scenario}
              onApplyLabDetection={handleApplyLabDetection}
              onFeedIntoDrift={handleFeedIntoDrift}
            />
          )}
        </main>
      </div>

      {/* MODALS */}
      <ForensicModal
        isOpen={isForensicOpen}
        onClose={() => setIsForensicOpen(false)}
        scenario={scenario || SCENARIOS['INC-001']}
      />

      <SentinelHubModal
        isOpen={isSentinelHubOpen}
        onClose={() => setIsSentinelHubOpen(false)}
      />

      <BhoonidhiModal
        isOpen={isBhoonidhiOpen}
        onClose={() => setIsBhoonidhiOpen(false)}
        scenario={scenario}
      />

      {/* MOBILE BOTTOM NAVIGATION DOCK (< 1024px) */}
      <MobileNav
        activeTab={activeTab}
        onSelectTab={handleSelectTab}
        isDrawerOpen={isMobileDrawerOpen}
        onToggleDrawer={() => setIsMobileDrawerOpen((prev) => !prev)}
        incidentCount={allIncidents.length}
      />

      {/* MOBILE SLIDE-UP DRAWER (< 1024px) */}
      <MobileDrawer
        isOpen={isMobileDrawerOpen}
        onClose={() => setIsMobileDrawerOpen(false)}
        activeTab={activeTab}
        onSelectTab={handleSelectTab}
        currentScenarioKey={currentScenarioKey}
        onSelectScenario={handleSelectScenario}
        onOpenSettings={() => setIsForensicOpen(true)}
        incidents={allIncidents}
      />
    </div>
    </>
  );
};

export default App;
