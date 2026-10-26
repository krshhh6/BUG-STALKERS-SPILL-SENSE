import React, { useState } from 'react';
import type { Scenario, TabType, SarDriftPayload } from '../../types/dashboard';
import type { LiveSimulationResult } from '../../services/marineWeatherService';
import type { LiveIncident } from '../../hooks/useIncidents';
import { MapPanel } from '../MapPanel';

interface DashboardViewProps {
  currentScenario: Scenario | null;
  onSelectTab: (tab: TabType) => void;
  onOpenForensicModal: () => void;
  onUpdateCoords: (coords: string) => void;
  onSelectScenario?: (key: string) => void;
  incidents?: LiveIncident[];
  scenarios?: Record<string, Scenario>;
  isFullscreen?: boolean;
  onToggleFullscreen?: () => void;
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

export const DashboardView: React.FC<DashboardViewProps> = ({
  currentScenario,
  onSelectTab,
  onOpenForensicModal: _onOpenForensicModal,
  onUpdateCoords,
  onSelectScenario,
  incidents,
  scenarios,
  isFullscreen: externalFullscreen,
  onToggleFullscreen: externalToggleFullscreen,
  targetLocation,
  sarDriftPayload,
  activeDriftSimulation,
}) => {
  const [internalFullscreen, setInternalFullscreen] = useState(false);
  const isFullscreen = externalFullscreen !== undefined ? externalFullscreen : internalFullscreen;
  const toggleFullscreen = externalToggleFullscreen || (() => setInternalFullscreen(!internalFullscreen));

  const activeSlicksCount = incidents && incidents.length > 0 ? incidents.length : 4;

  const statusColor = !currentScenario
    ? '#2563EB'
    : currentScenario.sev.includes('CRITICAL')
    ? '#EF4444'
    : currentScenario.sev.includes('HIGH')
    ? '#F97316'
    : '#F59E0B';

  return (
    <div
      id="tab-dashboard"
      className={`tab-content visible modern-dashboard-root ${isFullscreen ? 'fullscreen-canvas' : ''}`}
    >
      {/* 1. EXECUTIVE HEADER */}
      <div className="workspace-header-bar">
        <div>
          <h1 className="workspace-main-title">Dashboard</h1>
          <p className="workspace-sub-title">All Maritime Workflows, Sensors And Active Spill Incidents</p>
        </div>

        <div className="workspace-header-actions">
          {currentScenario && onSelectScenario && (
            <button
              className="action-pill-btn secondary"
              onClick={() => onSelectScenario('')}
              title="Return to National Indian Ocean Overview"
            >
              <span className="material-symbols-outlined" style={{ fontSize: 15 }}>zoom_out_map</span>
              <span>National Overview</span>
            </button>
          )}

          <button
            className="action-pill-btn primary"
            onClick={() => onSelectTab('detection')}
            title="Open Dual-Pol SAR Detection & Classification Lab"
          >
            <span className="material-symbols-outlined" style={{ fontSize: 15 }}>science</span>
            <span>SAR Detection Lab</span>
          </button>
        </div>
      </div>

      {/* 2. EXECUTIVE METRIC CARDS (Matches reference screenshot 340 +204%) */}
      {!isFullscreen && (
        <div className="executive-metrics-grid">
          {/* Card 1: Active Slicks */}
          <div className="metric-card-neumorphic" onClick={() => onSelectScenario && onSelectScenario('INC-001')}>
            <div className="metric-card-header">
              <span className="metric-card-label">Active Slicks</span>
            </div>
            <div className="metric-card-body">
              <span className="metric-number">{activeSlicksCount}</span>
              <span className="metric-trend-pill positive">↑ 100%</span>
            </div>
            <div className="metric-card-footer">
              <span>See Incidents</span>
              <span className="material-symbols-outlined arrow-icon">arrow_forward</span>
            </div>
          </div>

          {/* Card 2: EEZ Surveillance Area */}
          <div className="metric-card-neumorphic">
            <div className="metric-card-header">
              <span className="metric-card-label">EEZ Surveillance</span>
            </div>
            <div className="metric-card-body">
              <span className="metric-number">2.02M <span className="metric-unit">km²</span></span>
              <span className="metric-trend-pill neutral">100% Active</span>
            </div>
            <div className="metric-card-footer" onClick={() => onSelectTab('analytics')}>
              <span>Sensors & Feeds</span>
              <span className="material-symbols-outlined arrow-icon">arrow_forward</span>
            </div>
          </div>

          {/* Card 3: AI Detection Accuracy */}
          <div className="metric-card-neumorphic" onClick={() => onSelectTab('detection')}>
            <div className="metric-card-header">
              <span className="metric-card-label">AI Detection (Zenodo)</span>
            </div>
            <div className="metric-card-body">
              <span className="metric-number">97.4% <span className="metric-unit">F1</span></span>
              <span className="metric-trend-pill positive">Dual-Pol ONNX</span>
            </div>
            <div className="metric-card-footer">
              <span>SAR Detection Lab</span>
              <span className="material-symbols-outlined arrow-icon">arrow_forward</span>
            </div>
          </div>

          {/* Card 4: Vessel Attribution */}
          <div className="metric-card-neumorphic" onClick={() => onSelectTab('attribution')}>
            <div className="metric-card-header">
              <span className="metric-card-label">Vessel Attribution</span>
            </div>
            <div className="metric-card-body">
              <span className="metric-number">14 <span className="metric-unit">Ships</span></span>
              <span className="metric-trend-pill neutral">AISHub Live</span>
            </div>
            <div className="metric-card-footer">
              <span>View Suspect Vessels</span>
              <span className="material-symbols-outlined arrow-icon">arrow_forward</span>
            </div>
          </div>
        </div>
      )}

      {/* 3. SECTION HEADER & WORKFLOW TABS */}
      <div className="workflow-nav-bar">
        <div className="workflow-title-area">
          <h2 className="workflow-title">
            {currentScenario ? currentScenario.title : 'Maritime Operations & Surveillance'}
          </h2>
          {currentScenario && (
            <span className="scenario-chip" style={{ borderColor: statusColor, color: statusColor }}>
              {currentScenario.id} · {currentScenario.oilType}
            </span>
          )}
        </div>
      </div>

      {/* 4. LARGE ROUNDED CANVAS CONTAINER (Houses MapPanel with floating controls) */}
      <div className="canvas-rounded-container">
        <div className="canvas-map-wrapper">
          <MapPanel
            scenario={currentScenario}
            onUpdateCoords={onUpdateCoords}
            onSelectScenario={onSelectScenario}
            scenarios={scenarios}
            targetLocation={targetLocation}
            sarDriftPayload={sarDriftPayload}
            activeDriftSimulation={activeDriftSimulation}
          />
        </div>

        {/* Floating Bottom-Right Corner Controls (Matches reference screenshot ⤢ and ↻) */}
        <div className="canvas-floating-controls">
          <button
            className="canvas-corner-btn"
            onClick={toggleFullscreen}
            title={isFullscreen ? 'Exit Fullscreen' : 'Expand Fullscreen'}
            aria-label="Toggle Fullscreen"
          >
            <span className="material-symbols-outlined">
              {isFullscreen ? 'fullscreen_exit' : 'open_in_full'}
            </span>
          </button>
          <button
            className="canvas-corner-btn"
            onClick={() => onSelectScenario && onSelectScenario('')}
            title="Recenter Chart Overview"
            aria-label="Recenter"
          >
            <span className="material-symbols-outlined">refresh</span>
          </button>
        </div>
      </div>
    </div>
  );
}; (feat(c2): integrate 3D tactical globe with live AIS tracking, glassmorphism UI, and CDSE setup)
