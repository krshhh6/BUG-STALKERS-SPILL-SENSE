import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import type { LiveSimulationResult } from '../services/marineWeatherService';

interface DriftTrajectoryMapProps {
  simulation: LiveSimulationResult | null;
  scenarioName: string;
  observedCoord: [number, number];
  selectedStepIndex: number;
  activeMode: 'forward' | 'backward' | 'unified';
  onSelectStepIndex?: (index: number) => void;
  suspectVesselName?: string;
  height?: number | string;
}

export const DriftTrajectoryMap: React.FC<DriftTrajectoryMapProps> = ({
  simulation,
  scenarioName,
  observedCoord,
  selectedStepIndex,
  activeMode,
  onSelectStepIndex,
  suspectVesselName = 'CRUDE ATLAS',
  height = 420,
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<L.Map | null>(null);
  const layersRef = useRef<L.LayerGroup | null>(null);

  // Initialize Leaflet map once
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = L.map(containerRef.current, {
      center: observedCoord,
      zoom: 10,
      zoomControl: false,
      attributionControl: false,
    });

    L.control.zoom({ position: 'topleft' }).addTo(map);

    // High performance CartoDB Dark Matter basemap
    L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/dark_all/{z}/{x}/{y}{r}.png', {
      maxZoom: 18,
      subdomains: 'abcd',
    }).addTo(map);

    const layerGroup = L.layerGroup().addTo(map);
    layersRef.current = layerGroup;
    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // Update layers whenever simulation, mode, or selectedStepIndex changes
  useEffect(() => {
    const map = mapRef.current;
    const group = layersRef.current;
    if (!map || !group || !simulation) return;

    group.clearLayers();

    const [t0Lat, t0Lng] = observedCoord;
    const allPointsForBounds: [number, number][] = [[t0Lat, t0Lng]];

    // ── 1. T0 OBSERVED SLICK MARKER ──
    const t0PulseIcon = L.divIcon({
      className: 't0-drift-marker',
      html: `
        <div style="position: relative; width: 20px; height: 20px; transform: translate(-50%, -50%);">
          <div style="position: absolute; inset: 0; border-radius: 50%; background: rgba(56, 189, 248, 0.35); animation: pulse 2s infinite;"></div>
          <div style="position: absolute; inset: 4px; border-radius: 50%; background: #0284c7; border: 2px solid #ffffff; box-shadow: 0 0 10px #38bdf8;"></div>
        </div>
      `,
      iconSize: [0, 0],
      iconAnchor: [0, 0],
    });

    const t0Marker = L.marker([t0Lat, t0Lng], { icon: t0PulseIcon })
      .bindTooltip(`<b>Observed Slick (T0)</b><br>${scenarioName}<br>${t0Lat.toFixed(4)}°N, ${t0Lng.toFixed(4)}°E`, {
        sticky: true,
        className: 'gis-custom-tooltip',
      });
    group.addLayer(t0Marker);

    // ── 2. BACKWARD ORIGIN RECONSTRUCTION ──
    if (activeMode === 'backward' || activeMode === 'unified') {
      // 90%, 75%, 50% Probability Isobars
      if (simulation.envelope90 && simulation.envelope90.length > 0) {
        const poly90 = L.polygon(simulation.envelope90, {
          color: '#B45309',
          weight: 1.2,
          dashArray: '4, 4',
          fillColor: '#B45309',
          fillOpacity: 0.10,
        }).bindTooltip('<b>90% Probability Origin Envelope</b><br>Monte Carlo Reverse Dispersion', {
          sticky: true,
          className: 'gis-custom-tooltip',
        });
        group.addLayer(poly90);
        simulation.envelope90.forEach((p) => allPointsForBounds.push(p));
      }

      if (simulation.envelope75 && simulation.envelope75.length > 0) {
        const poly75 = L.polygon(simulation.envelope75, {
          color: '#D97706',
          weight: 1.5,
          fillColor: '#D97706',
          fillOpacity: 0.20,
        }).bindTooltip('<b>75% Probability Area</b><br>Reverse Advection-Diffusion Isobar', {
          sticky: true,
          className: 'gis-custom-tooltip',
        });
        group.addLayer(poly75);
      }

      if (simulation.envelope50 && simulation.envelope50.length > 0) {
        const poly50 = L.polygon(simulation.envelope50, {
          color: '#F59E0B',
          weight: 2.0,
          fillColor: '#F59E0B',
          fillOpacity: 0.35,
        }).bindTooltip('<b>50% Core Probability Origin</b><br>High-likelihood discharge envelope', {
          sticky: true,
          className: 'gis-custom-tooltip',
        });
        group.addLayer(poly50);
      }

      // Backward Lagrangian Trajectory Line
      if (simulation.backtrackPath && simulation.backtrackPath.length > 1) {
        const backtrackLine = L.polyline(simulation.backtrackPath, {
          color: '#F59E0B',
          weight: 3.5,
          dashArray: '6, 6',
          opacity: 0.95,
        }).bindTooltip('<b>⏪ Backward Lagrangian Trajectory (T0 → T-24h)</b><br>Runge-Kutta 4th Order with Coriolis & Tidal Oscillation', {
          sticky: true,
          className: 'gis-custom-tooltip',
        });
        group.addLayer(backtrackLine);
        simulation.backtrackPath.forEach((p) => allPointsForBounds.push(p));
      }

      // Origin Pin Marker (T-24h)
      const originCoord = simulation.originCoordNumeric;
      if (originCoord) {
        const originIcon = L.divIcon({
          className: 'origin-drift-marker',
          html: `
            <div style="position: relative; width: 22px; height: 22px; transform: translate(-50%, -50%);">
              <div style="position: absolute; inset: 0; border-radius: 50%; background: rgba(245, 158, 11, 0.4); animation: pulse 1.5s infinite;"></div>
              <div style="position: absolute; inset: 3px; border-radius: 50%; background: #d97706; border: 2px solid #ffffff; box-shadow: 0 0 12px #f59e0b; display: flex; align-items: center; justify-content: center;">
                <span style="font-size: 9px; font-weight: 900; color: #ffffff;">T</span>
              </div>
            </div>
          `,
          iconSize: [0, 0],
          iconAnchor: [0, 0],
        });

        const originMarker = L.marker(originCoord, { icon: originIcon })
          .bindPopup(`
            <div style="font-family: inherit; font-size: 12px; color: #0f172a; padding: 4px;">
              <strong style="color: #d97706; font-size: 13px;">DISCHARGE ORIGIN (T - 24h)</strong><br>
              <strong>Coordinates:</strong> ${originCoord[0].toFixed(4)}°N, ${originCoord[1].toFixed(4)}°E<br>
              <strong>Spatiotemporal Window:</strong> ${simulation.originWindow}<br>
              <strong>Suspect Tanker:</strong> ${suspectVesselName}
            </div>
          `);
        group.addLayer(originMarker);
        allPointsForBounds.push(originCoord);
      }

      // Offending Vessel Track & Waypoints
      if (simulation.vesselTrack && simulation.vesselTrack.length > 1) {
        const vesselTrackLine = L.polyline(simulation.vesselTrack, {
          color: '#EF4444',
          weight: 2.5,
          dashArray: '5, 4',
          opacity: 0.85,
        }).bindTooltip(`<b>AIS Suspect Track: ${suspectVesselName}</b><br>Passes directly through T-22h discharge window`, {
          sticky: true,
          className: 'gis-custom-tooltip',
        });
        group.addLayer(vesselTrackLine);
        simulation.vesselTrack.forEach((p) => allPointsForBounds.push(p));
      }
    }

    // ── 3. FORWARD FORECAST DISPERSION CONES & TRAJECTORY ──
    if (activeMode === 'forward' || activeMode === 'unified') {
      if (simulation.forwardCone24h && simulation.forwardCone24h.length > 0) {
        const cone24 = L.polygon(simulation.forwardCone24h, {
          color: '#10B981',
          weight: 1.2,
          dashArray: '4, 4',
          fillColor: '#10B981',
          fillOpacity: 0.10,
        }).bindTooltip('<b>T+24h Predicted Spill Dispersion Cone</b>', {
          sticky: true,
          className: 'gis-custom-tooltip',
        });
        group.addLayer(cone24);
        simulation.forwardCone24h.forEach((p) => allPointsForBounds.push(p));
      }

      if (simulation.forwardCone12h && simulation.forwardCone12h.length > 0) {
        const cone12 = L.polygon(simulation.forwardCone12h, {
          color: '#10B981',
          weight: 1.6,
          fillColor: '#10B981',
          fillOpacity: 0.18,
        }).bindTooltip('<b>T+12h Predicted Core Impact Zone</b>', {
          sticky: true,
          className: 'gis-custom-tooltip',
        });
        group.addLayer(cone12);
      }

      if (simulation.forwardDriftPath && simulation.forwardDriftPath.length > 1) {
        const forwardPathLine = L.polyline(simulation.forwardDriftPath, {
          color: '#10B981',
          weight: 3.2,
          dashArray: '5, 5',
          opacity: 0.95,
        }).bindTooltip('<b>⏩ Forward Hydrodynamic Drift Forecast (T0 → T+24h)</b>', {
          sticky: true,
          className: 'gis-custom-tooltip',
        });
        group.addLayer(forwardPathLine);
        simulation.forwardDriftPath.forEach((p) => allPointsForBounds.push(p));
      }

      // Forecast step waypoint dots
      if (simulation.forecasts && simulation.forecasts.length > 0) {
        simulation.forecasts.forEach((f, idx) => {
          const isSelected = idx === selectedStepIndex;
          const threatColor =
            f.threatLevel === 'CRITICAL' ? '#EF4444' : f.threatLevel === 'HIGH' ? '#F97316' : '#10B981';

          const stepDotIcon = L.divIcon({
            className: 'drift-step-dot',
            html: `
              <div style="
                width: ${isSelected ? '14px' : '8px'};
                height: ${isSelected ? '14px' : '8px'};
                border-radius: 50%;
                background: ${threatColor};
                border: 2px solid #ffffff;
                box-shadow: 0 0 ${isSelected ? '12px' : '4px'} ${threatColor};
                transform: translate(-50%, -50%);
                cursor: pointer;
              "></div>
            `,
            iconSize: [0, 0],
            iconAnchor: [0, 0],
          });

          const stepMarker = L.marker([f.lat, f.lng], { icon: stepDotIcon })
            .bindTooltip(`<b>${f.label}</b> (${f.timestamp})<br>Area: ${f.areaKm2} km² · Shore dist: ${f.distanceToCoastKm} km<br>Threat: <span style="color:${threatColor};font-weight:700;">${f.threatLevel}</span>`, {
              sticky: true,
              className: 'gis-custom-tooltip',
            });

          stepMarker.on('click', () => {
            if (onSelectStepIndex) onSelectStepIndex(idx);
          });

          group.addLayer(stepMarker);
          allPointsForBounds.push([f.lat, f.lng]);
        });
      }
    }

    // ── 4. ACTIVE TIMESTEP FOCUS RING ──
    const activeForecast = simulation.forecasts?.[selectedStepIndex];
    if (activeForecast && (activeMode === 'forward' || activeMode === 'unified')) {
      const activeRingIcon = L.divIcon({
        className: 'active-drift-focus-ring',
        html: `
          <div style="position: relative; width: 34px; height: 34px; transform: translate(-50%, -50%);">
            <div style="position: absolute; inset: 0; border-radius: 50%; border: 2px dashed #10b981; animation: spin 6s linear infinite;"></div>
            <div style="position: absolute; inset: 8px; border-radius: 50%; background: rgba(16, 185, 129, 0.4); border: 2px solid #ffffff;"></div>
          </div>
        `,
        iconSize: [0, 0],
        iconAnchor: [0, 0],
      });

      const activeRing = L.marker([activeForecast.lat, activeForecast.lng], {
        icon: activeRingIcon,
        zIndexOffset: 1000,
      }).bindTooltip(
        `<b>ACTIVE HORIZON: ${activeForecast.label}</b><br>Threat: ${activeForecast.threatLevel}<br>Area: ${activeForecast.areaKm2} km²`,
        { permanent: false, sticky: true, className: 'gis-custom-tooltip' }
      );
      group.addLayer(activeRing);
    }

    // Fit map bounds to encompass all trajectory features
    if (allPointsForBounds.length > 1) {
      const bounds = L.latLngBounds(allPointsForBounds);
      map.fitBounds(bounds, { padding: [40, 40], maxZoom: 12 });
    }
  }, [simulation, activeMode, selectedStepIndex, observedCoord, scenarioName, suspectVesselName, onSelectStepIndex]);

  const handleZoomFull = () => {
    if (!mapRef.current || !simulation) return;
    const pts: [number, number][] = [observedCoord];
    if (simulation.originCoordNumeric) pts.push(simulation.originCoordNumeric);
    if (simulation.predictedCoord24h) pts.push(simulation.predictedCoord24h);
    mapRef.current.fitBounds(L.latLngBounds(pts), { padding: [40, 40], maxZoom: 12 });
  };

  const handleFocusOrigin = () => {
    if (!mapRef.current || !simulation?.originCoordNumeric) return;
    mapRef.current.setView(simulation.originCoordNumeric, 12, { animate: true });
  };

  const handleFocusT0 = () => {
    if (!mapRef.current) return;
    mapRef.current.setView(observedCoord, 12, { animate: true });
  };

  return (
    <div
      style={{
        position: 'relative',
        width: '100%',
        height: typeof height === 'number' ? `${height}px` : height,
        borderRadius: 12,
        overflow: 'hidden',
        border: '1px solid var(--border-subtle)',
        boxShadow: '0 4px 16px rgba(0, 0, 0, 0.25)',
        background: '#0d1117',
        marginBottom: 16,
      }}
    >
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />

      {/* FLOATING ACTION TOOLBAR */}
      <div
        style={{
          position: 'absolute',
          top: 12,
          right: 12,
          zIndex: 1000,
          display: 'flex',
          gap: 6,
          background: 'rgba(15, 23, 42, 0.85)',
          backdropFilter: 'blur(8px)',
          WebkitBackdropFilter: 'blur(8px)',
          padding: '4px 6px',
          borderRadius: 8,
          border: '1px solid rgba(255, 255, 255, 0.12)',
        }}
      >
        <button
          onClick={handleZoomFull}
          title="Zoom to Full Extent"
          style={{
            background: 'transparent',
            border: 'none',
            color: '#94a3b8',
            cursor: 'pointer',
            padding: '4px 8px',
            borderRadius: 6,
            fontSize: 11,
            fontWeight: 700,
            display: 'flex',
            alignItems: 'center',
            gap: 4,
          }}
          onMouseEnter={(e) => (e.currentTarget.style.color = '#ffffff')}
          onMouseLeave={(e) => (e.currentTarget.style.color = '#94a3b8')}
        >
          <span className="material-symbols-outlined" style={{ fontSize: 16 }}>fit_screen</span>
          <span>Full Extent</span>
        </button>

        <button
          onClick={handleFocusOrigin}
          title="Focus on T-24h Discharge Origin"
          style={{
            background: 'rgba(245, 158, 11, 0.15)',
            border: '1px solid rgba(245, 158, 11, 0.4)',
            color: '#f59e0b',
            cursor: 'pointer',
            padding: '4px 8px',
            borderRadius: 6,
            fontSize: 11,
            fontWeight: 700,
            display: 'flex',
            alignItems: 'center',
            gap: 4,
          }}
        >
          <span className="material-symbols-outlined" style={{ fontSize: 16 }}>history</span>
          <span>Origin (T-24h)</span>
        </button>

        <button
          onClick={handleFocusT0}
          title="Focus on Observed Slick (T0)"
          style={{
            background: 'rgba(56, 189, 248, 0.15)',
            border: '1px solid rgba(56, 189, 248, 0.4)',
            color: '#38bdf8',
            cursor: 'pointer',
            padding: '4px 8px',
            borderRadius: 6,
            fontSize: 11,
            fontWeight: 700,
            display: 'flex',
            alignItems: 'center',
            gap: 4,
          }}
        >
          <span className="material-symbols-outlined" style={{ fontSize: 16 }}>adjust</span>
          <span>Slick (T0)</span>
        </button>
      </div>

      {/* FLOATING HUD LEGEND */}
      <div
        style={{
          position: 'absolute',
          bottom: 12,
          left: 12,
          zIndex: 1000,
          background: 'rgba(15, 23, 42, 0.88)',
          backdropFilter: 'blur(8px)',
          WebkitBackdropFilter: 'blur(8px)',
          padding: '8px 12px',
          borderRadius: 8,
          border: '1px solid rgba(255, 255, 255, 0.12)',
          fontSize: 11,
          display: 'flex',
          flexDirection: 'column',
          gap: 5,
        }}
      >
        <div style={{ fontWeight: 800, fontSize: 11.5, color: '#f8fafc', marginBottom: 2 }}>
          Drift Hydrodynamic Layers
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#cbd5e1' }}>
          <span style={{ width: 9, height: 9, borderRadius: '50%', background: '#38bdf8', boxShadow: '0 0 6px #38bdf8' }} />
          <span>Observed Slick (T0)</span>
        </div>
        {(activeMode === 'backward' || activeMode === 'unified') && (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#cbd5e1' }}>
              <span style={{ width: 14, height: 3, background: '#f59e0b', borderRadius: 2 }} />
              <span>Lagrangian Backtrack (T-24h)</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#cbd5e1' }}>
              <span style={{ width: 9, height: 9, borderRadius: 2, background: 'rgba(245, 158, 11, 0.4)', border: '1px solid #f59e0b' }} />
              <span>50% / 75% / 90% Origin Envelopes</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#cbd5e1' }}>
              <span style={{ width: 14, height: 2, background: '#ef4444', borderTop: '1px dashed #ef4444' }} />
              <span>Attributed Vessel Track</span>
            </div>
          </>
        )}
        {(activeMode === 'forward' || activeMode === 'unified') && (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#cbd5e1' }}>
              <span style={{ width: 14, height: 3, background: '#10b981', borderRadius: 2 }} />
              <span>Forward Drift Forecast (T+24h)</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#cbd5e1' }}>
              <span style={{ width: 9, height: 9, borderRadius: 2, background: 'rgba(16, 185, 129, 0.3)', border: '1px solid #10b981' }} />
              <span>Dispersion Cones (T+12h / T+24h)</span>
            </div>
          </>
        )}
      </div>
    </div>
  );
};
