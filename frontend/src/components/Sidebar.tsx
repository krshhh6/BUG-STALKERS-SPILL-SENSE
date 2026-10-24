import React, { useState } from 'react';
import type { TabType } from '../types/dashboard';
import type { LiveIncident } from '../hooks/useIncidents';
import { SCENARIOS } from '../data/scenarios';

interface SidebarProps {
  activeTab: TabType;
  onSelectTab: (tab: TabType) => void;
  currentScenarioKey?: string;
  onSelectScenario?: (key: string) => void;
  onOpenSettings?: () => void;
  incidents?: LiveIncident[];
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  onSelectTab,
  currentScenarioKey = '',
  onSelectScenario,
  onOpenSettings: _onOpenSettings,
  incidents,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [isTreeExpanded, setIsTreeExpanded] = useState(true);

  // Use live incidents when available, fall back to static SCENARIOS
  const allEntries: [string, { id: string; title: string; oilType: string }][] =
    incidents && incidents.length > 0
      ? incidents.map((inc) => [inc.id, { id: inc.id, title: inc.title, oilType: inc.oil_type }])
      : Object.entries(SCENARIOS).map(([key, s]) => [key, { id: s.id, title: s.title, oilType: s.oilType }]);

  const filteredScenarios = allEntries.filter(([key, s]) => {
    if (!searchQuery.trim()) return true;
    const q = searchQuery.toLowerCase();
    return key.toLowerCase().includes(q) || s.title.toLowerCase().includes(q) || s.oilType.toLowerCase().includes(q) || s.id.toLowerCase().includes(q);
  });

  const totalCount = allEntries.length;

  const [aisCount, setAisCount] = useState<number | null>(null);

  React.useEffect(() => {
    let mounted = true;
    const fetchAis = async () => {
      try {
        const res = await fetch('/api/v1/ais/stream-status');
        if (res.ok && res.headers.get('content-type')?.includes('application/json')) {
          const data = await res.json();
          if (mounted && data.total_live_vessels_tracked !== undefined) {
            setAisCount(data.total_live_vessels_tracked);
          }
        }
      } catch (e) {
        // quiet fallback
      }
    };
    fetchAis();
    const interval = setInterval(fetchAis, 12000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <aside className="secondary-drawer">
      {/* User Profile Card */}
      <div className="user-profile-card" style={{ marginTop: 14 }}>
        <div className="user-avatar">
          <span className="material-symbols-rounded">security</span>
          <span className="user-status-indicator" />
        </div>
        <div className="user-info">
          <div className="user-name-row">
            <span className="user-name">Duty Officer</span>
            <span className="material-symbols-outlined dropdown-arrow">expand_more</span>
          </div>
          <span className="user-email">c2-watch@icg.gov.in</span>
        </div>
      </div>

      {/* Scrollable Navigation Body */}
      <div className="drawer-scroll-body">
        {/* OPERATIONS / MODULES */}
        <div className="drawer-section">
          <div className="drawer-sec-label">Operations</div>
          <nav className="drawer-nav-list">
            <button
              className={`drawer-nav-item ${activeTab === 'dashboard' ? 'active' : ''}`}
              onClick={() => onSelectTab('dashboard')}
            >
              <div className="drawer-nav-left">
                <span className="material-symbols-outlined nav-icon">dashboard</span>
                <span className="nav-label">Dashboard</span>
              </div>
              <span className="nav-pill-badge">{totalCount}</span>
            </button>

            <button
              className={`drawer-nav-item ${activeTab === 'detection' ? 'active' : ''}`}
              onClick={() => onSelectTab('detection')}
            >
              <div className="drawer-nav-left">
                <span className="material-symbols-outlined nav-icon">science</span>
                <span className="nav-label">SAR Detection Lab</span>
              </div>
            </button>

            <button
              className={`drawer-nav-item ${activeTab === 'drift' ? 'active' : ''}`}
              onClick={() => onSelectTab('drift')}
            >
              <div className="drawer-nav-left">
                <span className="material-symbols-outlined nav-icon">air</span>
                <span className="nav-label">Drift Backtracking</span>
              </div>
            </button>

            <button
              className={`drawer-nav-item ${activeTab === 'attribution' ? 'active' : ''}`}
              onClick={() => onSelectTab('attribution')}
            >
              <div className="drawer-nav-left">
                <span className="material-symbols-outlined nav-icon">directions_boat</span>
                <span className="nav-label">Vessel Attribution</span>
              </div>
            </button>

            <button
              className={`drawer-nav-item ${activeTab === 'evidence' ? 'active' : ''}`}
              onClick={() => onSelectTab('evidence')}
            >
              <div className="drawer-nav-left">
                <span className="material-symbols-outlined nav-icon">verified</span>
                <span className="nav-label">Evidence Center</span>
              </div>
            </button>

            <button
              className={`drawer-nav-item ${activeTab === 'analytics' ? 'active' : ''}`}
              onClick={() => onSelectTab('analytics')}
            >
              <div className="drawer-nav-left">
                <span className="material-symbols-outlined nav-icon">monitoring</span>
                <span className="nav-label">Spill Analytics</span>
              </div>
              <span className="nav-pill-badge">14d</span>
            </button>
          </nav>
        </div>

        {/* STATUS & SENSOR TELEMETRY */}
        <div className="drawer-section">
          <div className="drawer-sec-label">Status & Telemetry</div>
          <div className="drawer-status-list">
            <div className="drawer-status-row">
              <div className="drawer-status-left">
                <span className="status-dot dot-live" />
                <span>Active Slicks</span>
              </div>
              <span className="status-count">{totalCount}</span>
            </div>
            <div className="drawer-status-row">
              <div className="drawer-status-left">
                <span className="status-dot dot-live" />
                <span>AISStream Live</span>
              </div>
              <span className="status-count">{aisCount !== null ? `${aisCount} Live` : 'Connecting...'}</span>
            </div>
            <div className="drawer-status-row">
              <div className="drawer-status-left">
                <span className="status-dot dot-ready" />
                <span>Copernicus SAR</span>
              </div>
              <span className="status-count">Ready</span>
            </div>
          </div>
        </div>

        {/* INCIDENT EXPLORER / DOCUMENTS FOLDER TREE */}
        <div className="drawer-section incident-tree-section">
          <div className="drawer-sec-header">
            <span className="drawer-sec-label">Incident Explorer</span>
            <button
              className="tree-add-btn"
              onClick={() => setIsTreeExpanded(!isTreeExpanded)}
              title={isTreeExpanded ? 'Collapse Folders' : 'Expand Folders'}
            >
              <span className="material-symbols-outlined" style={{ fontSize: 16 }}>
                {isTreeExpanded ? 'remove' : 'add'}
              </span>
            </button>
          </div>

          {/* Search Bar */}
          <div className="tree-search-bar">
            <span className="material-symbols-outlined tree-search-icon">search</span>
            <input
              type="text"
              placeholder="Search incidents, IMO..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="tree-search-input"
            />
          </div>

          {/* Directory Tree */}
          {isTreeExpanded && (
            <div className="tree-container">
              <div
                className={`tree-folder-root ${!currentScenarioKey ? 'selected' : ''}`}
                onClick={() => onSelectScenario && onSelectScenario('')}
              >
                <span className="material-symbols-outlined tree-icon">folder_open</span>
                <span className="tree-label">National Indian Ocean</span>
                <span className="tree-count">{totalCount}</span>
              </div>

              <div className="tree-children">
                {filteredScenarios.map(([key, s]) => {
                  const isSelected = currentScenarioKey === key;
                  return (
                    <div
                      key={key}
                      className={`tree-child-item ${isSelected ? 'selected' : ''}`}
                      onClick={() => onSelectScenario && onSelectScenario(key)}
                    >
                      <span className="material-symbols-outlined tree-icon">
                        {isSelected ? 'folder' : 'folder_open'}
                      </span>
                      <span className="tree-label" title={`${s.id}: ${s.title}`}>
                        {s.title}
                      </span>
                      <span className="tree-count">—</span>
                    </div>
                  );
                })}

                <div className="tree-child-item muted">
                  <span className="material-symbols-outlined tree-icon">folder</span>
                  <span className="tree-label">Off-Grid / Unattributed</span>
                  <span className="tree-count">2</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </aside>
  );
};
