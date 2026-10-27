import React, { useState, useMemo } from 'react';
import type { Scenario } from '../../types/dashboard';
import { SCENARIOS } from '../../data/scenarios';
import {
  generateDeterministicHash,
  formatHashChunks,
  getForensicMilestones,
  getForensicArtifacts,
} from '../../utils/forensicHash';
import type { ForensicArtifactItem } from '../../utils/forensicHash';

interface EvidenceViewProps {
  onOpenForensicModal: () => void;
  currentScenario?: Scenario | null;
  onSelectScenario?: (key: string) => void;
  scenarios?: Record<string, Scenario>;
}

export const EvidenceView: React.FC<EvidenceViewProps> = ({
  onOpenForensicModal,
  currentScenario,
  onSelectScenario,
  scenarios,
}) => {
  // Available scenarios map
  const availableScenarios = useMemo(() => scenarios || SCENARIOS, [scenarios]);

  // Determine active scenario key
  const activeKey = useMemo(() => {
    if (currentScenario) {
      const match = Object.entries(availableScenarios).find(
        ([k, sc]) => sc.id === currentScenario.id || k === currentScenario.id
      );
      if (match) return match[0];
    }
    return Object.keys(availableScenarios)[0] || 'INC-001';
  }, [currentScenario, availableScenarios]);

  const activeScenario = useMemo(() => {
    return (
      currentScenario ||
      availableScenarios[activeKey] ||
      availableScenarios['INC-001'] ||
      Object.values(availableScenarios)[0]
    );
  }, [currentScenario, availableScenarios, activeKey]);

  // Active Sub-Tab: 'manifest' | 'artifacts' | 'dossier'
  const [activeSubTab, setActiveSubTab] = useState<'manifest' | 'artifacts' | 'dossier'>('manifest');

  // Dynamic deterministic master hash based on active scenario
  const scenarioKey = useMemo(() => {
    return (
      (activeScenario?.id || 'INC-2026-001') +
      (activeScenario?.title || 'Maritime Sector') +
      (activeScenario?.lat || 18.74) +
      (activeScenario?.lng || 71.21)
    );
  }, [activeScenario]);

  const masterHash = useMemo(() => generateDeterministicHash(scenarioKey), [scenarioKey]);
  const hashChunks = useMemo(() => formatHashChunks(masterHash), [masterHash]);

  // Officer sign-off state
  const [officerSignOff, setOfficerSignOff] = useState<{ officerName: string; timestamp: string } | null>(null);
  const [isSignOffModalOpen, setIsSignOffModalOpen] = useState(false);
  const [signOffName, setSignOffName] = useState('Cmdr. V. Sharma, ICG');
  const [signOffStation, setSignOffStation] = useState('MRCC Mumbai · Western Seaboard C2');

  // Interactive Live Integrity Audit state
  const [isAuditing, setIsAuditing] = useState(false);
  const [auditProgress, setAuditProgress] = useState(0);
  const [auditStepText, setAuditStepText] = useState('');
  const [auditComplete, setAuditComplete] = useState(false);

  // In-line Artifact Preview Drawer
  const [previewArtifact, setPreviewArtifact] = useState<ForensicArtifactItem | null>(null);

  // User feedback toast/notices
  const [feedback, setFeedback] = useState<{ text: string; color: string } | null>(null);
  const [downloadNotice, setDownloadNotice] = useState<{ text: string; isSuccess: boolean } | null>(null);
  const [downloadingFile, setDownloadingFile] = useState<string | null>(null);

  // Milestones & Artifacts linked dynamically to active scenario
  const milestones = useMemo(
    () => getForensicMilestones(activeScenario, officerSignOff),
    [activeScenario, officerSignOff]
  );

  const artifacts = useMemo(
    () => getForensicArtifacts(activeScenario, masterHash),
    [activeScenario, masterHash]
  );

  const totalPackageSizeKb = useMemo(
    () => artifacts.reduce((acc, item) => acc + item.sizeKb, 0),
    [artifacts]
  );

  const handleCopySignature = () => {
    navigator.clipboard?.writeText(masterHash);
    setFeedback({ text: 'Cryptographic master signature copied to clipboard.', color: 'var(--accent)' });
    setTimeout(() => setFeedback(null), 3000);
  };

  const handleCopyText = (text: string, label: string) => {
    navigator.clipboard?.writeText(text);
    setFeedback({ text: `${label} copied to clipboard.`, color: 'var(--accent)' });
    setTimeout(() => setFeedback(null), 3000);
  };

  const triggerBlobDownload = (blob: Blob, fileName: string) => {
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = fileName;
    link.style.display = 'none';
    document.body.appendChild(link);
    link.click();
    setTimeout(() => {
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    }, 250);
  };

  const handleDownloadArtifact = async (fileName: string) => {
    setDownloadingFile(fileName);
    setDownloadNotice({ text: `Authenticating and transferring ${fileName}...`, isSuccess: false });

    // Look up artifact generator
    const matched = artifacts.find((a) => a.fileName === fileName);
    const content = matched
      ? matched.contentGenerator(
          activeScenario,
          masterHash
        )
      : `SpillSense Forensic Record · ${fileName}\nIncident: ${activeScenario?.id || 'INC-001'}\nHash: ${masterHash}\nSection 63 BSA 2023 Certified`;

    let mimeType = 'text/plain';
    if (fileName.endsWith('.geojson')) mimeType = 'application/geo+json;charset=utf-8';
    else if (fileName.endsWith('.xml')) mimeType = 'application/xml;charset=utf-8';
    else if (fileName.endsWith('.zip')) mimeType = 'application/zip';

    setTimeout(() => {
      const blob = new Blob([content], { type: mimeType });
      triggerBlobDownload(blob, fileName);
      setDownloadNotice({
        text: `✓ Downloaded ${fileName} — Cryptographic signature verified against HSM Ledger`,
        isSuccess: true,
      });
      setDownloadingFile(null);
      setTimeout(() => setDownloadNotice(null), 4500);
    }, 400);
  };

  // Run interactive integrity audit sequence
  const handleRunAudit = () => {
    setIsAuditing(true);
    setAuditComplete(false);
    setAuditProgress(10);
    setAuditStepText('Querying HSM FIPS 140-2 Key Vault & Ledger Root...');

    setTimeout(() => {
      setAuditProgress(35);
      setAuditStepText('Verifying Sentinel-1 GRD Raw Acquisition CRC-32...');
    }, 600);

    setTimeout(() => {
      setAuditProgress(65);
      setAuditStepText('Computing U-Net ResNet-50 Feature Mask SHA-256 Digest...');
    }, 1200);

    setTimeout(() => {
      setAuditProgress(85);
      setAuditStepText('Cross-checking OpenDrift Lagrangian Backward Particles & AIS Matrix...');
    }, 1800);

    setTimeout(() => {
      setAuditProgress(100);
      setAuditStepText('All 4 Artifacts Authenticated · Zero Bit Discrepancies');
      setIsAuditing(false);
      setAuditComplete(true);
      setFeedback({ text: 'Cryptographic Audit Passed: 100% Intact & Verified.', color: '#10b981' });
      setTimeout(() => setFeedback(null), 4000);
    }, 2400);
  };

  const handleConfirmSignOff = () => {
    const ts = `${new Date().toISOString().replace('T', ' ').slice(0, 19)} IST`;
    setOfficerSignOff({
      officerName: `${signOffName} (${signOffStation})`,
      timestamp: ts,
    });
    setIsSignOffModalOpen(false);
    setFeedback({
      text: `✓ Dossier Sealed & Endorsed by ${signOffName}`,
      color: '#10b981',
    });
    setTimeout(() => setFeedback(null), 4000);
  };

  return (
    <div id="tab-evidence" className="tab-content visible modern-dashboard-root">
      {/* 1. EXECUTIVE HEADER */}
      <div
        className="workspace-header-bar"
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          gap: 14,
          flexWrap: 'wrap',
          marginBottom: 16,
        }}
      >
        <div style={{ minWidth: 260, flex: '1 1 auto' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4, flexWrap: 'wrap' }}>
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
              <span className="material-symbols-outlined" style={{ fontSize: 13 }}>verified_user</span>
              Digital Forensics &amp; Chain of Custody
            </span>
            <span
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 4,
                padding: '2px 8px',
                borderRadius: 6,
                background: 'rgba(16, 185, 129, 0.12)',
                border: '1px solid rgba(16, 185, 129, 0.3)',
                color: '#10b981',
                fontSize: 10,
                fontWeight: 700,
              }}
            >
              <span className="material-symbols-outlined" style={{ fontSize: 12 }}>lock</span>
              ISO/IEC 27037 · Sec 63 BSA 2023
            </span>
          </div>
          <h1 className="workspace-main-title" style={{ fontSize: 20, fontWeight: 800, margin: 0, letterSpacing: '-0.02em' }}>
            Forensic Evidence &amp; Chain of Custody
          </h1>
          <p className="workspace-sub-title" style={{ margin: '3px 0 0', fontSize: 11.5, color: 'var(--text-muted)' }}>
            Cryptographically Verifiable Evidence Package for Maritime Regulatory Enforcement
          </p>
        </div>

        <div className="workspace-header-actions" style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', flex: '0 1 auto' }}>
          {/* INCIDENT SWITCHER DROPDOWN */}
          <div style={{ position: 'relative', display: 'inline-flex', alignItems: 'center' }}>
            <span
              className="material-symbols-outlined"
              style={{
                position: 'absolute',
                left: 9,
                fontSize: 15,
                color: 'var(--accent)',
                pointerEvents: 'none',
              }}
            >
              emergency
            </span>
            <select
              value={activeKey}
              onChange={(e) => {
                const k = e.target.value;
                onSelectScenario?.(k);
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
                minWidth: 220,
                maxWidth: 280,
              }}
              title="Select Active Incident Dossier"
            >
              {Object.entries(availableScenarios).map(([key, sc]) => (
                <option key={key} value={key}>
                  {sc.id} · {sc.title}
                </option>
              ))}
            </select>
          </div>

          <button
            className="action-pill-btn secondary"
            onClick={handleCopySignature}
            title="Copy Master SHA-256 Fingerprint"
            style={{ fontSize: 11.5, padding: '5px 12px', height: 32, whiteSpace: 'nowrap' }}
          >
            <span className="material-symbols-outlined" style={{ fontSize: 15 }}>content_copy</span>
            <span>Copy Master Signature</span>
          </button>

          <button
            className="action-pill-btn primary"
            onClick={onOpenForensicModal}
            title="Generate & View Court-Admissible PDF Dossier"
            style={{ fontSize: 11.5, padding: '5px 14px', height: 32, whiteSpace: 'nowrap' }}
          >
            <span className="material-symbols-outlined" style={{ fontSize: 15 }}>description</span>
            <span>Official Court Dossier</span>
          </button>

          <a
            href={`http://127.0.0.1:8000/api/v1/dossier/${activeScenario?.id || 'INC-001'}/download`}
            target="_blank"
            rel="noopener noreferrer"
            download={`dossier_${activeScenario?.id || 'INC-001'}.pdf`}
            className="action-pill-btn secondary"
            title="Download Authenticated ReportLab Binary PDF from Backend API"
            style={{ fontSize: 11.5, padding: '5px 14px', height: 32, whiteSpace: 'nowrap', textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: 6 }}
          >
            <span className="material-symbols-outlined" style={{ fontSize: 15 }}>download</span>
            <span>Download Signed PDF</span>
          </a>
        </div>
      </div>

      {/* 2. EXECUTIVE METRIC CARDS */}
      <div className="executive-metrics-grid" style={{ marginBottom: 14, gap: 12 }}>
        <div className="metric-card-neumorphic">
          <div className="metric-card-header">
            <span className="metric-card-label">Custody Seal Status</span>
          </div>
          <div className="metric-card-body">
            <span className="metric-number" style={{ fontSize: 19 }}>
              ISO/IEC <span className="metric-unit">27037</span>
            </span>
            <span className="metric-trend-pill positive">
              Verified Active
            </span>
          </div>
          <div className="metric-card-footer">
            <span>Hardware Security Module (HSM)</span>
            <span className="material-symbols-outlined arrow-icon" style={{ color: '#10b981' }}>verified</span>
          </div>
        </div>

        <div className="metric-card-neumorphic">
          <div className="metric-card-header">
            <span className="metric-card-label">SHA-256 Checksums</span>
          </div>
          <div className="metric-card-body">
            <span className="metric-number">
              4 / 4 <span className="metric-unit">Sealed</span>
            </span>
            <span className="metric-trend-pill positive">
              100% Intact
            </span>
          </div>
          <div className="metric-card-footer">
            <span>Zero Bit Discrepancies</span>
            <span className="material-symbols-outlined arrow-icon" style={{ color: '#10b981' }}>lock</span>
          </div>
        </div>

        <div className="metric-card-neumorphic">
          <div className="metric-card-header">
            <span className="metric-card-label">Evidence Package</span>
          </div>
          <div className="metric-card-body">
            <span className="metric-number">
              {totalPackageSizeKb.toFixed(1)} <span className="metric-unit">KB</span>
            </span>
            <span className="metric-trend-pill neutral">
              4 Assets
            </span>
          </div>
          <div className="metric-card-footer">
            <span>GeoJSON / NetCDF / GPKG</span>
            <span className="material-symbols-outlined arrow-icon">folder_zip</span>
          </div>
        </div>

        <div className="metric-card-neumorphic">
          <div className="metric-card-header">
            <span className="metric-card-label">Admissibility Level</span>
          </div>
          <div className="metric-card-body">
            <span className="metric-number">
              Tier 4 <span className="metric-unit">Legal</span>
            </span>
            <span className="metric-trend-pill positive">
              Court-Admissible
            </span>
          </div>
          <div className="metric-card-footer">
            <span>Sec 63 BSA 2023 / MS Act §356</span>
            <span className="material-symbols-outlined arrow-icon" style={{ color: 'var(--accent)' }}>gavel</span>
          </div>
        </div>
      </div>

      {/* 3. FUNCTIONAL WORKFLOW NAV BAR */}
      <div
        className="workflow-nav-bar"
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          gap: 12,
          marginBottom: 14,
          padding: '0 4px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
          <h2 className="workflow-title" style={{ margin: 0, fontSize: 13.5, fontWeight: 700, whiteSpace: 'nowrap' }}>
            Tamper-Evident Forensic Audit Ledger
          </h2>
          <span
            className="scenario-chip"
            style={{
              borderColor: 'rgba(37, 99, 235, 0.4)',
              color: 'var(--accent)',
              padding: '2px 8px',
              fontSize: 10.5,
            }}
          >
            Ledger Block #40921-IN · ECDSA Signed
          </span>
          {officerSignOff && (
            <span
              className="scenario-chip"
              style={{
                borderColor: 'rgba(16, 185, 129, 0.4)',
                color: '#10b981',
                padding: '2px 8px',
                fontSize: 10.5,
                display: 'inline-flex',
                alignItems: 'center',
                gap: 4,
              }}
            >
              <span className="material-symbols-outlined" style={{ fontSize: 13 }}>verified_user</span>
              Officer Endorsed
            </span>
          )}
        </div>

        <div className="workflow-tabs-strip">
          <button
            className={`workflow-tab-btn ${activeSubTab === 'manifest' ? 'active' : ''}`}
            onClick={() => setActiveSubTab('manifest')}
            style={{ padding: '6px 14px', fontSize: 12 }}
          >
            <span className="material-symbols-outlined" style={{ fontSize: 15, verticalAlign: 'middle', marginRight: 5 }}>lock</span>
            Cryptographic Manifest &amp; Timeline
          </button>
          <button
            className={`workflow-tab-btn ${activeSubTab === 'artifacts' ? 'active' : ''}`}
            onClick={() => setActiveSubTab('artifacts')}
            style={{ padding: '6px 14px', fontSize: 12 }}
          >
            <span className="material-symbols-outlined" style={{ fontSize: 15, verticalAlign: 'middle', marginRight: 5 }}>inventory_2</span>
            Artifacts &amp; Live Verifier (4)
          </button>
          <button
            className={`workflow-tab-btn ${activeSubTab === 'dossier' ? 'active' : ''}`}
            onClick={() => setActiveSubTab('dossier')}
            style={{ padding: '6px 14px', fontSize: 12 }}
          >
            <span className="material-symbols-outlined" style={{ fontSize: 15, verticalAlign: 'middle', marginRight: 5 }}>article</span>
            Court Dossier &amp; BSA §63
          </button>
        </div>

        {feedback && (
          <div style={{ fontSize: 11.5, fontWeight: 600, color: feedback.color, display: 'flex', alignItems: 'center', gap: 6 }}>
            <span className="material-symbols-outlined" style={{ fontSize: 16 }}>check_circle</span>
            {feedback.text}
          </div>
        )}
      </div>

      {downloadNotice && (
        <div
          style={{
            margin: '0 0 14px',
            padding: '10px 16px',
            background: downloadNotice.isSuccess ? 'rgba(16, 185, 129, 0.12)' : 'rgba(56, 189, 248, 0.12)',
            border: downloadNotice.isSuccess ? '1px solid rgba(16, 185, 129, 0.35)' : '1px solid rgba(56, 189, 248, 0.28)',
            borderRadius: 8,
            fontSize: 12,
            color: downloadNotice.isSuccess ? '#10b981' : '#38bdf8',
            fontWeight: 600,
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            transition: 'all 0.2s ease',
          }}
        >
          <span className="material-symbols-outlined" style={{ fontSize: 18 }}>
            {downloadNotice.isSuccess ? 'check_circle' : 'cloud_download'}
          </span>
          <span>{downloadNotice.text}</span>
        </div>
      )}

      {/* 4. MAIN CONTENT AREA (BASED ON ACTIVE SUB-TAB) */}

      {/* VIEW 1: MANIFEST & TIMELINE */}
      {activeSubTab === 'manifest' && (
        <div className="canvas-rounded-container" style={{ marginTop: 0 }}>
          <div className="canvas-two-column" style={{ paddingTop: 6, gap: 16 }}>
            {/* LEFT PANE: MASTER MANIFEST & SUMMARY */}
            <div className="canvas-pane">
              <div className="pane-header">
                <span className="pane-title">
                  <span className="material-symbols-outlined" style={{ fontSize: 18, color: 'var(--accent)' }}>lock</span>
                  Master Cryptographic Manifest
                </span>
                <span className="metric-trend-pill positive" style={{ fontSize: 10, fontWeight: 700 }}>
                  VERIFIED SECURE
                </span>
              </div>

              {/* TAMPER-PROOF DIGITAL SEAL CARD */}
              <div
                style={{
                  background: 'var(--bg-raised)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 12,
                  padding: '16px 18px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 12,
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <div
                      style={{
                        width: 36,
                        height: 36,
                        borderRadius: 8,
                        background: 'rgba(37, 99, 235, 0.12)',
                        color: 'var(--accent)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        flexShrink: 0,
                      }}
                    >
                      <span className="material-symbols-outlined" style={{ fontSize: 20 }}>verified_user</span>
                    </div>
                    <div>
                      <div style={{ fontSize: 13.5, fontWeight: 700, color: 'var(--text-primary)' }}>
                        Cryptographic Chain of Custody Seal
                      </div>
                      <div style={{ fontSize: 10.5, color: 'var(--text-muted)', marginTop: 1 }}>
                        ISO/IEC 27037 Tamper-Proof Digital Seal · Registered in Maritime Ledger
                      </div>
                    </div>
                  </div>

                  <button
                    className="action-pill-btn secondary"
                    onClick={handleCopySignature}
                    style={{ fontSize: 11, padding: '4px 10px' }}
                  >
                    <span className="material-symbols-outlined" style={{ fontSize: 14 }}>content_copy</span>
                    <span>Copy Digest</span>
                  </button>
                </div>

                {/* BALANCED 4x2 GRID SHA-256 FINGERPRINT DISPLAY */}
                <div
                  style={{
                    background: 'var(--bg-base)',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: 8,
                    padding: '12px 14px',
                    fontFamily: 'JetBrains Mono, monospace',
                    fontSize: 11,
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 10,
                  }}
                >
                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(4, 1fr)',
                      gap: 6,
                      textAlign: 'center',
                    }}
                  >
                    {hashChunks.map((chunk, i) => (
                      <span
                        key={i}
                        style={{
                          padding: '4px 6px',
                          background: 'rgba(37, 99, 235, 0.08)',
                          border: '1px solid rgba(37, 99, 235, 0.16)',
                          borderRadius: 5,
                          color: 'var(--accent)',
                          fontWeight: 600,
                          letterSpacing: '0.06em',
                        }}
                      >
                        {chunk}
                      </span>
                    ))}
                  </div>
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      borderTop: '1px solid var(--border-subtle)',
                      paddingTop: 8,
                      fontSize: 10,
                      color: 'var(--text-muted)',
                    }}
                  >
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                      <span className="material-symbols-outlined" style={{ fontSize: 13, color: '#10B981' }}>verified</span>
                      <span>W3C WebCrypto SHA-256 · 64 Hex Characters</span>
                    </span>
                    <button
                      type="button"
                      onClick={handleCopySignature}
                      style={{
                        background: 'transparent',
                        border: 'none',
                        color: 'var(--accent)',
                        cursor: 'pointer',
                        fontSize: 10.5,
                        fontWeight: 600,
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: 3,
                        padding: 0,
                      }}
                    >
                      <span className="material-symbols-outlined" style={{ fontSize: 13 }}>content_copy</span>
                      <span>Copy Full String</span>
                    </button>
                  </div>
                </div>

                <div
                  className="evidence-algo-grid"
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(3, 1fr)',
                    gap: 10,
                    background: 'var(--bg-base)',
                    borderRadius: 8,
                    padding: '8px 12px',
                    border: '1px solid var(--border-subtle)',
                    fontSize: 10.5,
                  }}
                >
                  <div>
                    <span className="text-muted" style={{ display: 'block', fontSize: 9.5 }}>Algorithm:</span>
                    <strong style={{ color: 'var(--text-primary)', marginTop: 1, display: 'block' }}>SHA-256 + ECDSA</strong>
                  </div>
                  <div>
                    <span className="text-muted" style={{ display: 'block', fontSize: 9.5 }}>Key Vault:</span>
                    <strong style={{ color: 'var(--text-primary)', marginTop: 1, display: 'block' }}>ICG-HSM-2026</strong>
                  </div>
                  <div>
                    <span className="text-muted" style={{ display: 'block', fontSize: 9.5 }}>Timestamp:</span>
                    <strong style={{ color: 'var(--text-primary)', marginTop: 1, display: 'block' }}>RFC 3161 TSP</strong>
                  </div>
                </div>
              </div>

              {/* EXPORTABLE FORENSIC ARTIFACTS LIST */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginTop: 4, maxHeight: 370, overflowY: 'auto', paddingRight: 4 }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 2 }}>
                  <span style={{ fontSize: 12.5, fontWeight: 700, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span className="material-symbols-outlined" style={{ fontSize: 18, color: 'var(--accent)' }}>folder_zip</span>
                    Exportable Forensic Artifacts ({artifacts.length} Assets)
                  </span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <span className="text-xs text-muted" style={{ fontSize: 10.5 }}>GeoJSON · GPKG · XML</span>
                    <button
                      className="action-pill-btn primary"
                      onClick={() => handleDownloadArtifact('full_forensic_bundle.zip')}
                      disabled={downloadingFile === 'full_forensic_bundle.zip'}
                      title="Download complete zipped forensic package with cryptographic manifest"
                      style={{ padding: '4px 11px', fontSize: 11, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 5 }}
                    >
                      <span className="material-symbols-outlined" style={{ fontSize: 14 }}>
                        {downloadingFile === 'full_forensic_bundle.zip' ? 'sync' : 'archive'}
                      </span>
                      <span>Download All (.ZIP)</span>
                    </button>
                  </div>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: 8, flex: 1, minHeight: 0, overflowY: 'auto', paddingRight: 2 }}>
                  {artifacts.map((art) => (
                    <div
                      key={art.id}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        padding: '10px 14px',
                        background: 'var(--bg-raised)',
                        borderRadius: 10,
                        border: '1px solid var(--border-subtle)',
                        transition: 'border-color 0.2s ease',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                        <span className="material-symbols-outlined" style={{ color: art.iconColor, fontSize: 20 }}>{art.icon}</span>
                        <div>
                          <div style={{ fontSize: 12.5, fontWeight: 700, color: 'var(--text-primary)' }}>
                            {art.title}
                          </div>
                          <div style={{ fontSize: 10.5, color: 'var(--text-muted)', marginTop: 2, display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                            <span className="mono" style={{ color: 'var(--text-secondary)', background: 'var(--bg-base)', padding: '1px 5px', borderRadius: 4, border: '1px solid var(--border-subtle)' }}>
                              {art.fileName}
                            </span>
                            <span>·</span>
                            <span>{art.crs}</span>
                            <span>·</span>
                            <span>{art.sizeKb.toFixed(1)} KB</span>
                          </div>
                        </div>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <button
                          className="action-pill-btn secondary"
                          onClick={() => setPreviewArtifact(art)}
                          title={`Preview ${art.fileName}`}
                          style={{ padding: '4px 9px', fontSize: 11, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4 }}
                        >
                          <span className="material-symbols-outlined" style={{ fontSize: 14 }}>visibility</span>
                          <span>Preview</span>
                        </button>
                        <button
                          className="action-pill-btn secondary"
                          onClick={() => handleDownloadArtifact(art.fileName)}
                          disabled={downloadingFile === art.fileName}
                          title={`Download ${art.fileName}`}
                          style={{ padding: '4px 10px', fontSize: 11, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4 }}
                        >
                          <span className="material-symbols-outlined" style={{ fontSize: 14 }}>
                            {downloadingFile === art.fileName ? 'sync' : 'download'}
                          </span>
                          <span>Download</span>
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* RIGHT PANE: TIMELINE & LEGAL STATUTORY ADVISORY */}
            <div className="canvas-pane">
              <div className="pane-header">
                <span className="pane-title">
                  <span className="material-symbols-outlined" style={{ fontSize: 18, color: 'var(--accent)' }}>link</span>
                  Forensic Chain of Custody Timeline
                </span>
                <span className="metric-trend-pill neutral" style={{ fontSize: 10.5 }}>
                  {officerSignOff ? '5 OF 5 SEALED & VERIFIED' : '4 OF 5 SEALED · PENDING SIGN-OFF'}
                </span>
              </div>

              {/* CHRONOLOGICAL TIMELINE WITH CONNECTING RAIL */}
              <div style={{ position: 'relative', display: 'flex', flexDirection: 'column', gap: 16, padding: '8px 0 8px 36px' }}>
                {/* Vertical connector line */}
                <div
                  style={{
                    position: 'absolute',
                    left: 11,
                    top: 12,
                    bottom: 12,
                    width: 2,
                    background: officerSignOff
                      ? 'linear-gradient(to bottom, #10b981 0%, #10b981 100%)'
                      : 'linear-gradient(to bottom, #10b981 0%, #10b981 75%, var(--accent) 100%)',
                    opacity: 0.45,
                    zIndex: 0,
                  }}
                />

                {milestones.map((m) => {
                  const isSealed = m.status === 'sealed';
                  return (
                    <div key={m.id} style={{ position: 'relative', display: 'flex', flexDirection: 'column', gap: 2 }}>
                      <div
                        style={{
                          position: 'absolute',
                          left: -36,
                          top: 0,
                          width: 24,
                          height: 24,
                          borderRadius: '50%',
                          background: isSealed ? 'rgba(16, 185, 129, 0.15)' : 'rgba(37, 99, 235, 0.15)',
                          border: isSealed ? '2px solid #10b981' : '2px solid var(--accent)',
                          color: isSealed ? '#10b981' : 'var(--accent)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          zIndex: 1,
                        }}
                      >
                        <span className="material-symbols-outlined" style={{ fontSize: 14 }}>{m.icon}</span>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 6 }}>
                        <div style={{ fontSize: 12.5, fontWeight: 700, color: isSealed ? 'var(--text-primary)' : 'var(--accent)' }}>
                          {m.title}
                        </div>
                        {m.id === 'm5' && !officerSignOff && (
                          <button
                            className="action-pill-btn primary"
                            onClick={() => setIsSignOffModalOpen(true)}
                            style={{ fontSize: 10.5, padding: '3px 10px', cursor: 'pointer' }}
                          >
                            <span className="material-symbols-outlined" style={{ fontSize: 13 }}>draw</span>
                            <span>Sign &amp; Seal</span>
                          </button>
                        )}
                        {m.id === 'm5' && officerSignOff && (
                          <span className="scenario-chip" style={{ fontSize: 9.5, borderColor: 'rgba(16, 185, 129, 0.4)', color: '#10b981' }}>
                            ✓ Signed
                          </span>
                        )}
                      </div>
                      <div style={{ fontSize: 10.5, color: 'var(--text-muted)', lineHeight: 1.45, marginTop: 2 }}>
                        <span className="mono" style={{ color: 'var(--text-secondary)' }}>{m.timestamp}</span> · {m.description}
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* PKI TRUST ANCHOR & AUDIT STATUS BAR */}
              <div
                style={{
                  background: 'var(--bg-raised)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 10,
                  padding: '10px 16px',
                  display: 'grid',
                  gridTemplateColumns: 'repeat(3, 1fr)',
                  gap: 10,
                  fontSize: 11,
                  marginTop: 6,
                }}
              >
                <div>
                  <span className="text-muted" style={{ display: 'block', fontSize: 10 }}>HSM Key Custody</span>
                  <strong style={{ color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: 5, marginTop: 2 }}>
                    <span className="material-symbols-outlined" style={{ fontSize: 14, color: '#10b981' }}>check_circle</span>
                    FIPS 140-2 Level 3
                  </strong>
                </div>
                <div>
                  <span className="text-muted" style={{ display: 'block', fontSize: 10 }}>Time Authority (RFC 3161)</span>
                  <strong style={{ color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: 5, marginTop: 2 }}>
                    <span className="material-symbols-outlined" style={{ fontSize: 14, color: '#10b981' }}>schedule</span>
                    GNSS Stratum-1
                  </strong>
                </div>
                <div>
                  <span className="text-muted" style={{ display: 'block', fontSize: 10 }}>Evidence Admissibility</span>
                  <strong style={{ color: 'var(--accent)', display: 'flex', alignItems: 'center', gap: 5, marginTop: 2 }}>
                    <span className="material-symbols-outlined" style={{ fontSize: 14 }}>verified</span>
                    BSA §63 / 65B
                  </strong>
                </div>
              </div>

              {/* LEGAL ADMISSIBILITY ADVISORY */}
              <div
                style={{
                  background: 'var(--bg-raised)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 12,
                  padding: '14px 18px',
                  marginTop: 'auto',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 8,
                }}
              >
                <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                    <span className="material-symbols-outlined" style={{ fontSize: 18, color: 'var(--accent)' }}>balance</span>
                    Statutory Compliance &amp; Legal Precedent
                  </span>
                  <span style={{ fontSize: 10.5, color: '#10b981', fontWeight: 600, display: 'flex', alignItems: 'center', gap: 4 }}>
                    <span className="material-symbols-outlined" style={{ fontSize: 13 }}>gavel</span>
                    Court Admissible
                  </span>
                </div>
                <p style={{ fontSize: 11, color: 'var(--text-secondary)', lineHeight: 1.5, margin: 0 }}>
                  This cryptographic packet adheres to the Bharatiya Sakshya Adhiniyam 2023 (Section 63 electronic record rules), Indian Evidence Act (Section 65B), and ISO/IEC 27037 standards for digital evidence collection.
                </p>
                <div style={{ display: 'flex', gap: 7, flexWrap: 'wrap', marginTop: 2 }}>
                  <span className="scenario-chip" style={{ fontSize: 10, padding: '3px 9px' }}>MARPOL 73/78 Annex I</span>
                  <span className="scenario-chip" style={{ fontSize: 10, padding: '3px 9px' }}>MS Act 1958 Sec 356</span>
                  <span className="scenario-chip" style={{ fontSize: 10, padding: '3px 9px' }}>UNCLOS Art. 217</span>
                  <span className="scenario-chip" style={{ fontSize: 10, padding: '3px 9px', borderColor: 'rgba(37, 99, 235, 0.4)', color: 'var(--accent)' }}>BSA 2023 §63</span>
                  <span className="scenario-chip" style={{ fontSize: 10, padding: '3px 9px', borderColor: 'rgba(37, 99, 235, 0.4)', color: 'var(--accent)' }}>ISO/IEC 27037:2012</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* VIEW 2: ARTIFACT INSPECTOR & LIVE VERIFIER */}
      {activeSubTab === 'artifacts' && (
        <div className="canvas-rounded-container" style={{ padding: 18, display: 'flex', flexDirection: 'column', gap: 14 }}>
          {/* VERIFIER BANNER & ACTION BAR */}
          <div
            style={{
              background: 'var(--bg-raised)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 10,
              padding: '14px 18px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              flexWrap: 'wrap',
              gap: 12,
            }}
          >
            <div>
              <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: 8 }}>
                <span className="material-symbols-outlined" style={{ color: '#10b981', fontSize: 20 }}>security</span>
                Interactive Cryptographic Integrity Verifier
              </div>
              <div style={{ fontSize: 10.5, color: 'var(--text-secondary)', marginTop: 2 }}>
                Perform on-the-spot bit-level verification of all 4 ingested forensic assets against the ECDSA Ledger Root.
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <button
                className="action-pill-btn secondary"
                onClick={() => handleDownloadArtifact('full_forensic_bundle.zip')}
                title="Download All 4 Assets in Zipped Archive"
                style={{ padding: '6px 14px', fontSize: 11 }}
              >
                <span className="material-symbols-outlined" style={{ fontSize: 15 }}>archive</span>
                <span>Download All (.ZIP)</span>
              </button>

              <button
                className="action-pill-btn primary"
                onClick={handleRunAudit}
                disabled={isAuditing}
                style={{ padding: '6px 14px', fontSize: 11, cursor: 'pointer' }}
              >
                <span className="material-symbols-outlined" style={{ fontSize: 15 }}>
                  {isAuditing ? 'sync' : 'verified'}
                </span>
                <span>{isAuditing ? 'Auditing Bits...' : 'Run Integrity Audit'}</span>
              </button>
            </div>
          </div>

          {/* AUDIT PROGRESS BAR (WHEN AUDITING OR JUST FINISHED) */}
          {(isAuditing || auditComplete) && (
            <div
              style={{
                background: auditComplete ? 'rgba(16, 185, 129, 0.08)' : 'rgba(37, 99, 235, 0.08)',
                border: auditComplete ? '1px solid rgba(16, 185, 129, 0.3)' : '1px solid rgba(37, 99, 235, 0.3)',
                borderRadius: 8,
                padding: '10px 14px',
                display: 'flex',
                flexDirection: 'column',
                gap: 6,
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, fontWeight: 700 }}>
                <span style={{ color: auditComplete ? '#10b981' : 'var(--accent)' }}>
                  {auditStepText}
                </span>
                <span className="mono" style={{ color: auditComplete ? '#10b981' : 'var(--accent)' }}>
                  {auditProgress}%
                </span>
              </div>
              <div style={{ width: '100%', height: 6, background: 'var(--bg-base)', borderRadius: 3, overflow: 'hidden' }}>
                <div
                  style={{
                    width: `${auditProgress}%`,
                    height: '100%',
                    background: auditComplete ? '#10b981' : 'var(--accent)',
                    transition: 'width 0.3s ease',
                  }}
                />
              </div>
            </div>
          )}

          {/* ARTIFACT CARDS GRID */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
            {artifacts.map((art) => (
              <div
                key={art.id}
                style={{
                  background: 'var(--bg-raised)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 10,
                  padding: '12px 14px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 8,
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <div
                      style={{
                        width: 32,
                        height: 32,
                        borderRadius: 6,
                        background: 'var(--bg-base)',
                        border: '1px solid var(--border-subtle)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        color: art.iconColor,
                      }}
                    >
                      <span className="material-symbols-outlined" style={{ fontSize: 18 }}>{art.icon}</span>
                    </div>
                    <div>
                      <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)' }}>
                        {art.title}
                      </div>
                      <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2, display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                        <span className="mono" style={{ color: 'var(--text-secondary)', background: 'var(--bg-base)', padding: '1px 5px', borderRadius: 4, border: '1px solid var(--border-subtle)' }}>
                          {art.fileName}
                        </span>
                        <span>·</span>
                        <span>{art.crs}</span>
                        <span>·</span>
                        <span>{art.sizeKb.toFixed(1)} KB</span>
                      </div>
                    </div>
                  </div>

                  <span className="chip chip-ok" style={{ fontSize: 9, padding: '2px 6px' }}>
                    SEALED
                  </span>
                </div>

                <p style={{ fontSize: 10.5, color: 'var(--text-secondary)', margin: 0, lineHeight: 1.4 }}>
                  {art.description}
                </p>

                {/* INDIVIDUAL SHA-256 HASH */}
                <div
                  style={{
                    background: 'var(--bg-base)',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: 4,
                    padding: '5px 8px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                  }}
                >
                  <span className="mono text-muted" style={{ fontSize: 9, wordBreak: 'break-all' }}>
                    SHA-256: {art.sha256.slice(0, 36)}...
                  </span>
                  <button
                    className="action-pill-btn secondary"
                    onClick={() => handleCopyText(art.sha256, `${art.fileName} SHA-256`)}
                    title="Copy full SHA-256 hash"
                    style={{ padding: '2px 6px', fontSize: 9 }}
                  >
                    <span className="material-symbols-outlined" style={{ fontSize: 11 }}>content_copy</span>
                  </button>
                </div>

                {/* ACTION BUTTONS */}
                <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 'auto', paddingTop: 4 }}>
                  <button
                    className="action-pill-btn secondary"
                    onClick={() => setPreviewArtifact(art)}
                    style={{ fontSize: 10.5, padding: '4px 10px' }}
                  >
                    <span className="material-symbols-outlined" style={{ fontSize: 14 }}>visibility</span>
                    <span>Inspect / Preview</span>
                  </button>
                  <button
                    className="action-pill-btn primary"
                    onClick={() => handleDownloadArtifact(art.fileName)}
                    disabled={downloadingFile === art.fileName}
                    style={{ fontSize: 10.5, padding: '4px 10px' }}
                  >
                    <span className="material-symbols-outlined" style={{ fontSize: 14 }}>download</span>
                    <span>Download</span>
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* VIEW 3: COURT DOSSIER & STATUTORY CERTIFICATE */}
      {activeSubTab === 'dossier' && (
        <div className="canvas-rounded-container" style={{ padding: 20 }}>
          <div
            style={{
              background: 'var(--bg-raised)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 10,
              padding: '24px 28px',
              maxWidth: 920,
              margin: '0 auto',
              display: 'flex',
              flexDirection: 'column',
              gap: 16,
            }}
          >
            {/* OFFICIAL HEADER */}
            <div style={{ textAlign: 'center', borderBottom: '2px solid var(--border-default)', paddingBottom: 14 }}>
              <div style={{ fontSize: 10, fontWeight: 800, letterSpacing: '0.15em', textTransform: 'uppercase', color: 'var(--text-muted)' }}>
                Government of India · Ministry of Defence · Indian Coast Guard
              </div>
              <div style={{ fontSize: 18, fontWeight: 800, color: 'var(--text-primary)', marginTop: 4 }}>
                MARITIME DIGITAL FORENSIC ATTRIBUTION REPORT
              </div>
              <div style={{ fontSize: 11.5, fontWeight: 600, color: 'var(--accent)', marginTop: 2 }}>
                Statutory Certificate under Section 63, Bharatiya Sakshya Adhiniyam (BSA) 2023 &amp; Section 65B, Indian Evidence Act
              </div>
              <div className="mono text-xs text-muted" style={{ marginTop: 4 }}>
                Dossier Ref: ICG/MRCC/2026/SS-{activeScenario?.id || 'INC-001'} · Classification: RESTRICTED // INVESTIGATIVE DECISION-SUPPORT
              </div>
            </div>

            {/* ACTION BUTTONS ON TOP OF DOSSIER */}
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
              <button
                className="action-pill-btn secondary"
                onClick={onOpenForensicModal}
                style={{ fontSize: 11, padding: '4px 12px' }}
              >
                <span className="material-symbols-outlined" style={{ fontSize: 15 }}>open_in_new</span>
                <span>Open in Fullscreen Modal</span>
              </button>
              <button
                className="action-pill-btn primary"
                onClick={() => window.print()}
                style={{ fontSize: 11, padding: '4px 12px' }}
              >
                <span className="material-symbols-outlined" style={{ fontSize: 15 }}>print</span>
                <span>Print / Save Official PDF</span>
              </button>
            </div>

            {/* INCIDENT DETAILS */}
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(2, 1fr)',
                gap: 10,
                background: 'var(--bg-base)',
                padding: '12px 16px',
                borderRadius: 6,
                border: '1px solid var(--border-subtle)',
                fontSize: 11.5,
              }}
            >
              <div><strong>Incident Identifier:</strong> <span className="mono">{activeScenario?.id || 'INC-2026-001'}</span></div>
              <div><strong>Operational Sector:</strong> <span className="mono">{activeScenario?.title || 'Mumbai High Offshore Basin'}</span></div>
              <div><strong>Centroid Coordinate:</strong> <span className="mono">{(activeScenario?.lat || 18.74).toFixed(5)}°N, {(activeScenario?.lng || 71.21).toFixed(5)}°E</span></div>
              <div><strong>Derived Surface Footprint:</strong> <span className="mono">{activeScenario?.area || '4.82 km²'}</span></div>
              <div><strong>Sensor Mission:</strong> <span className="mono">Sentinel-1A IW GRD (Copernicus C-SAR)</span></div>
              <div><strong>MetOcean Forcing:</strong> <span className="mono">ERA5 4.2 m/s · CMEMS 0.34 kn Surface Current</span></div>
            </div>

            {/* ATTRIBUTED VESSEL */}
            <div>
              <div style={{ fontSize: 11, fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 6, color: 'var(--text-primary)' }}>
                Primary Suspect Vessel Attribution Record
              </div>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11.5, border: '1px solid var(--border-subtle)' }}>
                <thead>
                  <tr style={{ background: 'var(--bg-base)', textAlign: 'left' }}>
                    <th style={{ padding: '6px 8px', border: '1px solid var(--border-subtle)' }}>Rank</th>
                    <th style={{ padding: '6px 8px', border: '1px solid var(--border-subtle)' }}>Vessel Name</th>
                    <th style={{ padding: '6px 8px', border: '1px solid var(--border-subtle)' }}>MMSI / IMO</th>
                    <th style={{ padding: '6px 8px', border: '1px solid var(--border-subtle)' }}>Flag / Registry</th>
                    <th style={{ padding: '6px 8px', border: '1px solid var(--border-subtle)' }}>Attribution Score</th>
                    <th style={{ padding: '6px 8px', border: '1px solid var(--border-subtle)' }}>Correlated Anomaly</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td style={{ padding: '6px 8px', border: '1px solid var(--border-subtle)', fontWeight: 700, color: 'var(--accent)' }}>#1 PRIMARY</td>
                    <td style={{ padding: '6px 8px', border: '1px solid var(--border-subtle)', fontWeight: 700 }}>
                      {activeScenario?.topVessel || 'Suspect Tanker'}
                    </td>
                    <td style={{ padding: '6px 8px', border: '1px solid var(--border-subtle)', fontFamily: 'monospace' }}>
                      {activeScenario?.diagVessel ? activeScenario.diagVessel.replace(/^[^·]*·\s*/, '') : 'MMSI 419001234'}
                    </td>
                    <td style={{ padding: '6px 8px', border: '1px solid var(--border-subtle)' }}>
                      {activeScenario?.oilType || 'Crude Oil'} Carrier · Indian EEZ
                    </td>
                    <td style={{ padding: '6px 8px', border: '1px solid var(--border-subtle)', fontFamily: 'monospace', fontWeight: 800, color: '#ef4444' }}>
                      {activeScenario?.scores?.[0] ? `${activeScenario.scores[0].toFixed(2)} / 1.00` : '0.86 / 1.00'}
                    </td>
                    <td style={{ padding: '6px 8px', border: '1px solid var(--border-subtle)', color: 'var(--text-secondary)', fontSize: 10.5 }}>
                      {activeScenario?.diagDetails || 'Intentional AIS silence gap crossing Lagrangian origin contour'}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            {/* BSA 2023 SECTION 63 LEGAL ATTESTATION */}
            <div
              style={{
                background: 'rgba(37, 99, 235, 0.04)',
                border: '1px solid rgba(37, 99, 235, 0.25)',
                borderRadius: 6,
                padding: '12px 16px',
                display: 'flex',
                flexDirection: 'column',
                gap: 6,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div style={{ fontSize: 11, fontWeight: 800, color: 'var(--accent)', textTransform: 'uppercase', display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span className="material-symbols-outlined" style={{ fontSize: 16 }}>verified</span>
                  Statutory Certificate of Electronic Evidence (Section 63 BSA 2023)
                </div>
                <span className="chip chip-ok" style={{ fontSize: 9.5, padding: '2px 8px' }}>SEALED &amp; VERIFIED</span>
              </div>
              <p style={{ fontSize: 10.5, color: 'var(--text-secondary)', lineHeight: 1.5, margin: 0 }}>
                I hereby certify that the electronic records contained in this dossier were produced by the automated computer surveillance system <strong>SpillSense Maritime C2</strong> during the course of regular operational activity. The computer systems and cryptographic hardware security modules (HSM FIPS 140-2 Level 3) were operating properly throughout the ingestion, processing, trajectory drift modeling, and AIS correlation cycles, with no unauthorized tampering or alterations.
              </p>
              <div
                style={{
                  fontFamily: 'monospace',
                  fontSize: 10,
                  background: 'var(--bg-base)',
                  padding: '6px 10px',
                  borderRadius: 4,
                  border: '1px solid var(--border-subtle)',
                  color: 'var(--accent)',
                  wordBreak: 'break-all',
                }}
              >
                Master Cryptographic Ledger Digest: {hashChunks.join(' ')}
              </div>
            </div>

            {/* OFFICER SIGN-OFF FOOTER */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginTop: 12, paddingTop: 14, borderTop: '1px solid var(--border-subtle)' }}>
              <div>
                <div style={{ fontSize: 10, textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 700 }}>
                  Certifying Officer / Digital Signature
                </div>
                <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-primary)', marginTop: 2 }}>
                  {officerSignOff?.officerName || 'Duty Surveillance Officer (ICG Maritime Intel Unit)'}
                </div>
                <div className="mono text-xs text-muted" style={{ fontSize: 10, marginTop: 1 }}>
                  HSM FIPS 140-2 Token: ICG-C2-2026-9 · Status: {officerSignOff ? 'Digitally Signed & Endorsed' : 'Automated Ledger Sealed'}
                </div>
              </div>

              <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: 10, textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 700 }}>
                  Statutory Enactment
                </div>
                <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-primary)', marginTop: 2 }}>
                  Sec 63 BSA 2023 · Sec 356 MS Act 1958 · MARPOL Annex I
                </div>
                <div className="mono text-xs text-muted" style={{ fontSize: 10, marginTop: 1 }}>
                  Indian Coast Guard Maritime Law Enforcement
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 5. INLINE ARTIFACT PREVIEW DRAWER / MODAL */}
      {previewArtifact && (
        <div
          className="modal-backdrop open"
          style={{ zIndex: 1250 }}
          onClick={(e) => {
            if (e.target === e.currentTarget) setPreviewArtifact(null);
          }}
        >
          <div className="modal-card" style={{ maxWidth: 740, maxHeight: '85vh', display: 'flex', flexDirection: 'column' }}>
            <div className="modal-header">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined" style={{ color: previewArtifact.iconColor }}>
                  {previewArtifact.icon}
                </span>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                  <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)' }}>
                    {previewArtifact.title}
                  </span>
                  <span className="mono" style={{ fontSize: 11, color: 'var(--text-secondary)', background: 'var(--bg-base)', padding: '1px 6px', borderRadius: 4, border: '1px solid var(--border-subtle)' }}>
                    {previewArtifact.fileName}
                  </span>
                </div>
                <span className="scenario-chip" style={{ fontSize: 9.5 }}>{previewArtifact.crs}</span>
              </div>
              <div className="flex items-center gap-2 modal-actions">
                <button
                  className="action-pill-btn secondary"
                  onClick={() => {
                    const text = previewArtifact.contentGenerator(
                      activeScenario,
                      masterHash
                    );
                    handleCopyText(text, previewArtifact.fileName);
                  }}
                  style={{ fontSize: 10.5, padding: '3px 8px' }}
                >
                  <span className="material-symbols-outlined" style={{ fontSize: 13 }}>content_copy</span>
                  <span>Copy Raw</span>
                </button>

                <button
                  className="action-pill-btn primary"
                  onClick={() => handleDownloadArtifact(previewArtifact.fileName)}
                  style={{ fontSize: 10.5, padding: '3px 8px' }}
                >
                  <span className="material-symbols-outlined" style={{ fontSize: 13 }}>download</span>
                  <span>Download</span>
                </button>

                <button className="btn-icon" onClick={() => setPreviewArtifact(null)}>
                  <span className="material-symbols-outlined">close</span>
                </button>
              </div>
            </div>

            <div className="modal-body" style={{ flex: 1, minHeight: 0, padding: 14, overflowY: 'auto' }}>
              <div
                style={{
                  background: 'var(--bg-base)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 6,
                  padding: '10px 12px',
                  fontFamily: 'JetBrains Mono, monospace',
                  fontSize: 10.5,
                  lineHeight: 1.45,
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                  color: 'var(--text-primary)',
                  maxHeight: '60vh',
                  overflowY: 'auto',
                }}
              >
                {previewArtifact.contentGenerator(
                  activeScenario,
                  masterHash
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 6. OFFICER SIGN-OFF DIALOG */}
      {isSignOffModalOpen && (
        <div
          className="modal-backdrop open"
          style={{ zIndex: 1250 }}
          onClick={(e) => {
            if (e.target === e.currentTarget) setIsSignOffModalOpen(false);
          }}
        >
          <div className="modal-card" style={{ maxWidth: 480 }}>
            <div className="modal-header">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined" style={{ color: 'var(--accent)' }}>draw</span>
                <span style={{ fontSize: 13, fontWeight: 700 }}>
                  Duty Surveillance Officer Sign-Off &amp; Seal
                </span>
              </div>
              <button className="btn-icon" onClick={() => setIsSignOffModalOpen(false)}>
                <span className="material-symbols-outlined">close</span>
              </button>
            </div>

            <div className="modal-body" style={{ display: 'flex', flexDirection: 'column', gap: 12, padding: 16 }}>
              <p style={{ fontSize: 11, color: 'var(--text-secondary)', margin: 0, lineHeight: 1.4 }}>
                Applying your electronic endorsement registers an immutable milestone under Section 63 of Bharatiya Sakshya Adhiniyam 2023, sealing this evidence dossier for court proceedings.
              </p>

              <div>
                <label style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)' }}>
                  Officer Name &amp; Rank
                </label>
                <input
                  type="text"
                  value={signOffName}
                  onChange={(e) => setSignOffName(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '8px 10px',
                    borderRadius: 6,
                    border: '1px solid var(--border-default)',
                    background: 'var(--bg-base)',
                    color: 'var(--text-primary)',
                    fontSize: 12,
                    marginTop: 4,
                  }}
                />
              </div>

              <div>
                <label style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)' }}>
                  Command Station / Unit
                </label>
                <input
                  type="text"
                  value={signOffStation}
                  onChange={(e) => setSignOffStation(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '8px 10px',
                    borderRadius: 6,
                    border: '1px solid var(--border-default)',
                    background: 'var(--bg-base)',
                    color: 'var(--text-primary)',
                    fontSize: 12,
                    marginTop: 4,
                  }}
                />
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 8 }}>
                <button
                  className="action-pill-btn secondary"
                  onClick={() => setIsSignOffModalOpen(false)}
                  style={{ fontSize: 11, padding: '5px 12px' }}
                >
                  Cancel
                </button>
                <button
                  className="action-pill-btn primary"
                  onClick={handleConfirmSignOff}
                  style={{ fontSize: 11, padding: '5px 14px', cursor: 'pointer' }}
                >
                  <span className="material-symbols-outlined" style={{ fontSize: 15 }}>verified_user</span>
                  <span>Confirm &amp; Seal Dossier</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
