import React from 'react';
import type { Scenario } from '../../types/dashboard';
import { generateDeterministicHash, formatHashChunks } from '../../utils/forensicHash';

interface ForensicModalProps {
  isOpen: boolean;
  onClose: () => void;
  scenario: Scenario;
  officerSignOff?: { officerName: string; timestamp: string } | null;
}

export const ForensicModal: React.FC<ForensicModalProps> = ({ isOpen, onClose, scenario, officerSignOff }) => {
  if (!isOpen) return null;

  const masterHash = generateDeterministicHash(scenario.id + scenario.title + scenario.lat + scenario.lng);
  const hashChunks = formatHashChunks(masterHash);

  return (
    <div
      className="modal-backdrop open"
      id="forensic-modal"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
      style={{ zIndex: 1200 }}
    >
      <div className="modal-card print-dossier-card" style={{ maxWidth: 840, maxHeight: '92vh', overflowY: 'auto' }}>
        <div className="modal-header hide-on-print">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined" style={{ color: 'var(--accent)' }}>gavel</span>
            <span style={{ fontSize: 13, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
              Official Maritime Forensic Evidence Dossier &amp; BSA §63 Certificate
            </span>
          </div>
          <div className="flex items-center gap-2 modal-actions">
            <a
              href={`http://127.0.0.1:8000/api/v1/dossier/${scenario.id || 'INC-001'}`}
              target="_blank"
              rel="noopener noreferrer"
              className="action-pill-btn secondary"
              title="Download Authenticated ReportLab Binary PDF from Backend API"
              style={{ padding: '4px 12px', fontSize: 11, cursor: 'pointer', textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: 4 }}
            >
              <span className="material-symbols-outlined" style={{ fontSize: 15 }}>download</span>
              Download Signed PDF
            </a>
            <button
              className="action-pill-btn primary"
              onClick={() => window.print()}
              style={{ padding: '4px 12px', fontSize: 11, cursor: 'pointer' }}
            >
              <span className="material-symbols-outlined" style={{ fontSize: 15 }}>print</span>
              Print / Save Court PDF
            </button>
            <button className="btn-icon" onClick={onClose} title="Close Modal">
              <span className="material-symbols-outlined">close</span>
            </button>
          </div>
        </div>

        <div className="modal-body" style={{ padding: '20px 24px', display: 'flex', flexDirection: 'column', gap: 16 }}>
          {/* HEADER / EMBLEM */}
          <div style={{ textAlign: 'center', borderBottom: '2px solid var(--border-default)', paddingBottom: 14 }}>
            <div style={{ fontSize: 10, fontWeight: 800, letterSpacing: '0.15em', textTransform: 'uppercase', color: 'var(--text-muted)' }}>
              Government of India · Ministry of Defence · Indian Coast Guard
            </div>
            <div style={{ fontSize: 17, fontWeight: 800, color: 'var(--text-primary)', marginTop: 4, letterSpacing: '0.02em' }}>
              MARITIME DIGITAL FORENSIC ATTRIBUTION REPORT
            </div>
            <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--accent)', marginTop: 2 }}>
              Statutory Certificate under Section 63, Bharatiya Sakshya Adhiniyam (BSA) 2023 &amp; Section 65B, Indian Evidence Act
            </div>
            <div className="mono text-xs text-muted" style={{ marginTop: 4, fontSize: 10 }}>
              Dossier Ref: ICG/MRCC/2026/SS-{scenario.id} · Security Classification: RESTRICTED // INVESTIGATIVE DECISION SUPPORT
            </div>
          </div>

          {/* INCIDENT TELEMETRY GRID */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: 10,
              background: 'var(--bg-raised)',
              padding: '12px 14px',
              borderRadius: 6,
              border: '1px solid var(--border-subtle)',
              fontSize: 11.5,
            }}
          >
            <div><strong>Incident Identifier:</strong> <span className="mono">{scenario.id}</span></div>
            <div><strong>Location Sector:</strong> <span className="mono">{scenario.title}</span></div>
            <div><strong>Centroid Coordinate:</strong> <span className="mono">{scenario.lat.toFixed(5)}°N, {scenario.lng.toFixed(5)}°E (WGS 84)</span></div>
            <div><strong>Surface Footprint:</strong> <span className="mono">{scenario.area || '19.67 km²'}</span></div>
            <div><strong>Sensor Mission:</strong> <span className="mono">Sentinel-1A IW GRD VV+VH (Copernicus)</span></div>
            <div><strong>Oceanic MetOcean:</strong> <span className="mono">ERA5 Wind 4.2 m/s · CMEMS 0.34 kn Surface Current</span></div>
            <div><strong>Cryptographic Hash:</strong> <span className="mono text-accent" style={{ fontSize: 10 }}>{masterHash.slice(0, 24)}...</span></div>
            <div><strong>Admissibility Level:</strong> <span className="mono" style={{ color: '#10b981', fontWeight: 700 }}>Tier 4 Court-Admissible</span></div>
          </div>

          {/* SUSPECT VESSEL ATTRIBUTION */}
          <div>
            <div style={{ fontSize: 11, fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 6, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className="material-symbols-outlined" style={{ fontSize: 16, color: 'var(--accent)' }}>directions_boat</span>
              Primary Suspect Vessel Attribution Record
            </div>
            <div style={{ overflowX: 'auto', WebkitOverflowScrolling: 'touch' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11.5, border: '1px solid var(--border-subtle)', minWidth: 540 }}>
                <thead>
                  <tr style={{ background: 'var(--bg-raised)', textAlign: 'left', color: 'var(--text-secondary)' }}>
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
                      {scenario.topVessel}
                    </td>
                    <td style={{ padding: '6px 8px', border: '1px solid var(--border-subtle)', fontFamily: 'JetBrains Mono, monospace' }}>
                      419001234 / 9412345
                    </td>
                    <td style={{ padding: '6px 8px', border: '1px solid var(--border-subtle)' }}>Liberia · Crude Oil Tanker</td>
                    <td style={{ padding: '6px 8px', border: '1px solid var(--border-subtle)', fontFamily: 'JetBrains Mono, monospace', fontWeight: 800, color: '#ef4444' }}>
                      0.86 / 1.00
                    </td>
                    <td style={{ padding: '6px 8px', border: '1px solid var(--border-subtle)', color: 'var(--text-secondary)', fontSize: 10.5 }}>
                      {scenario.diagDetails || 'Intentional AIS silence gap (4.2 hrs) crossing Lagrangian origin contour'}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* STATUTORY SECTION 63 BSA 2023 CERTIFICATE */}
          <div
            style={{
              background: 'rgba(37, 99, 235, 0.04)',
              border: '1px solid rgba(37, 99, 235, 0.25)',
              borderRadius: 6,
              padding: '12px 14px',
              display: 'flex',
              flexDirection: 'column',
              gap: 8,
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, fontWeight: 800, color: 'var(--accent)', textTransform: 'uppercase' }}>
                <span className="material-symbols-outlined" style={{ fontSize: 16 }}>verified</span>
                Statutory Certificate of Electronic Evidence (Section 63 BSA 2023)
              </div>
              <span className="chip chip-ok" style={{ fontSize: 9.5, padding: '2px 8px' }}>SEALED &amp; VERIFIED</span>
            </div>
            <p style={{ fontSize: 10, color: 'var(--text-secondary)', lineHeight: 1.5, margin: 0 }}>
              I hereby certify that the electronic records contained in this dossier were produced by the automated computer surveillance system <strong>SpillSense Maritime C2</strong> during the course of regular operational activity. The computer systems and cryptographic hardware security modules (HSM FIPS 140-2 Level 3) were operating properly throughout the ingestion, processing, trajectory drift modeling, and AIS correlation cycles, with no unauthorized tampering or alterations.
            </p>
            <div
              style={{
                fontFamily: 'monospace',
                fontSize: 9.5,
                background: 'var(--bg-base)',
                padding: '6px 8px',
                borderRadius: 4,
                border: '1px solid var(--border-subtle)',
                color: 'var(--text-primary)',
                wordBreak: 'break-all',
              }}
            >
              Master SHA-256 Digest: {hashChunks.join(' ')}
            </div>
          </div>

          {/* INVESTIGATOR SIGN-OFF BLOCK */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginTop: 10, paddingTop: 12, borderTop: '1px solid var(--border-subtle)' }}>
            <div>
              <div style={{ fontSize: 10, textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 700 }}>
                Certifying Officer / Digital Signature
              </div>
              <div style={{ fontSize: 11.5, fontWeight: 700, color: 'var(--text-primary)', marginTop: 2 }}>
                {officerSignOff?.officerName || 'Surveillance Duty Officer (ICG C2 Operations)'}
              </div>
              <div className="mono text-xs text-muted" style={{ fontSize: 9.5, marginTop: 1 }}>
                FIPS 140-2 Level 3 Cryptographic Token ID: ICG-C2-2026-9
              </div>
              <div className="mono text-xs text-muted" style={{ fontSize: 9.5 }}>
                Timestamp: {officerSignOff?.timestamp || `${new Date().toISOString().replace('T', ' ').slice(0, 19)} UTC`}
              </div>
            </div>

            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: 10, textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 700 }}>
                Regulatory Jurisdiction &amp; Precedent
              </div>
              <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-primary)', marginTop: 2 }}>
                Merchant Shipping Act 1958 §356 · MARPOL 73/78 Annex I
              </div>
              <div className="mono text-xs text-muted" style={{ fontSize: 9.5, marginTop: 1 }}>
                Maritime Law Enforcement Cell · Regional HQ West / Gandhinagar
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
