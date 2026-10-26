import React, { useState, useEffect } from 'react';

interface KeyMeta {
  key: string;
  title: string;
  service: string;
  description: string;
  placeholder: string;
  docsUrl: string;
  required: boolean;
  configured: boolean;
  maskedValue: string | null;
  length: number;
}

interface ApiKeySetupModalProps {
  isOpen: boolean;
  onClose: () => void;
  onKeysUpdated?: () => void;
}

const DEFAULT_KEYS: Record<string, KeyMeta> = {
  CESIUM_ION_TOKEN: {
    key: 'CESIUM_ION_TOKEN',
    title: 'Cesium Ion Access Token',
    service: 'Cesium Ion 3D Cloud',
    description: 'Enables high-resolution Cesium World Terrain, bathymetry, photorealistic 3D tiles, and satellite basemaps.',
    placeholder: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...',
    docsUrl: 'https://ion.cesium.com/tokens',
    required: false,
    configured: true,
    maskedValue: 'eyJhbG••••••••R5uY',
    length: 300,
  },
  AISSTREAM_API_KEY: {
    key: 'AISSTREAM_API_KEY',
    title: 'AISStream.io Live WebSocket Key',
    service: 'AISStream.io Global Feed',
    description: 'Powers real-time transponder telemetry for commercial tankers and cargo vessels across the Indian EEZ.',
    placeholder: 'a5905a6907b539642a35fb5f80dc5b19e423f4a4',
    docsUrl: 'https://aisstream.io',
    required: true,
    configured: true,
    maskedValue: 'a5905a••••••••f4a4',
    length: 40,
  },
  CDSE_CLIENT_ID: {
    key: 'CDSE_CLIENT_ID',
    title: 'Copernicus CDSE Client ID',
    service: 'Copernicus Data Space',
    description: 'Accesses Sentinel-1 C-SAR IW GRD satellite passes and raw radar imagery for oil slick delineation.',
    placeholder: 'cdse-public',
    docsUrl: 'https://dataspace.copernicus.eu',
    required: false,
    configured: true,
    maskedValue: 'cdse-public',
    length: 11,
  },
  CDSE_CLIENT_SECRET: {
    key: 'CDSE_CLIENT_SECRET',
    title: 'Copernicus CDSE Client Secret',
    service: 'Copernicus Data Space',
    description: 'OAuth2 client secret for automated Sentinel-1 product catalog queries and download pipelines.',
    placeholder: '••••••••••••••••',
    docsUrl: 'https://dataspace.copernicus.eu',
    required: false,
    configured: true,
    maskedValue: 'cdse-k••••••••hhh6',
    length: 27,
  },
  AISHUB_USERNAME: {
    key: 'AISHUB_USERNAME',
    title: 'AISHub Service Username',
    service: 'AISHub Maritime Network',
    description: 'Secondary terrestrial AIS stream provider for vessel identity reconciliation.',
    placeholder: 'krshhh6',
    docsUrl: 'https://www.aishub.net',
    required: false,
    configured: true,
    maskedValue: 'krshhh6',
    length: 7,
  },
};

export const ApiKeySetupModal: React.FC<ApiKeySetupModalProps> = ({
  isOpen,
  onClose,
  onKeysUpdated,
}) => {
  const [keyMeta, setKeyMeta] = useState<Record<string, KeyMeta>>(DEFAULT_KEYS);
  const [editValues, setEditValues] = useState<Record<string, string>>({});

  const [isSaving, setIsSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState<{ success: boolean; msg: string } | null>(null);

  // Diagnostics test states
  const [aisTestStatus, setAisTestStatus] = useState<{ testing: boolean; result?: string; success?: boolean }>({
    testing: false,
  });
  const [cesiumTestStatus, setCesiumTestStatus] = useState<{ testing: boolean; result?: string; success?: boolean }>({
    testing: false,
  });

  // Fetch status on open
  useEffect(() => {
    if (!isOpen) return;

    let mounted = true;
    

    fetch('/api/v1/setup/status')
      .then((r) => r.json())
      .then((data) => {
        if (!mounted) return;
        if (data && data.keys) {
          setKeyMeta(data.keys);
        }
      })
      .catch((e) => {
        console.warn('Could not fetch /api/v1/setup/status, using default metadata:', e);
      })
      .finally(() => {
        
      });

    return () => {
      mounted = false;
    };
  }, [isOpen]);

  if (!isOpen) return null;

  const handleInputChange = (k: string, val: string) => {
    setEditValues((prev) => ({ ...prev, [k]: val }));
  };

  const handleSaveKeys = async () => {
    setIsSaving(true);
    setSaveStatus(null);

    // Only include edited values with content or explicit clears
    const payloadKeys: Record<string, string> = {};
    for (const [k, v] of Object.entries(editValues)) {
      if (v !== undefined) {
        payloadKeys[k] = v;
      }
    }

    try {
      const res = await fetch('/api/v1/setup/keys', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ keys: payloadKeys }),
      });

      if (!res.ok) {
        throw new Error(`Server returned HTTP ${res.status}`);
      }

      const result = await res.json();
      setSaveStatus({
        success: true,
        msg: result.message || 'API Keys saved successfully to environment.',
      });

      // Clear input buffers and reload status
      setEditValues({});
      const statusRes = await fetch('/api/v1/setup/status');
      if (statusRes.ok) {
        const data = await statusRes.json();
        if (data && data.keys) setKeyMeta(data.keys);
      }

      if (onKeysUpdated) onKeysUpdated();
    } catch (err: any) {
      setSaveStatus({
        success: false,
        msg: err.message || 'Failed to save keys. Please verify backend is running.',
      });
    } finally {
      setIsSaving(false);
    }
  };

  const handleTestAisStream = async () => {
    setAisTestStatus({ testing: true });
    try {
      const keyOverride = editValues['AISSTREAM_API_KEY'];
      const url = keyOverride
        ? `/api/v1/setup/test-ais?key_override=${encodeURIComponent(keyOverride)}`
        : '/api/v1/setup/test-ais';

      const res = await fetch(url, { method: 'POST' });
      const data = await res.json();

      if (data.success) {
        setAisTestStatus({
          testing: false,
          success: true,
          result: `SUCCESS (${data.latencyMs}ms): ${data.message}`,
        });
      } else {
        setAisTestStatus({
          testing: false,
          success: false,
          result: `FAILED: ${data.error || 'Connection rejected'}`,
        });
      }
    } catch (e: any) {
      setAisTestStatus({
        testing: false,
        success: false,
        result: `ERROR: ${e.message}`,
      });
    }
  };

  const handleTestCesiumIon = async () => {
    setCesiumTestStatus({ testing: true });
    try {
      const tokenOverride = editValues['CESIUM_ION_TOKEN'];
      const url = tokenOverride
        ? `/api/v1/setup/test-cesium?token_override=${encodeURIComponent(tokenOverride)}`
        : '/api/v1/setup/test-cesium';

      const res = await fetch(url, { method: 'POST' });
      const data = await res.json();

      if (data.success) {
        setCesiumTestStatus({
          testing: false,
          success: true,
          result: `VERIFIED (${data.latencyMs}ms): ${data.message}`,
        });
      } else {
        setCesiumTestStatus({
          testing: false,
          success: false,
          result: `REJECTED: ${data.error || 'Token unauthorized'}`,
        });
      }
    } catch (e: any) {
      setCesiumTestStatus({
        testing: false,
        success: false,
        result: `ERROR: ${e.message}`,
      });
    }
  };

  return (
    <div
      className="modal-backdrop open"
      id="api-key-setup-modal"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
      style={{
        zIndex: 1300,
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(2, 6, 23, 0.85)',
        backdropFilter: 'blur(8px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '16px',
      }}
    >
      <div
        className="modal-card"
        style={{
          width: '100%',
          maxWidth: 780,
          maxHeight: '90vh',
          backgroundColor: '#0a0f1d',
          border: '1px solid rgba(139, 92, 246, 0.35)',
          borderRadius: 12,
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.8)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          color: '#e2e8f0',
        }}
      >
        {/* Modal Header */}
        <div
          style={{
            padding: '14px 20px',
            borderBottom: '1px solid rgba(139, 92, 246, 0.25)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            background: 'rgba(15, 23, 42, 0.95)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ fontSize: 20 }}>⚡</span>
            <div>
              <div style={{ fontFamily: 'monospace', fontWeight: 800, fontSize: 14, letterSpacing: '0.05em' }}>
                C2 API KEY SETUP &amp; LIVE FEEDS CONFIGURATION
              </div>
              <div style={{ fontSize: 11, color: '#94a3b8' }}>
                Manage live AIS vessel streams, Cesium Ion 3D tokens, and spaceborne SAR credentials
              </div>
            </div>
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'transparent',
              border: 'none',
              color: '#94a3b8',
              cursor: 'pointer',
              fontSize: 18,
              padding: '4px 8px',
            }}
          >
            ✕
          </button>
        </div>

        {/* Modal Body */}
        <div style={{ padding: '20px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 16 }}>
          {saveStatus && (
            <div
              style={{
                padding: '10px 14px',
                borderRadius: 6,
                fontSize: 12,
                fontFamily: 'monospace',
                background: saveStatus.success ? 'rgba(34, 197, 94, 0.15)' : 'rgba(239, 68, 68, 0.15)',
                border: `1px solid ${saveStatus.success ? '#22c55e' : '#ef4444'}`,
                color: saveStatus.success ? '#4ade80' : '#f87171',
              }}
            >
              {saveStatus.success ? '✓' : '⚠'} {saveStatus.msg}
            </div>
          )}

          {/* Key Items List */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {Object.entries(keyMeta).map(([k, item]) => {
              const isConfigured = item.configured;
              const hasEditValue = editValues[k] !== undefined;
              const currentInput = editValues[k] ?? '';

              return (
                <div
                  key={k}
                  style={{
                    background: 'rgba(15, 23, 42, 0.65)',
                    border: '1px solid rgba(255, 255, 255, 0.08)',
                    borderRadius: 8,
                    padding: '14px',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 8,
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span style={{ fontFamily: 'monospace', fontWeight: 'bold', fontSize: 13, color: '#f1f5f9' }}>
                        {item.title}
                      </span>
                      <span style={{ fontSize: 11, color: '#94a3b8', background: 'rgba(255,255,255,0.06)', padding: '2px 6px', borderRadius: 4 }}>
                        {item.service}
                      </span>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span
                        style={{
                          fontSize: 11,
                          fontWeight: 'bold',
                          padding: '2px 8px',
                          borderRadius: 4,
                          background: isConfigured ? 'rgba(34, 197, 94, 0.2)' : 'rgba(239, 68, 68, 0.15)',
                          color: isConfigured ? '#4ade80' : '#f87171',
                          border: `1px solid ${isConfigured ? 'rgba(34, 197, 94, 0.4)' : 'rgba(239, 68, 68, 0.3)'}`,
                          fontFamily: 'monospace',
                        }}
                      >
                        {isConfigured ? 'ACTIVE' : 'NOT CONFIGURED'}
                      </span>
                    </div>
                  </div>

                  <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.4 }}>
                    {item.description}
                  </div>

                  {/* Masked status if currently set */}
                  {isConfigured && !hasEditValue && item.maskedValue && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, fontFamily: 'monospace', color: '#38bdf8' }}>
                      <span>Current Saved Value:</span>
                      <code style={{ background: '#020617', padding: '2px 6px', borderRadius: 4 }}>
                        {item.maskedValue}
                      </code>
                      <span style={{ color: '#64748b' }}>({item.length} chars)</span>
                    </div>
                  )}

                  {/* Key Input */}
                  <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
                    <input
                      type="password"
                      placeholder={isConfigured ? 'Enter new value to update...' : item.placeholder}
                      value={currentInput}
                      onChange={(e) => handleInputChange(k, e.target.value)}
                      style={{
                        flex: 1,
                        background: '#020617',
                        border: '1px solid rgba(139, 92, 246, 0.3)',
                        borderRadius: 6,
                        padding: '6px 12px',
                        color: '#f8fafc',
                        fontFamily: 'monospace',
                        fontSize: 12,
                        outline: 'none',
                      }}
                    />

                    {/* Action buttons per key */}
                    {k === 'AISSTREAM_API_KEY' && (
                      <button
                        onClick={handleTestAisStream}
                        disabled={aisTestStatus.testing}
                        style={{
                          background: '#1e293b',
                          color: '#38bdf8',
                          border: '1px solid rgba(56, 189, 248, 0.4)',
                          borderRadius: 6,
                          padding: '4px 12px',
                          fontSize: 11,
                          fontFamily: 'monospace',
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: 6,
                        }}
                      >
                        {aisTestStatus.testing ? 'Testing...' : 'Test Feed'}
                      </button>
                    )}

                    {k === 'CESIUM_ION_TOKEN' && (
                      <button
                        onClick={handleTestCesiumIon}
                        disabled={cesiumTestStatus.testing}
                        style={{
                          background: '#1e293b',
                          color: '#c084fc',
                          border: '1px solid rgba(192, 132, 252, 0.4)',
                          borderRadius: 6,
                          padding: '4px 12px',
                          fontSize: 11,
                          fontFamily: 'monospace',
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: 6,
                        }}
                      >
                        {cesiumTestStatus.testing ? 'Testing...' : 'Test Token'}
                      </button>
                    )}
                  </div>

                  {/* Diagnostic Output for AISStream */}
                  {k === 'AISSTREAM_API_KEY' && aisTestStatus.result && (
                    <div
                      style={{
                        fontSize: 11,
                        fontFamily: 'monospace',
                        padding: '6px 10px',
                        borderRadius: 4,
                        background: aisTestStatus.success ? 'rgba(34, 197, 94, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                        color: aisTestStatus.success ? '#4ade80' : '#f87171',
                        border: `1px solid ${aisTestStatus.success ? '#22c55e' : '#ef4444'}`,
                      }}
                    >
                      {aisTestStatus.result}
                    </div>
                  )}

                  {/* Diagnostic Output for Cesium Ion */}
                  {k === 'CESIUM_ION_TOKEN' && cesiumTestStatus.result && (
                    <div
                      style={{
                        fontSize: 11,
                        fontFamily: 'monospace',
                        padding: '6px 10px',
                        borderRadius: 4,
                        background: cesiumTestStatus.success ? 'rgba(34, 197, 94, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                        color: cesiumTestStatus.success ? '#4ade80' : '#f87171',
                        border: `1px solid ${cesiumTestStatus.success ? '#22c55e' : '#ef4444'}`,
                      }}
                    >
                      {cesiumTestStatus.result}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Modal Footer */}
        <div
          style={{
            padding: '14px 20px',
            borderTop: '1px solid rgba(255, 255, 255, 0.08)',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            background: 'rgba(15, 23, 42, 0.95)',
          }}
        >
          <div style={{ fontSize: 11, color: '#64748b' }}>
            Keys are securely stored in server environment &amp; mirrored to local .env
          </div>

          <div style={{ display: 'flex', gap: 10 }}>
            <button
              onClick={onClose}
              style={{
                background: '#1e293b',
                color: '#cbd5e1',
                border: '1px solid rgba(255, 255, 255, 0.15)',
                borderRadius: 6,
                padding: '6px 14px',
                fontSize: 12,
                fontFamily: 'monospace',
                cursor: 'pointer',
              }}
            >
              Close
            </button>
            <button
              onClick={handleSaveKeys}
              disabled={isSaving || Object.keys(editValues).length === 0}
              style={{
                background: Object.keys(editValues).length > 0 ? '#8b5cf6' : '#475569',
                color: '#ffffff',
                border: 'none',
                borderRadius: 6,
                padding: '6px 16px',
                fontSize: 12,
                fontWeight: 'bold',
                fontFamily: 'monospace',
                cursor: Object.keys(editValues).length > 0 ? 'pointer' : 'not-allowed',
                display: 'flex',
                alignItems: 'center',
                gap: 6,
              }}
            >
              {isSaving ? 'Saving...' : '💾 Save Keys to Environment'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ApiKeySetupModal;
