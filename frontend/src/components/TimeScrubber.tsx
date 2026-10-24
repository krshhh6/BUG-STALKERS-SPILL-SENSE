import React, { useState, useEffect, useRef } from 'react';

interface TimeScrubberProps {
  onTimeChange?: (hours: number) => void;
}

export const TimeScrubber: React.FC<TimeScrubberProps> = ({ onTimeChange }) => {
  const [scrubHours, setScrubHours] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const playTimerRef = useRef<number | null>(null);

  // Dynamically anchored to current live time
  const [baseDate] = useState<Date>(() => new Date());

  const getReadoutText = (hrs: number) => {
    const tDate = new Date(baseDate.getTime() + hrs * 3600 * 1000);
    const dateStr = tDate.toISOString().replace('T', ' ').substring(0, 16) + ' UTC';
    if (hrs === 0) return `T = 0h (SAR Detection: ${dateStr})`;
    if (hrs >= -24 && hrs <= -20) {
      return `T = ${hrs}h (CRITICAL INTERCEPT: ${dateStr}) [P=50%]`;
    }
    return `T = ${hrs}h (Lagrangian Step: ${dateStr})`;
  };

  const handleSliderChange = (val: number) => {
    setScrubHours(val);
    if (onTimeChange) onTimeChange(val);
  };

  const stepScrubber = (step: number) => {
    setScrubHours((prev) => {
      const next = Math.min(0, Math.max(-72, prev + step));
      if (onTimeChange) onTimeChange(next);
      return next;
    });
  };

  const resetScrubber = () => {
    setScrubHours(0);
    setIsPlaying(false);
    if (onTimeChange) onTimeChange(0);
  };

  const togglePlayback = () => {
    setIsPlaying((prev) => !prev);
  };

  useEffect(() => {
    if (isPlaying) {
      playTimerRef.current = window.setInterval(() => {
        setScrubHours((prev) => {
          if (prev >= 0) {
            return -72; // loop back
          }
          const next = prev + 1;
          if (onTimeChange) onTimeChange(next);
          return next;
        });
      }, 350);
    } else if (playTimerRef.current) {
      clearInterval(playTimerRef.current);
      playTimerRef.current = null;
    }

    return () => {
      if (playTimerRef.current) clearInterval(playTimerRef.current);
    };
  }, [isPlaying]);

  return (
    <div className="time-scrubber-bar" style={{ padding: '6px 24px' }}>
      <div className="scrubber-controls">
        <div className="scrubber-playback">
          <button
            className="scrub-btn"
            onClick={() => stepScrubber(-6)}
            title="Step -6h"
          >
            <span className="material-symbols-outlined" style={{ fontSize: 16 }}>replay_10</span>
          </button>
          <button
            className="scrub-btn"
            onClick={togglePlayback}
            title={isPlaying ? 'Pause' : 'Play'}
          >
            <span className="material-symbols-outlined" style={{ fontSize: 18 }}>
              {isPlaying ? 'pause' : 'play_arrow'}
            </span>
          </button>
          <button
            className="scrub-btn"
            onClick={() => stepScrubber(6)}
            title="Step +6h"
          >
            <span className="material-symbols-outlined" style={{ fontSize: 16 }}>forward_10</span>
          </button>
          <button
            className="scrub-btn"
            onClick={resetScrubber}
            title="Jump to T=0 Detection"
          >
            <span className="material-symbols-outlined" style={{ fontSize: 16 }}>skip_next</span>
          </button>
        </div>

        <div className="scrubber-readout" id="scrub-readout">
          {getReadoutText(scrubHours)}
        </div>
        <div className="text-xs text-muted">Lagrangian Backward Drift Timeline</div>
      </div>

      <div className="scrubber-track-wrap">
        <span className="scrub-label">T - 72h (Discharge Window)</span>
        <input
          type="range"
          min="-72"
          max="0"
          value={scrubHours}
          step="1"
          className="scrubber-slider"
          id="time-slider"
          onChange={(e) => handleSliderChange(parseInt(e.target.value, 10))}
        />
        <span className="scrub-label">T=0 (Detection)</span>
      </div>
    </div>
  );
};
