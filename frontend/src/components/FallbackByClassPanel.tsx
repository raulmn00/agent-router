import type { CSSProperties } from "react";

import { INTENT_COLOR_VAR, isKnownIntent, type FallbackStats } from "../api";

interface FallbackByClassPanelProps {
  fallbacks: FallbackStats;
  totalRequests: number;
}

// THIS is the panel that validates the per-class thresholds. If `chitchat`
// (or any class) shows up here disproportionately, its threshold is
// probably wrong. The defaults are chitchat=0.45 and 0.65 for the others;
// see backend/app/dispatch.py.
export function FallbackByClassPanel({
  fallbacks,
  totalRequests,
}: FallbackByClassPanelProps) {
  const entries = Object.entries(fallbacks.by_attempted_intent).sort(
    (a, b) => b[1] - a[1],
  );
  const overallRate = totalRequests > 0 ? fallbacks.rate : 0;

  return (
    <div className="card card--accent stats-fallback">
      <div className="stats-fallback__header">
        <span className="stats-section__label">Calibration signal</span>
        <h3 className="stats-section__title">Fallback by attempted intent</h3>
        <p className="stats-section__hint">
          The intent the classifier picked <em>before</em> the per-class
          threshold gate kicked in. If a class dominates this list, its
          threshold is likely mis-calibrated — empirical defaults are{" "}
          <code>chitchat = 0.45</code>, others <code>= 0.65</code>.
        </p>
      </div>

      <div className="stats-fallback__totals">
        <span>
          <span className="stats-kv__label">Total fallbacks</span>
          <span className="stats-kv__value">{fallbacks.total}</span>
        </span>
        <span className="stats-fallback__divider" aria-hidden="true" />
        <span>
          <span className="stats-kv__label">Overall rate</span>
          <span className="stats-kv__value">{(overallRate * 100).toFixed(1)}%</span>
        </span>
      </div>

      {entries.length === 0 ? (
        <p className="stats-empty">
          No fallbacks observed yet — every classified request stayed above its
          per-class threshold.
        </p>
      ) : (
        <ul className="stats-fallback__list">
          {entries.map(([intent, count]) => {
            const shareOfFallbacks =
              fallbacks.total > 0 ? count / fallbacks.total : 0;
            const colorStyle: CSSProperties = isKnownIntent(intent)
              ? { ["--row-color" as string]: INTENT_COLOR_VAR[intent] }
              : { ["--row-color" as string]: "var(--text-muted)" };
            return (
              <li key={intent} className="stats-fallback__row" style={colorStyle}>
                <span className="stats-fallback__intent">{intent}</span>
                <div className="stats-fallback__bar">
                  <div
                    className="stats-fallback__bar-fill"
                    style={{ width: `${shareOfFallbacks * 100}%` }}
                  />
                </div>
                <span className="stats-fallback__count">
                  {count} · {(shareOfFallbacks * 100).toFixed(1)}%
                </span>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
