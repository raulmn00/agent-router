import type { CSSProperties } from "react";

import { INTENT_COLOR_VAR, isKnownIntent, type IntentStats } from "../api";

interface ConfidenceByIntentTableProps {
  intents: Record<string, IntentStats>;
}

function formatConf(value: number): string {
  return value.toFixed(3);
}

// Per-intent confidence stats (min / p50 / mean / p95 / max + n samples). The
// `confidence` bucket on IntentStats is `null` until the first sample lands —
// we render an em-dash row in that case so the column count stays consistent.
export function ConfidenceByIntentTable({ intents }: ConfidenceByIntentTableProps) {
  const entries = Object.entries(intents).sort((a, b) => a[0].localeCompare(b[0]));

  if (entries.length === 0) {
    return (
      <div className="stats-empty">
        No confidence samples yet — same condition as the intent distribution.
      </div>
    );
  }

  return (
    <table className="stats-table">
      <thead>
        <tr>
          <th>intent</th>
          <th className="num">min</th>
          <th className="num">p50</th>
          <th className="num">mean</th>
          <th className="num">p95</th>
          <th className="num">max</th>
          <th className="num">n</th>
        </tr>
      </thead>
      <tbody>
        {entries.map(([intent, stats]) => {
          const swatch: CSSProperties = isKnownIntent(intent)
            ? { background: INTENT_COLOR_VAR[intent] }
            : { background: "var(--text-muted)" };

          const c = stats.confidence;
          return (
            <tr key={intent}>
              <td>
                <span className="stats-table__swatch" style={swatch} />
                <code>{intent}</code>
              </td>
              {c === null ? (
                <>
                  <td className="num">—</td>
                  <td className="num">—</td>
                  <td className="num">—</td>
                  <td className="num">—</td>
                  <td className="num">—</td>
                  <td className="num">0</td>
                </>
              ) : (
                <>
                  <td className="num">{formatConf(c.min)}</td>
                  <td className="num">{formatConf(c.p50)}</td>
                  <td className="num">{formatConf(c.mean)}</td>
                  <td className="num">{formatConf(c.p95)}</td>
                  <td className="num">{formatConf(c.max)}</td>
                  <td className="num">{c.count}</td>
                </>
              )}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
