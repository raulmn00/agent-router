import type { CSSProperties } from "react";

import { INTENT_COLOR_VAR, isKnownIntent, type IntentStats } from "../api";

interface IntentDistributionChartProps {
  intents: Record<string, IntentStats>;
}

// Horizontal SVG bar chart. Each bar is colored by INTENT_COLOR_VAR when the
// intent is in our known set; unknown intents (e.g. a future class added on
// the backend before the frontend is updated) get the muted text color.
export function IntentDistributionChart({ intents }: IntentDistributionChartProps) {
  const entries = Object.entries(intents).sort((a, b) => b[1].count - a[1].count);

  if (entries.length === 0) {
    return (
      <div className="stats-empty">
        No intents observed yet. Run a few requests against{" "}
        <code>POST /route</code> from the <strong>Roteamento</strong> tab and
        come back — auto-refresh polls every 5s.
      </div>
    );
  }

  const max = Math.max(...entries.map(([, s]) => s.count), 1);
  const ROW_H = 28;
  const GAP = 6;
  const LABEL_W = 130;
  const VAL_W = 70;

  return (
    <div className="stats-bar-chart" role="figure" aria-label="Intent distribution">
      <svg
        viewBox={`0 0 600 ${entries.length * (ROW_H + GAP)}`}
        preserveAspectRatio="none"
        width="100%"
        height={entries.length * (ROW_H + GAP)}
      >
        {entries.map(([intent, stats], i) => {
          const y = i * (ROW_H + GAP);
          const barX = LABEL_W;
          const barMaxW = 600 - LABEL_W - VAL_W;
          const barW = (stats.count / max) * barMaxW;
          const fill: CSSProperties["fill"] = isKnownIntent(intent)
            ? INTENT_COLOR_VAR[intent]
            : "var(--text-muted)";
          return (
            <g key={intent}>
              <text
                x={LABEL_W - 8}
                y={y + ROW_H / 2}
                textAnchor="end"
                dominantBaseline="middle"
                className="stats-bar-chart__label"
              >
                {intent}
              </text>
              <rect
                x={barX}
                y={y + 4}
                width={Math.max(barW, 1)}
                height={ROW_H - 8}
                rx={4}
                style={{ fill }}
              />
              <text
                x={barX + barW + 8}
                y={y + ROW_H / 2}
                dominantBaseline="middle"
                className="stats-bar-chart__value"
              >
                {stats.count} · {(stats.share * 100).toFixed(1)}%
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
