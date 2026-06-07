import type { LatencyBucket } from "../api";

interface LatencyByPathTableProps {
  latencies: Record<string, LatencyBucket>;
}

function formatMs(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return "—";
  if (value < 10) return value.toFixed(2);
  if (value < 1000) return value.toFixed(1);
  return `${(value / 1000).toFixed(2)}s`;
}

// Compact table — paths in rows, p50 / p95 / mean / count in columns. A tiny
// inline SVG bar shows the relative magnitude of p95 vs the slowest path, so
// the latency gap between simple_qa:direct_llm and complex_task:orchestrator
// is visible at a glance without dragging in a chart lib.
export function LatencyByPathTable({ latencies }: LatencyByPathTableProps) {
  const entries = Object.entries(latencies).sort((a, b) => b[1].count - a[1].count);

  if (entries.length === 0) {
    return (
      <div className="stats-empty">
        No latency samples yet. Each <code>POST /route</code> records its
        timing under its <code>path_taken</code>.
      </div>
    );
  }

  const maxP95 = Math.max(...entries.map(([, b]) => b.p95), 1);

  return (
    <table className="stats-table">
      <thead>
        <tr>
          <th>path_taken</th>
          <th className="num">p50</th>
          <th className="num">p95</th>
          <th className="num">mean</th>
          <th className="num">n</th>
          <th aria-label="p95 relative to slowest path" />
        </tr>
      </thead>
      <tbody>
        {entries.map(([path, b]) => {
          const barShare = Math.min(1, b.p95 / maxP95);
          return (
            <tr key={path}>
              <td><code>{path}</code></td>
              <td className="num">{formatMs(b.p50)}</td>
              <td className="num"><strong>{formatMs(b.p95)}</strong></td>
              <td className="num">{formatMs(b.mean)}</td>
              <td className="num">{b.count}</td>
              <td className="stats-table__bar-cell">
                <svg width="100%" height="8" preserveAspectRatio="none" viewBox="0 0 100 8">
                  <rect width="100" height="8" rx="2" className="stats-table__bar-bg" />
                  <rect
                    width={Math.max(barShare * 100, 1)}
                    height="8"
                    rx="2"
                    className="stats-table__bar-fill"
                  />
                </svg>
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
