import type { RouterResult } from "../api";

import { ConfidenceGauge } from "./ConfidenceGauge";
import { IntentBadge } from "./IntentBadge";

interface RouterColumnProps {
  result: RouterResult;
  revealed: boolean;
  isFastest: boolean;
  isDivergent: boolean;
  subtitle?: string;
}

function formatLatency(ms: number): string {
  if (ms === 0) return "—";
  if (ms < 10) return ms.toFixed(2);
  if (ms < 1000) return ms.toFixed(1);
  return ms.toFixed(0);
}

function formatCost(usd: number): string {
  if (usd === 0) return "$0.0000";
  if (usd < 0.001) return `$${usd.toFixed(6)}`;
  return `$${usd.toFixed(4)}`;
}

export function RouterColumn({
  result,
  revealed,
  isFastest,
  isDivergent,
  subtitle,
}: RouterColumnProps) {
  const errored = result.error !== null;
  const classes = [
    "router-column",
    revealed ? "router-column--revealed" : "",
    errored ? "router-column--error" : "",
    !errored && isFastest ? "router-column--fastest" : "",
    !errored && isDivergent ? "router-column--divergent" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={classes} aria-busy={!revealed}>
      <div className="router-column__head">
        <h3 className="router-column__title">{result.router_name}</h3>
        {subtitle ? <span className="router-column__subtitle">{subtitle}</span> : null}
      </div>

      {errored ? (
        <div className="router-column__error" role="alert">
          <strong>Sem resultado</strong>
          <span>{result.error}</span>
        </div>
      ) : !revealed ? (
        <>
          <div className="router-column__skeleton" />
          <div className="router-column__skeleton" style={{ width: "60%" }} />
          <div className="router-column__skeleton" style={{ width: "40%" }} />
        </>
      ) : (
        <>
          <div>
            <div className="router-column__metric-label">Intent</div>
            <div style={{ marginTop: 6 }}>
              <IntentBadge intent={result.intent} />
            </div>
          </div>

          <ConfidenceGauge value={result.confidence} size="sm" />

          <div className="router-column__metric">
            <span className="router-column__metric-label">Latência</span>
            <span className="router-column__metric-value router-column__metric-value--lg">
              {formatLatency(result.latency_ms)}
              <span
                style={{ fontSize: 14, color: "var(--text-soft)", marginLeft: 6 }}
              >
                ms
              </span>
            </span>
          </div>

          <div className="router-column__metric">
            <span className="router-column__metric-label">Custo / 1k</span>
            <span className="router-column__metric-value">{formatCost(result.cost_per_1k_usd)}</span>
            <span className="router-column__metric-sub">por 1000 classificações</span>
          </div>
        </>
      )}
    </div>
  );
}
