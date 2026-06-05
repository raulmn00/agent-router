import type { RouteResponse } from "../api";

import { ConfidenceGauge } from "./ConfidenceGauge";
import { IntentBadge } from "./IntentBadge";
import { PathFlow } from "./PathFlow";
import { TraceViewer } from "./TraceViewer";

interface RouteResultProps {
  data: RouteResponse;
}

export function RouteResult({ data }: RouteResultProps) {
  const isComplex = data.intent === "complex_task";

  return (
    <div className={`card${isComplex ? " card--accent" : ""} result-card`}>
      <div className="result-card__header">
        <div className="result-card__intent">
          <IntentBadge intent={data.intent} size="lg" />
          <span style={{ color: "var(--text-soft)", fontSize: 12.5 }}>
            {data.path_taken}
          </span>
        </div>
        <div style={{ minWidth: 220, flex: "0 0 auto" }}>
          <ConfidenceGauge value={data.confidence} />
        </div>
      </div>

      <PathFlow intent={data.intent} pathTaken={data.path_taken} />

      <div className="result-card__answer">{data.answer}</div>

      <TraceViewer
        trace={data.trace}
        featured={isComplex}
        defaultOpen={isComplex && data.trace.length > 0}
      />
    </div>
  );
}
