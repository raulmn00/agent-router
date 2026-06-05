import type { CSSProperties } from "react";

import { INTENT_COLOR_VAR, isKnownIntent } from "../api";

const PATHS: Array<{ id: string; label: string; description: string }> = [
  { id: "simple_qa", label: "simple_qa", description: "1 LLM call" },
  { id: "complex_task", label: "complex_task", description: "Orchestrator" },
  { id: "document_qa", label: "document_qa", description: "RAG stub" },
  { id: "chitchat", label: "chitchat", description: "1 LLM call · curto" },
];

interface PathFlowProps {
  intent: string;
  pathTaken: string;
}

export function PathFlow({ intent, pathTaken }: PathFlowProps) {
  // path_taken comes from the backend as e.g. "simple_qa:direct_llm".
  const matchedPath = pathTaken.split(":")[0] ?? "";

  const activeStyle: CSSProperties = isKnownIntent(intent)
    ? { ["--intent-color" as string]: INTENT_COLOR_VAR[intent] }
    : {};

  return (
    <div className="path-flow" aria-label="Caminho de dispatch">
      <span className="path-flow__node">input</span>
      <span className="path-flow__arrow">→</span>
      <span className="path-flow__node path-flow__node--router">DistilBERT</span>
      <span className="path-flow__arrow">→</span>
      {PATHS.map((p, idx) => {
        const isActive = p.id === matchedPath;
        return (
          <span key={p.id} style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
            <span
              className={[
                "path-flow__node",
                isActive ? "path-flow__node--active" : "path-flow__node--dim",
              ].join(" ")}
              style={isActive ? activeStyle : undefined}
              title={p.description}
            >
              {p.label}
            </span>
            {idx < PATHS.length - 1 ? <span className="path-flow__arrow">|</span> : null}
          </span>
        );
      })}
    </div>
  );
}
