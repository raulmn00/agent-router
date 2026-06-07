import { useState } from "react";

import { CompareMode } from "./components/CompareMode";
import { ModeToggle, type Mode } from "./components/ModeToggle";
import { RouteMode } from "./components/RouteMode";
import { StatsMode } from "./components/StatsMode";

function renderMode(mode: Mode) {
  switch (mode) {
    case "route":
      return <RouteMode />;
    case "compare":
      return <CompareMode />;
    case "stats":
      return <StatsMode />;
  }
}

export function App() {
  const [mode, setMode] = useState<Mode>("route");
  const isWide = mode !== "route";

  return (
    <div className="shell">
      <div className={`container${isWide ? " container--wide" : ""}`}>
        <header className="app-header">
          <h1 className="app-title">agent-router</h1>
          <p className="app-subtitle">
            DistilBERT fine-tuned para roteamento de intenção, multi-agent
            orchestration (Planner/Executor/Critic), e comparação lado a lado
            entre roteador local, LLM zero-shot e embeddings+LogReg.
          </p>
        </header>

        <ModeToggle mode={mode} onChange={setMode} />

        {renderMode(mode)}

        <footer className="app-footer">
          <span>
            backend:{" "}
            <code style={{ fontSize: 11 }}>
              {import.meta.env.VITE_API_URL ?? "http://localhost:8000"}
            </code>
          </span>
        </footer>
      </div>
    </div>
  );
}
