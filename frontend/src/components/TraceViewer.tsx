import { useState } from "react";

interface TraceViewerProps {
  trace: string[];
  featured?: boolean;
  defaultOpen?: boolean;
  itemCountLabel?: string;
}

export function TraceViewer({
  trace,
  featured = false,
  defaultOpen = false,
  itemCountLabel = "items",
}: TraceViewerProps) {
  const [open, setOpen] = useState(defaultOpen);
  const summary = featured
    ? `Trace Planner → Executors → Critic (${trace.length} ${itemCountLabel})`
    : `Trace de dispatch (${trace.length} ${itemCountLabel})`;

  return (
    <div
      className={[
        "trace",
        featured ? "trace--featured" : "",
        open ? "trace--open" : "",
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <button
        type="button"
        className="trace__toggle"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        <span>{summary}</span>
        <span className="trace__caret" aria-hidden="true">
          ▶
        </span>
      </button>
      {open ? (
        <div className="trace__body">
          {trace.length === 0 ? (
            <span style={{ color: "var(--text-soft)" }}>(empty trace)</span>
          ) : (
            trace.map((line, i) => (
              <div key={i} className="trace__item">
                {line}
              </div>
            ))
          )}
        </div>
      ) : null}
    </div>
  );
}
