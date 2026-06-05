export type Mode = "route" | "compare";

interface ModeToggleProps {
  mode: Mode;
  onChange: (next: Mode) => void;
}

export function ModeToggle({ mode, onChange }: ModeToggleProps) {
  return (
    <div className="mode-toggle" role="tablist" aria-label="Modo">
      <button
        type="button"
        role="tab"
        aria-selected={mode === "route"}
        className={`mode-toggle__btn${mode === "route" ? " mode-toggle__btn--active" : ""}`}
        onClick={() => onChange("route")}
      >
        Roteamento
      </button>
      <button
        type="button"
        role="tab"
        aria-selected={mode === "compare"}
        className={`mode-toggle__btn${mode === "compare" ? " mode-toggle__btn--active" : ""}`}
        onClick={() => onChange("compare")}
      >
        Comparação
      </button>
    </div>
  );
}
