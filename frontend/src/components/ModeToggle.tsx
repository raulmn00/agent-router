export type Mode = "route" | "compare" | "stats";

const TABS: ReadonlyArray<{ mode: Mode; label: string }> = [
  { mode: "route", label: "Roteamento" },
  { mode: "compare", label: "Comparação" },
  { mode: "stats", label: "Stats" },
];

interface ModeToggleProps {
  mode: Mode;
  onChange: (next: Mode) => void;
}

export function ModeToggle({ mode, onChange }: ModeToggleProps) {
  return (
    <div className="mode-toggle" role="tablist" aria-label="Modo">
      {TABS.map((tab) => (
        <button
          key={tab.mode}
          type="button"
          role="tab"
          aria-selected={mode === tab.mode}
          className={`mode-toggle__btn${mode === tab.mode ? " mode-toggle__btn--active" : ""}`}
          onClick={() => onChange(tab.mode)}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
