import type { CSSProperties } from "react";

interface ConfidenceGaugeProps {
  value: number; // 0..1
  size?: "sm" | "md";
  label?: string;
}

function colorFor(value: number): string {
  if (value >= 0.8) return "var(--confidence-high)";
  if (value >= 0.5) return "var(--confidence-mid)";
  return "var(--confidence-low)";
}

export function ConfidenceGauge({ value, size = "md", label = "Confidence" }: ConfidenceGaugeProps) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  const style: CSSProperties = { ["--gauge-color" as string]: colorFor(value), width: `${pct}%` };

  return (
    <div className="gauge" data-size={size}>
      <div className="gauge__top">
        <span className="gauge__label">{label}</span>
        <span className="gauge__value">{pct.toFixed(1)}%</span>
      </div>
      <div
        className="gauge__track"
        role="progressbar"
        aria-valuenow={Number(pct.toFixed(1))}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={label}
      >
        <div className="gauge__fill" style={style} />
      </div>
    </div>
  );
}
