import type { CSSProperties } from "react";

import { INTENT_COLOR_VAR, isKnownIntent } from "../api";

interface IntentBadgeProps {
  intent: string;
  size?: "sm" | "lg";
}

export function IntentBadge({ intent, size = "sm" }: IntentBadgeProps) {
  const known = isKnownIntent(intent);
  const style: CSSProperties = known
    ? { ["--intent-color" as string]: INTENT_COLOR_VAR[intent], background: INTENT_COLOR_VAR[intent] }
    : {};

  return (
    <span
      className={[
        "intent-badge",
        size === "lg" ? "intent-badge--lg" : "",
        known ? "" : "intent-badge--unknown",
      ]
        .filter(Boolean)
        .join(" ")}
      style={style}
      title={known ? intent : "intent unknown"}
    >
      {intent || "—"}
    </span>
  );
}
