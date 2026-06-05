import { useEffect, useMemo, useRef, useState } from "react";

import type { CompareResponse } from "../api";

import { RouterColumn } from "./RouterColumn";
import { ScoreBanner } from "./ScoreBanner";

interface ComparisonGridProps {
  data: CompareResponse;
}

/**
 * Schedule a "staggered arrival" reveal of each column ordered by measured
 * latency. The backend returns everything at once, but the LLM/embed paths
 * genuinely took longer than DistilBERT in real life — surfacing that as
 * sequential reveals makes the speed difference legible. This is a
 * presentation animation derived from the MEASURED latency_ms; we never
 * fabricate timing.
 */
function buildRevealSchedule(data: CompareResponse): Map<string, number> {
  const ok = data.results.filter((r) => r.error === null);
  const errored = data.results.filter((r) => r.error !== null);

  const schedule = new Map<string, number>();

  // Errored rows reveal immediately — no point in delaying a failure message.
  for (const r of errored) {
    schedule.set(r.router_name, 0);
  }

  if (ok.length === 0) return schedule;

  // Sort by latency and map to a 200–1400ms window so the animation respects
  // the relative ordering without dragging if the LLM took 30s.
  const sorted = [...ok].sort((a, b) => a.latency_ms - b.latency_ms);
  const minLat = sorted[0]?.latency_ms ?? 0;
  const maxLat = sorted[sorted.length - 1]?.latency_ms ?? 0;
  const range = maxLat - minLat;

  const FIRST = 150;
  const LAST = 1400;

  for (const r of sorted) {
    const relative = range > 0 ? (r.latency_ms - minLat) / range : 0;
    schedule.set(r.router_name, FIRST + relative * (LAST - FIRST));
  }

  return schedule;
}

export function ComparisonGrid({ data }: ComparisonGridProps) {
  const [revealed, setRevealed] = useState<Set<string>>(new Set());
  const timeoutsRef = useRef<number[]>([]);

  const schedule = useMemo(() => buildRevealSchedule(data), [data]);

  useEffect(() => {
    // Reset on every new response, then schedule reveals.
    setRevealed(new Set());
    timeoutsRef.current.forEach((id) => window.clearTimeout(id));
    timeoutsRef.current = [];

    for (const [routerName, delay] of schedule.entries()) {
      const id = window.setTimeout(() => {
        setRevealed((prev) => {
          if (prev.has(routerName)) return prev;
          const next = new Set(prev);
          next.add(routerName);
          return next;
        });
      }, delay);
      timeoutsRef.current.push(id);
    }

    return () => {
      timeoutsRef.current.forEach((id) => window.clearTimeout(id));
      timeoutsRef.current = [];
    };
  }, [schedule]);

  // Fastest / divergent decoration is computed once per response.
  const okResults = data.results.filter((r) => r.error === null);
  const fastest = okResults.length
    ? okResults.reduce((a, b) => (a.latency_ms <= b.latency_ms ? a : b))
    : null;
  const intents = new Set(okResults.map((r) => r.intent));
  const divergent = intents.size > 1;

  return (
    <>
      <div className="compare-grid" aria-live="polite">
        {data.results.map((r) => (
          <RouterColumn
            key={r.router_name}
            result={r}
            revealed={revealed.has(r.router_name)}
            isFastest={fastest?.router_name === r.router_name}
            isDivergent={divergent}
            subtitle={subtitleFor(r.router_name)}
          />
        ))}
      </div>
      <ScoreBanner data={data} />
    </>
  );
}

function subtitleFor(name: string): string {
  if (name.startsWith("DistilBERT")) return "local · CPU/MPS · fine-tuned";
  if (name.startsWith("LLM")) return "API · gpt-4o-mini · zero-shot";
  if (name.startsWith("Embeddings")) return "API · embedding-3-small + LogReg";
  return "";
}
