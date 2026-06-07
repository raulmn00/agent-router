import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, getStats, type StatsResponse } from "../api";

import { ConfidenceByIntentTable } from "./ConfidenceByIntentTable";
import { FallbackByClassPanel } from "./FallbackByClassPanel";
import { IntentDistributionChart } from "./IntentDistributionChart";
import { LatencyByPathTable } from "./LatencyByPathTable";
import { Spinner } from "./Spinner";

const REFRESH_MS = 5000;

function formatUptime(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
}

function formatPct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

interface StatsCardProps {
  label: string;
  value: string;
  sub?: string;
}

function StatsCard({ label, value, sub }: StatsCardProps) {
  return (
    <div className="card stats-card">
      <span className="stats-kv__label">{label}</span>
      <span className="stats-card__value">{value}</span>
      {sub ? <span className="stats-kv__sub">{sub}</span> : null}
    </div>
  );
}

export function StatsMode() {
  const [data, setData] = useState<StatsResponse | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [busy, setBusy] = useState(false);
  const [paused, setPaused] = useState(false);
  const [lastFetched, setLastFetched] = useState<Date | null>(null);

  // Ref for cleanup — the user can flip pause while a request is in flight.
  const inFlight = useRef(false);

  const fetchOnce = useCallback(async () => {
    if (inFlight.current) return;
    inFlight.current = true;
    setBusy(true);
    try {
      const next = await getStats();
      setData(next);
      setError(null);
      setLastFetched(new Date());
    } catch (e) {
      if (e instanceof ApiError) {
        setError(e);
      } else {
        setError(new ApiError("unknown", null, "Erro inesperado ao buscar /stats."));
      }
    } finally {
      setBusy(false);
      inFlight.current = false;
    }
  }, []);

  useEffect(() => {
    void fetchOnce();
  }, [fetchOnce]);

  useEffect(() => {
    if (paused) return;
    const id = window.setInterval(() => {
      void fetchOnce();
    }, REFRESH_MS);
    return () => window.clearInterval(id);
  }, [paused, fetchOnce]);

  // ----- render branches ----------------------------------------------------

  // First load, no data, no error.
  if (data === null && error === null) {
    return (
      <div className="card stats-loading">
        <Spinner size="lg" />
        <span>Fetching /stats…</span>
      </div>
    );
  }

  // First load failed AND we never got any data. Show an error notice but keep
  // the page usable so the user can pause/resume.
  if (data === null && error !== null) {
    return (
      <div className="notice notice--error">
        <p className="notice__title">Não consegui buscar as métricas</p>
        <p className="notice__body">{error.message}</p>
        <button
          type="button"
          className="message-input__btn"
          onClick={() => void fetchOnce()}
          style={{ marginTop: 12 }}
        >
          Tentar novamente
        </button>
      </div>
    );
  }

  // From here on, data is non-null. TS narrows from the check above.
  if (data === null) return null;

  const isEmpty = data.total_requests === 0;
  const hasErrors = Object.keys(data.errors).length > 0;

  return (
    <>
      <div className="stats-toolbar">
        <div>
          <h2 className="stats-toolbar__title">/stats · live</h2>
          <span className="stats-toolbar__sub">
            since {data.since}
            {lastFetched
              ? ` · last fetch ${lastFetched.toLocaleTimeString()}`
              : null}
            {busy ? " · refreshing…" : null}
            {error ? " · last fetch failed (retrying)" : null}
          </span>
        </div>
        <div className="stats-toolbar__actions">
          <button
            type="button"
            className="message-input__btn"
            onClick={() => setPaused((v) => !v)}
            aria-pressed={paused}
          >
            {paused ? "▶ Resume auto-refresh" : "⏸ Pause auto-refresh"}
          </button>
        </div>
      </div>

      {data.ephemeral ? (
        <div className="notice notice--rate-limit" role="status">
          <p className="notice__title">In-memory · ephemeral</p>
          <p className="notice__body">{data.note}</p>
        </div>
      ) : null}

      {isEmpty ? (
        <div className="notice">
          <p className="notice__title">No traffic observed yet</p>
          <p className="notice__body">
            The metrics collector is empty. Use the <strong>Roteamento</strong>{" "}
            tab to fire a few <code>POST /route</code> requests — this page
            auto-refreshes every 5s and the data will populate.
          </p>
        </div>
      ) : null}

      {/* Highlight cards */}
      <div className="stats-cards">
        <StatsCard
          label="Total requests"
          value={String(data.total_requests)}
          sub={`uptime ${formatUptime(data.uptime_seconds)}`}
        />
        <StatsCard
          label="Fallback rate"
          value={isEmpty ? "—" : formatPct(data.fallbacks.rate)}
          sub={`${data.fallbacks.total} of ${data.total_requests}`}
        />
        <StatsCard
          label="Errors"
          value={
            hasErrors
              ? String(
                  Object.values(data.errors).reduce((a, b) => a + b, 0),
                )
              : "0"
          }
          sub={
            hasErrors
              ? Object.entries(data.errors)
                  .map(([code, n]) => `${code}:${n}`)
                  .join(" · ")
              : "no 429 / 500 / 503"
          }
        />
      </div>

      {/* Calibration signal — highlighted */}
      <FallbackByClassPanel
        fallbacks={data.fallbacks}
        totalRequests={data.total_requests}
      />

      {/* Intent distribution */}
      <section className="card stats-section">
        <span className="stats-section__label">Volume</span>
        <h3 className="stats-section__title">Intent distribution</h3>
        <IntentDistributionChart intents={data.intents} />
      </section>

      {/* Latency per path */}
      <section className="card stats-section">
        <span className="stats-section__label">Latency</span>
        <h3 className="stats-section__title">Per path · p50 / p95 / mean</h3>
        <p className="stats-section__hint">
          The big gap between <code>simple_qa:direct_llm</code> (one LLM call,
          sub-second) and <code>complex_task:orchestrator</code> (Planner + N
          parallel Executors + Critic) is the structural latency story.
        </p>
        <LatencyByPathTable latencies={data.latency_ms} />
      </section>

      {/* Confidence per intent */}
      <section className="card stats-section">
        <span className="stats-section__label">Calibration</span>
        <h3 className="stats-section__title">Confidence per intent</h3>
        <p className="stats-section__hint">
          Sanity check on the per-class thresholds. <code>chitchat</code>{" "}
          saturates lower than the other classes — defaults reflect that
          (chitchat 0.45 vs 0.65 for the rest).
        </p>
        <ConfidenceByIntentTable intents={data.intents} />
      </section>
    </>
  );
}
