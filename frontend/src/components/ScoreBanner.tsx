import type { CompareResponse } from "../api";

interface ScoreBannerProps {
  data: CompareResponse;
}

export function ScoreBanner({ data }: ScoreBannerProps) {
  const ok = data.results.filter((r) => r.error === null);
  if (ok.length === 0) {
    return (
      <div className="score-banner" role="status">
        <span className="score-banner__item">
          <span className="score-banner__label">Resultado</span>
          <span className="score-banner__value">
            Nenhum roteador respondeu com sucesso
          </span>
        </span>
      </div>
    );
  }

  const agreementText = data.agreement
    ? `Os ${ok.length} concordaram na intenção`
    : "Divergiram na intenção";

  return (
    <div className="score-banner" role="status">
      <span className="score-banner__item">
        <span className="score-banner__label">Acordo</span>
        <span
          className={[
            "score-banner__value",
            data.agreement ? "score-banner__value--positive" : "score-banner__value--neutral",
          ].join(" ")}
        >
          {agreementText}
        </span>
      </span>
      <span className="score-banner__divider" aria-hidden="true" />
      <span className="score-banner__item">
        <span className="score-banner__label">Mais rápido</span>
        <span className="score-banner__value">{data.fastest || "—"}</span>
      </span>
      <span className="score-banner__divider" aria-hidden="true" />
      <span className="score-banner__item">
        <span className="score-banner__label">Mais barato</span>
        <span className="score-banner__value">{data.cheapest || "—"}</span>
      </span>
    </div>
  );
}
