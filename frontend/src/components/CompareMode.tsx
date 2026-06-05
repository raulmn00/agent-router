import { useState } from "react";

import { ApiError, postCompare, type CompareResponse } from "../api";

import { ComparisonGrid } from "./ComparisonGrid";
import { MessageInput } from "./MessageInput";

export function CompareMode() {
  const [data, setData] = useState<CompareResponse | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [busy, setBusy] = useState(false);

  const onSubmit = async (text: string) => {
    setBusy(true);
    setError(null);
    try {
      const resp = await postCompare(text);
      setData(resp);
    } catch (e) {
      if (e instanceof ApiError) {
        setError(e);
      } else {
        setError(new ApiError("unknown", null, "Erro inesperado."));
      }
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <MessageInput
        placeholder="Digite uma mensagem para comparar os três roteadores no mesmo input."
        submitLabel={busy ? "Comparando…" : "Comparar"}
        busy={busy}
        onSubmit={onSubmit}
      />

      {error ? <ErrorNotice err={error} /> : null}

      {data ? (
        <ComparisonGrid key={data.input + data.results.length} data={data} />
      ) : !error && !busy ? (
        <EmptyState />
      ) : null}
    </>
  );
}

function EmptyState() {
  return (
    <div className="notice">
      <p className="notice__title">Compare os três roteadores no mesmo input</p>
      <p className="notice__body">
        Cada um vai classificar a mesma mensagem; o frontend mostra a intenção
        prevista, a confiança, a latência medida e o custo por 1k chamadas. O
        DistilBERT roda local; o LLM zero-shot e o embeddings+LogReg passam por
        APIs. Endpoint limitado a 10 comparações por minuto.
      </p>
    </div>
  );
}

function ErrorNotice({ err }: { err: ApiError }) {
  const variant = err.kind === "rate_limit" ? "notice--rate-limit" : "notice--error";
  const title =
    err.kind === "rate_limit"
      ? "Calma aí — muitas comparações seguidas"
      : "Não consegui comparar";
  return (
    <div className={`notice ${variant}`} role="alert">
      <p className="notice__title">{title}</p>
      <p className="notice__body">{err.message}</p>
    </div>
  );
}
