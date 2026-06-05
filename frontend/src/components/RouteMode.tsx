import { useState } from "react";

import { ApiError, postRoute, type RouteResponse } from "../api";

import { MessageInput } from "./MessageInput";
import { RouteResult } from "./RouteResult";

export function RouteMode() {
  const [data, setData] = useState<RouteResponse | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [busy, setBusy] = useState(false);

  const onSubmit = async (text: string) => {
    setBusy(true);
    setError(null);
    try {
      const resp = await postRoute(text);
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
        placeholder="Digite uma pergunta, tarefa, referência a documento, ou um cumprimento — o roteador decide o caminho."
        submitLabel={busy ? "Roteando…" : "Rotear"}
        busy={busy}
        onSubmit={onSubmit}
      />

      {error ? <ErrorNotice err={error} /> : null}

      {data ? (
        <RouteResult data={data} />
      ) : !error && !busy ? (
        <EmptyState />
      ) : null}
    </>
  );
}

function EmptyState() {
  return (
    <div className="notice">
      <p className="notice__title">Pronto pra rotear</p>
      <p className="notice__body">
        Tente: <em>“What is the capital of France?”</em>, <em>“Build me a Slack
        bot that summarizes my unread channels”</em>, <em>“In the attached PDF,
        what is the conclusion?”</em>, <em>“hey, how are you?”</em>.
      </p>
    </div>
  );
}

function ErrorNotice({ err }: { err: ApiError }) {
  const variant = err.kind === "rate_limit" ? "notice--rate-limit" : "notice--error";
  return (
    <div className={`notice ${variant}`} role="alert">
      <p className="notice__title">Não consegui processar</p>
      <p className="notice__body">{err.message}</p>
    </div>
  );
}
