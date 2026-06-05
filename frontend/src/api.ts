// Centralized API client + types — mirror of backend/app/schemas.py.

const API_URL = (import.meta.env.VITE_API_URL ?? "http://localhost:8000").replace(
  /\/+$/,
  "",
);

// --------------------------------------------------------------------------- //
// Domain types                                                                 //
// --------------------------------------------------------------------------- //

export const INTENTS = [
  "simple_qa",
  "complex_task",
  "document_qa",
  "chitchat",
] as const;

export type Intent = (typeof INTENTS)[number];

export interface RouteRequest {
  input: string;
}

export interface RouteResponse {
  intent: Intent;
  confidence: number;
  answer: string;
  path_taken: string;
  trace: string[];
}

export interface CompareRequest {
  input: string;
}

export interface RouterResult {
  router_name: string;
  intent: string; // empty string when error !== null
  confidence: number;
  latency_ms: number;
  cost_per_1k_usd: number;
  error: string | null;
}

export interface CompareResponse {
  input: string;
  results: RouterResult[];
  agreement: boolean;
  fastest: string;
  cheapest: string;
}

// --------------------------------------------------------------------------- //
// Errors                                                                       //
// --------------------------------------------------------------------------- //

export type ApiErrorKind =
  | "network"
  | "rate_limit" // 429
  | "validation" // 422
  | "provider" // 503
  | "server" // 500
  | "unknown";

export class ApiError extends Error {
  constructor(
    public readonly kind: ApiErrorKind,
    public readonly status: number | null,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function userFacingMessage(kind: ApiErrorKind): string {
  switch (kind) {
    case "network":
      return "Não foi possível conectar ao backend. Verifique se o serviço está no ar.";
    case "rate_limit":
      return "Muitas requisições seguidas. Aguarde um minuto antes de tentar de novo.";
    case "validation":
      return "Entrada inválida. Tente reformular a mensagem.";
    case "provider":
      return "O backend está sem credenciais de provedor LLM. Não é possível responder agora.";
    case "server":
      return "Erro interno no servidor. Tente novamente em instantes.";
    default:
      return "Algo deu errado. Tente novamente.";
  }
}

function kindFromStatus(status: number): ApiErrorKind {
  if (status === 429) return "rate_limit";
  if (status === 422) return "validation";
  if (status === 503) return "provider";
  if (status >= 500) return "server";
  return "unknown";
}

// --------------------------------------------------------------------------- //
// Fetch wrappers                                                               //
// --------------------------------------------------------------------------- //

async function postJson<TResponse>(
  path: string,
  body: unknown,
): Promise<TResponse> {
  let resp: Response;
  try {
    resp = await fetch(`${API_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (e) {
    throw new ApiError("network", null, userFacingMessage("network"));
  }

  if (!resp.ok) {
    const kind = kindFromStatus(resp.status);
    throw new ApiError(kind, resp.status, userFacingMessage(kind));
  }

  return (await resp.json()) as TResponse;
}

export async function postRoute(input: string): Promise<RouteResponse> {
  return postJson<RouteResponse>("/route", { input } satisfies RouteRequest);
}

export async function postCompare(input: string): Promise<CompareResponse> {
  return postJson<CompareResponse>("/compare", { input } satisfies CompareRequest);
}

// --------------------------------------------------------------------------- //
// Helpers used by the UI                                                       //
// --------------------------------------------------------------------------- //

/** Stable color hint per intent — kept here so it travels with the type. */
export const INTENT_COLOR_VAR: Record<Intent, string> = {
  simple_qa: "var(--intent-simple-qa)",
  complex_task: "var(--intent-complex-task)",
  document_qa: "var(--intent-document-qa)",
  chitchat: "var(--intent-chitchat)",
};

export const INTENT_LABEL: Record<Intent, string> = {
  simple_qa: "simple_qa",
  complex_task: "complex_task",
  document_qa: "document_qa",
  chitchat: "chitchat",
};

export function isKnownIntent(value: string): value is Intent {
  return (INTENTS as readonly string[]).includes(value);
}
