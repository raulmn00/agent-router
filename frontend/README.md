# frontend

Vite + React 18 + TypeScript demo for `agent-router`. Two modes:

- **Roteamento** — calls `POST /route`, renders the dispatched intent, the
  `path_taken` flow, the answer, and the dispatch trace (expanded
  automatically when intent is `complex_task` to show Planner → Executors →
  Critic).
- **Comparação** — calls `POST /compare` and renders the three routers side
  by side: predicted intent, confidence, **measured** latency, and
  cost-per-1k. Columns reveal in order of latency to make the speed
  difference visible.

No UI library; CSS hand-written with tokens in `src/styles.css`. TypeScript
`strict: true`, no `any` anywhere.

## Configuration

```bash
cp .env.example .env
# edit if backend isn't on localhost:8000
```

Single env var:

| Var | Default | Effect |
|---|---|---|
| `VITE_API_URL` | `http://localhost:8000` | Base URL for `POST /route` and `POST /compare`. |

## Run

```bash
npm install
npm run dev       # http://localhost:5173
npm run build     # tsc -b && vite build → ./dist
npm run preview   # serve the production build for a sanity check
```

`npm run build` runs `tsc -b` first — it fails the build if anything in
`src/` has a type error.

## Layout

```
frontend/
├── index.html
├── package.json
├── tsconfig.json (root) + tsconfig.app.json + tsconfig.node.json
├── vite.config.ts
└── src/
    ├── main.tsx
    ├── App.tsx
    ├── api.ts                # types + fetch wrappers; ApiError taxonomy
    ├── styles.css            # tokens + components
    ├── vite-env.d.ts         # types for import.meta.env
    └── components/
        ├── ModeToggle.tsx
        ├── MessageInput.tsx       # textarea + button + Ctrl/⌘+Enter
        ├── Spinner.tsx
        ├── IntentBadge.tsx        # fixed color per intent
        ├── ConfidenceGauge.tsx    # 0–100% bar, semantic colors
        ├── PathFlow.tsx           # mini-flow: input → DistilBERT → <branch>
        ├── TraceViewer.tsx        # collapsible, featured when complex_task
        ├── RouteResult.tsx        # Mode 1 result card
        ├── RouterColumn.tsx       # one router in the compare grid
        ├── ScoreBanner.tsx        # agreement / fastest / cheapest
        ├── ComparisonGrid.tsx     # Mode 2 grid + staggered reveal
        ├── RouteMode.tsx          # Mode 1 page
        └── CompareMode.tsx        # Mode 2 page
```

## Numbers shown in the UI

Every latency and cost on the Comparison view comes straight from the
backend's `POST /compare` response. No client-side fabrication.

If the backend can't run a router (missing model artifact, missing API key,
network timeout), that router's column shows its error and the other two
still render normally — the score banner is computed only over successful
routers.

## Error states

| HTTP / situation | UI behavior |
|---|---|
| 422 (validation) | "Entrada inválida". Should be prevented by the maxLength on the input. |
| 429 (rate limit) | Friendly "muitas comparações seguidas, aguarde um minuto" notice. |
| 503 (provider) | "O backend está sem credenciais de provedor LLM". |
| 500 (server) | "Erro interno no servidor". Tech details stay on the server. |
| Network failure | "Não foi possível conectar ao backend". |
