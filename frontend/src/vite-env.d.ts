/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Backend base URL. Fallback in `api.ts` is `http://localhost:8000`. */
  readonly VITE_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
