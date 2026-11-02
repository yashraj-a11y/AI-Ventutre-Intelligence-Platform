/// <reference types="vite/client" />

// Day 19 — typed custom env vars (Vite exposes VITE_*-prefixed vars on import.meta.env).
interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string;
}
interface ImportMeta {
  readonly env: ImportMetaEnv;
}
