import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Day 19 — React SPA over the FastAPI backend. API base comes from VITE_API_BASE_URL (Phase B);
// the backend must allow the dev origin via CORS_ORIGINS=http://localhost:5173 (Day-18 deny-by-default).
export default defineConfig({
  plugins: [react(), tailwindcss()],
})
