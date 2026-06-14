import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 3000,
    host: true,
    // During local dev, proxy API calls to the Flask backend so the browser
    // can call relative paths like /auth/login, /predict/diabetes, etc.
    // (Only used if VITE_API_URL is left empty / set to "/")
    proxy: {
      "/auth": "http://localhost:5000",
      "/predict": "http://localhost:5000",
      // NOTE: /dashboard is intentionally NOT proxied here — that path is
      // handled by React Router. The api.js client calls Flask directly via
      // VITE_API_URL (http://localhost:5000) using absolute URLs.
      "/health": "http://localhost:5000",
      "/model": "http://localhost:5000",
    },
  },
  build: {
    outDir: "dist",
  },
});
