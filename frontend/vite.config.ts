import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/**
 * Dev server configuration.
 *
 * The API is proxied rather than called cross-origin in development. That is not a
 * convenience: it means a phone test needs **one** HTTPS tunnel instead of two, and it
 * removes CORS from the picture entirely, because the browser sees a single origin.
 * Chasing a CORS misconfiguration through a tunnel is exactly the kind of accidental
 * complexity that makes real-device testing not happen.
 *
 * In production the frontend and backend are on different hosts (Vercel and Render), so
 * `VITE_API_BASE_URL` is set there and the proxy is irrelevant.
 */
const BACKEND = process.env.TYRETREAD_BACKEND ?? "http://127.0.0.1:8010";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Bind to all interfaces so a tunnel (or a LAN device) can reach the dev server.
    host: true,
    // Vite blocks requests whose Host header it does not recognise. A tunnel arrives
    // with its own hostname, so those suffixes are allowed explicitly rather than
    // disabling the protection wholesale.
    allowedHosts: [".trycloudflare.com", ".ngrok-free.app", ".loca.lt", "localhost"],
    proxy: {
      "/api": {
        target: BACKEND,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
  build: {
    target: "es2020",
    // Evidence panels arrive as base64 data URIs, so the bundle itself stays small.
    // Keeping the warning threshold low makes accidental bloat visible.
    chunkSizeWarningLimit: 300,
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test-setup.ts"],
    globals: true,
  },
});
