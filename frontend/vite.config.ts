import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";
import { siteNotes } from "./dev/notesServer.ts";

export default defineConfig({
  plugins: [
    react(),
    siteNotes(
      fileURLToPath(new URL("../notes/site-notes.json", import.meta.url)),
    ),
  ],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": {
        target: process.env.PIPE_API_URL || "http://127.0.0.1:8000",
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
