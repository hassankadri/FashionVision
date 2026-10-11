
import { defineConfig } from 'vite';

export default defineConfig({
  esbuild: {
    jsx: 'automatic',
    jsxImportSource: 'react',
  },

  server: {
    port: 5173,
    strictPort: true,

    proxy: {
      '/api': 'http://127.0.0.1:8765',
    },
  },
});
