import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// vite.config.js
// Proxy target priority:
//   1. VITE_PROXY_TARGET env var (set in docker-compose for container-to-container routing)
//   2. http://localhost:8080 for local dev (gateway mapped to host:8080 in docker-compose)
//
// VITE_API_BASE should always be '' (empty) so the browser uses relative URLs
// that Vite's dev server proxies. Set VITE_API_BASE only when pointing the
// browser directly at a remote gateway (e.g. production deployment).

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const proxyTarget = env.VITE_PROXY_TARGET || 'http://localhost:8080'

  return {
    plugins: [react()],
    server: {
      host: '0.0.0.0',   // bind to all interfaces — required inside Docker
      port: 5173,
      proxy: {
        // Proxy all /api/* requests to the gateway
        '/api': {
          target: proxyTarget,
          changeOrigin: true,
          // SSE streams need these headers to flow through without buffering
          configure: (proxy) => {
            proxy.on('proxyReq', (proxyReq) => {
              proxyReq.setHeader('Accept', 'text/event-stream')
            })
          },
        },
        // Health + metrics endpoints
        '/health': {
          target: proxyTarget,
          changeOrigin: true,
        },
        '/metrics': {
          target: proxyTarget,
          changeOrigin: true,
        },
      },
    },
  }
})
