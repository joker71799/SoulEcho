import { defineConfig } from 'vite';

// 轻量化前端：仅做静态页面 + 开发代理
// 后端 SoulEcho API 监听 127.0.0.1:9000（见 backend/app/main.py）
// 通过 Vite 代理把 /api、/health 转发到后端，浏览器同源访问，规避 CORS
export default defineConfig({
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:9000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://127.0.0.1:9000',
        changeOrigin: true,
      },
    },
  },
});
