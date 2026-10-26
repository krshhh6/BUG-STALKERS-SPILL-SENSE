import react from '@vitejs/plugin-react'
import { defineConfig, type Plugin } from 'vite'
import cesium from 'vite-plugin-cesium'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

function onnxWasmPlugin(): Plugin {
  return {
    name: 'serve-onnx-wasm',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (req.url && req.url.startsWith('/onnx-dist/')) {
          const fileName = req.url.replace('/onnx-dist/', '').split('?')[0];
          const filePath = path.resolve(__dirname, 'node_modules/onnxruntime-web/dist', fileName);
          if (fs.existsSync(filePath)) {
            if (fileName.endsWith('.wasm')) {
              res.setHeader('Content-Type', 'application/wasm');
            } else if (fileName.endsWith('.mjs') || fileName.endsWith('.js')) {
              res.setHeader('Content-Type', 'application/javascript');
            }
            return fs.createReadStream(filePath).pipe(res);
          }
        }
        next();
      });
    },
    closeBundle() {
      const srcDir = path.resolve(__dirname, 'node_modules/onnxruntime-web/dist');
      const destDir = path.resolve(__dirname, 'dist/onnx-dist');
      if (fs.existsSync(srcDir)) {
        if (!fs.existsSync(destDir)) {
          fs.mkdirSync(destDir, { recursive: true });
        }
        const files = fs.readdirSync(srcDir);
        for (const file of files) {
          if (file.endsWith('.wasm') || file.endsWith('.mjs') || (file.endsWith('.js') && !file.endsWith('.map'))) {
            const destFile = path.join(destDir, file);
            if (!fs.existsSync(destFile)) {
              fs.copyFileSync(path.join(srcDir, file), destFile);
            }
          }
        }
        console.log('[onnxWasmPlugin] Copied onnxruntime-web dist files to dist/onnx-dist');
      }
    },
  };
}

export default defineConfig({
  // cesium() must come before react() so it can intercept the Cesium asset
  // copy step before Vite resolves modules.
  plugins: [(cesium as unknown as () => Plugin)(), react(), onnxWasmPlugin()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  server: {
    hmr: {
      overlay: false,
    },
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/socket.io': {
        target: 'http://127.0.0.1:8000',
        ws: true,
      },
    },
  },
  preview: {},
})
