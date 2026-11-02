import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// transformers.js loads onnxruntime's WebAssembly from a version-pinned jsDelivr URL.
// Vite still emits a local copy that nothing requests, and at about 27 MB it is over
// Cloudflare Pages' 25 MiB file limit, so drop it from the build.
const dropUnusedWasm = {
  name: 'drop-unused-onnx-wasm',
  generateBundle(_, bundle) {
    for (const name of Object.keys(bundle)) if (name.endsWith('.wasm')) delete bundle[name];
  },
};

export default defineConfig({
  plugins: [react(), dropUnusedWasm],
  worker: { format: 'es', plugins: () => [dropUnusedWasm] },
  build: { target: 'es2022', chunkSizeWarningLimit: 1200 },
  test: { environment: 'jsdom', include: ['tests/**/*.test.{js,jsx}'] },
});
