import { describe, expect, it } from 'vitest';
import { staleModelUrls } from '../src/lib/cache.js';

const repo = 'SiobhanGoh/malay-english-sentiment-onnx';
const now = '7452008881da33359fd4600be46efbee0381531e';
const old = '80f942af84c6b05e7e5c7b56a4e7d0bda6d85595';

describe('staleModelUrls', () => {
  it('keeps the current revision and unrelated files, and lists older revisions', () => {
    const urls = [
      `https://huggingface.co/${repo}/resolve/${now}/onnx/model_fp16.onnx`,
      `https://huggingface.co/${repo}/resolve/${old}/onnx/model_fp16.onnx`,
      `https://huggingface.co/${repo}/resolve/${old}/config.json`,
      'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.0/dist/ort-wasm-simd-threaded.asyncify.wasm',
      `https://huggingface.co/someone/else/resolve/${old}/config.json`,
    ];
    expect(staleModelUrls(urls, repo, now)).toEqual([urls[1], urls[2]]);
  });
});
