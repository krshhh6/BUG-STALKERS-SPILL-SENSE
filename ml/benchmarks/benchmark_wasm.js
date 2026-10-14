/**
 * SpillSense (SIH-26143) — ONNX Runtime Web (WASM) Latency Benchmarking
 * Runs in Node using onnxruntime-web with the WASM execution provider.
 */

const path = require('path');
const fs = require('fs');

async function benchmarkWasm() {
    const ort = require('../../frontend/node_modules/onnxruntime-web');

    const classifierPath = path.resolve(__dirname, '../../frontend/public/models/oil_classifier.onnx');
    const segmenterPath = path.resolve(__dirname, '../../frontend/public/models/oil_segmenter.onnx');

    const results = {};

    // 1. Classifier WASM (1, 2, 400, 400)
    if (fs.existsSync(classifierPath)) {
        const t0_init = performance.now();
        const clsSession = await ort.InferenceSession.create(classifierPath, { executionProviders: ['wasm'] });
        const initTime = performance.now() - t0_init;

        const clsTensor = new ort.Tensor('float32', new Float32Array(1 * 2 * 400 * 400), [1, 2, 400, 400]);

        // Warm-up
        for (let i = 0; i < 5; i++) {
            await clsSession.run({ input: clsTensor });
        }

        const clsLatencies = [];
        const iterations = 30;
        for (let i = 0; i < iterations; i++) {
            const t0 = performance.now();
            await clsSession.run({ input: clsTensor });
            clsLatencies.push(performance.now() - t0);
        }

        results['classifier_wasm'] = {
            model: 'oil_classifier.onnx',
            input_shape: [1, 2, 400, 400],
            init_time_ms: initTime,
            iterations: iterations,
            latencies_ms: clsLatencies
        };
    } else {
        results['classifier_wasm'] = { error: 'Model file not found' };
    }

    // 2. Segmenter WASM (1, 2, 512, 512)
    if (fs.existsSync(segmenterPath)) {
        const t0_init = performance.now();
        const segSession = await ort.InferenceSession.create(segmenterPath, { executionProviders: ['wasm'] });
        const initTime = performance.now() - t0_init;

        const segTensor = new ort.Tensor('float32', new Float32Array(1 * 2 * 512 * 512), [1, 2, 512, 512]);

        // Warm-up
        for (let i = 0; i < 3; i++) {
            await segSession.run({ input: segTensor });
        }

        const segLatencies = [];
        const iterations = 20;
        for (let i = 0; i < iterations; i++) {
            const t0 = performance.now();
            await segSession.run({ input: segTensor });
            segLatencies.push(performance.now() - t0);
        }

        results['segmenter_wasm'] = {
            model: 'oil_segmenter.onnx',
            input_shape: [1, 2, 512, 512],
            init_time_ms: initTime,
            iterations: iterations,
            latencies_ms: segLatencies
        };
    } else {
        results['segmenter_wasm'] = { error: 'Model file not found' };
    }

    console.log(JSON.stringify(results, null, 2));
}

benchmarkWasm().catch(err => {
    console.error(JSON.stringify({ error: err.message }));
    process.exit(1);
});
