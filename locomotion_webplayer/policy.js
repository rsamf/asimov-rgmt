// ONNX tracking policy (onnxruntime-web, single-threaded WASM backend so it
// works without cross-origin isolation, e.g. on a static HF Space).
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.wasm.min.mjs';

ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';

export class Policy {
  static async load(url, plant) {
    const session = await ort.InferenceSession.create(url, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    return new Policy(session, plant);
  }

  constructor(session, plant) {
    this.session = session;
    this.H = plant.hist_len; this.C = plant.cmd_len;
    this.O = plant.obs_dim; this.D = plant.cmd_dim; this.A = plant.joint_names.length;
    console.log('[policy] inputs', session.inputNames, 'outputs', session.outputNames);
  }

  // obs Float32Array(O), history Float32Array(H*O), cmd Float32Array(C*D) -> Float32Array(A)
  async run(obs, history, cmd) {
    const feeds = {
      obs: new ort.Tensor('float32', obs, [1, this.O]),
      history: new ort.Tensor('float32', history, [1, this.H, this.O]),
      cmd: new ort.Tensor('float32', cmd, [1, this.C, this.D]),
    };
    const out = await this.session.run(feeds);
    const t = out.action || out[this.session.outputNames[0]];
    return t.data;
  }
}
