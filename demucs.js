/**
 * Optional high-memory Demucs v4 ONNX backend.
 * Downloads the 166 MB fp16-weights model on first use. Audio never leaves the browser.
 */
const MODEL_URL = 'https://huggingface.co/StemSplitio/htdemucs-onnx/resolve/main/htdemucs_fp16weights.onnx';
const ORT_WASM_PATH = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/';
const SEGMENT = 343980;
const OVERLAP = Math.floor(SEGMENT * .25);
const HOP = SEGMENT - OVERLAP;

async function at44100(audioBuffer) {
  if (audioBuffer.sampleRate === 44100) return audioBuffer;
  const frameCount = Math.ceil(audioBuffer.length * 44100 / audioBuffer.sampleRate);
  const offline = new OfflineAudioContext(2, frameCount, 44100);
  const source = offline.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(offline.destination);
  source.start(0);
  return await offline.startRendering();
}

/**
 * @param {{audioBuffer:AudioBuffer,backend:'cpu'|'auto',onProgress:(fraction:number,message:string)=>void,aborted:()=>boolean}} opts
 * @returns {Promise<{sampleRate:number,instrument:Float32Array[],vocals:Float32Array[]}>}
 */
export async function separateDemucs({audioBuffer, backend='auto', onProgress=()=>{}, aborted=()=>false}) {
  if (aborted()) throw new DOMException('Cancelled', 'AbortError');
  const ort = await import('onnxruntime-web');
  ort.env.wasm.wasmPaths = ORT_WASM_PATH;
  // Single-threaded CPU works without COOP/COEP on GitHub Pages.
  ort.env.wasm.numThreads = 1;
  onProgress(0, 'Loading 166 MB Demucs model; memory use may be several GB…');
  let session;
  try {
    const providers = backend === 'cpu' || !navigator.gpu ? ['wasm'] : ['webgpu','wasm'];
    session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: providers,
      graphOptimizationLevel: 'all'
    });
    if (aborted()) throw new DOMException('Cancelled', 'AbortError');
    onProgress(.05, 'Resampling audio to 44.1 kHz…');
    const pcm = await at44100(audioBuffer);
    const count = pcm.length;
    const left = pcm.getChannelData(0);
    const right = pcm.getChannelData(pcm.numberOfChannels > 1 ? 1 : 0);
    const instrument = [new Float32Array(count), new Float32Array(count)];
    const vocals = [new Float32Array(count), new Float32Array(count)];
    const weights = new Float32Array(count);
    const total = Math.max(1, Math.ceil(Math.max(0, count - SEGMENT) / HOP) + 1);
    const buf = new Float32Array(2 * SEGMENT);
    for (let chunk = 0; chunk < total; chunk++) {
      if (aborted()) throw new DOMException('Cancelled', 'AbortError');
      const offset = chunk * HOP;
      const length = Math.min(SEGMENT, count - offset);
      buf.fill(0);
      buf.set(left.subarray(offset,offset+length), 0);
      buf.set(right.subarray(offset,offset+length), SEGMENT);
      onProgress(.05 + .85*chunk/total, 'Demucs segment '+(chunk+1)+' of '+total+'…');
      const output = await session.run({mix: new ort.Tensor('float32',buf,[1,2,SEGMENT])});
      if (aborted()) throw new DOMException('Cancelled', 'AbortError');
      const stem = output.stems;
      if (!stem || stem.data.length < 8*SEGMENT) throw new Error('Demucs returned an unexpected tensor shape.');
      const data = stem.data;
      // Source order: drums, bass, other, vocals. Stereo L/R for each source.
      for (let i=0; i<length; i++) {
        let weight=1;
        if (chunk>0 && i<OVERLAP) weight=Math.min(weight,i/OVERLAP);
        if (chunk<total-1 && i>=HOP) weight=Math.min(weight,(SEGMENT-i)/OVERLAP);
        const at = offset + i;
        const r = i+SEGMENT;
        weights[at]+=weight;
        vocals[0][at]+=data[6*SEGMENT+i]*weight;
        vocals[1][at]+=data[6*SEGMENT+r]*weight;
        instrument[0][at]+=(data[i]+data[2*SEGMENT+i]+data[4*SEGMENT+i])*weight;
        instrument[1][at]+=(data[r]+data[2*SEGMENT+r]+data[4*SEGMENT+r])*weight;
      }
      // Yield for the progress indicator and cancel button.
      await new Promise(resolve=>setTimeout(resolve,0));
    }
    for(let i=0;i<count;i++) {
      const inv=1/Math.max(1e-7,weights[i]);
      instrument[0][i]*=inv; instrument[1][i]*=inv;
      vocals[0][i]*=inv; vocals[1][i]*=inv;
    }
    onProgress(1,'Demucs separation complete');
    return {sampleRate:44100,instrument,vocals};
  } finally {
    if(session) try {await session.release();}catch(_){/* best effort */}
  }
}
