# StemLab — Local MP3 Vocal Remover

A single-page, browser-based vocal remover. Upload an MP3 and preview/download instrumental and vocal tracks as WAV or MP3. Audio is processed locally; no files are uploaded to a server.

## Choose a method

| Method | First download | Notes |
|---|---:|---|
| Quick stereo removal | None | Instant stereo-center cancellation. Doesn't specifically recognize vocals; can remove centered instruments. WAV export fully offline. |
| Compact AI — **default for AI** | ~30 MB | UVR_MDXNET_KARA_2; aims to remove the lead vocal, preserving more backing vocals. |
| Balanced AI | ~67 MB | Kim_Vocal_2; attempts to remove lead and backing vocals. |
| Alternative AI | ~67 MB | UVR-MDX-NET-Voc_FT; alternate vocal model. |
| Advanced Demucs v4 | ~166 MB | High-memory HTDemucs 4-stem ONNX; optional, never downloaded automatically. Requires much more RAM than download size suggests. |

When you select **AI separation**, the Compact (~30 MB) model is selected by default. Choosing the Advanced Demucs option is an explicit opt-in. It is not used as a hidden fallback.

### High-memory warning

Demucs can fail with `Can't create a session`, `std::bad_alloc`, or WebGPU device-loss errors on memory-constrained machines. Its 166 MB model file is *not* its RAM footprint; running it may need several gigabytes. Try Compact or CPU-only mode first, close other tabs, and test with a short song. The CPU-only mode is not guaranteed to avoid memory errors.

## Running locally

- Open `index.html` directly in Chrome for Quick Removal.
- To use AI separation, first use an internet connection to fetch the corresponding ONNX model and inference libraries.
- For the **Demucs** module, open the site through an HTTP server or GitHub Pages; `file://` can block JavaScript module imports.
- MP3 export may fetch the LAME encoder library on first use; WAV export needs no encoder download.

### GitHub Pages

In the repo's **Settings → Pages**, deploy from **main** / **root**. Once enabled, your site will normally be at:

https://tbenitz.github.io/MP3-Vocal-Remover/

The repository is plain static HTML and JavaScript. No backend, build step, Python, login, API key, or paid service is needed.

## Implementation

- `index.html`: responsive site, local decoding, quick filtering, MDX-Net AI, WAV/MP3 encoding, controls, results.
- `demucs.js`: opt-in 166 MB HTDemucs ONNX inference, resampling to 44.1 kHz, chunking, overlap-add, two-stem audio export.

Third-party components loaded on demand:
- [web-audio-separation](https://github.com/incidentist/web-audio-separation), MDX-Net processing, and ONNX Runtime Web.
- [StemSplit HTDemucs ONNX](https://huggingface.co/StemSplitio/htdemucs-onnx) 166 MB fp16-weights model.
- [LAME.js](https://github.com/zhuker/lamejs), optional MP3 encoding.

## Verification

The previous version's Quick Removal path was tested using a short audio file. JavaScript syntax for the Demucs module and page script was validated, but the 166 MB AI model was **not** run end-to-end in the development environment. Browser model loading, inference results, caching and GPU/CPU memory requirements need device testing.
