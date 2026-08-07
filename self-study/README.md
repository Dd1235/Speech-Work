# Self-study plan

Picking this up whenever there's time. Tick as I go. Roughly two weeks per group.

---

## 1. The signal

- [ ] **Speech production, acoustic phonetics** — source–filter model, formants, why phones differ spectrally
- [ ] **Time–frequency analysis** — windowing, STFT, spectrograms, filterbanks, mel scale
- [ ] **Cepstral features** — MFCCs, deltas, what truncating to 12 coefficients discards
- [ ] **Linear prediction, F0** — LPC, pitch estimation, pre-emphasis, VAD

> *Build:* MFCC pipeline from scratch, checked against librosa on TIMIT.
> Answers the thing I did empirically last time — why `c0` had to be standardized.

## 2. Acoustic modelling

- [ ] **Classical stack** — VQ, GMM-HMM, forced alignment ← where my per-class Gaussians come from
- [ ] **Neural ASR** — CTC, attention seq2seq, RNN-T, Whisper-style encoder–decoder
- [ ] **Decoding** — beam search, LM fusion
- [ ] **Evaluation properly** — PER/WER, alignment scoring, confidence intervals

> *Build:* fine-tuned phone recognizer on TIMIT, PER with a CI.
> Gives the supervised ceiling my unsupervised work never had to compare against.

## 3. Learned representations

- [ ] **Self-supervised models** — wav2vec 2.0, HuBERT, WavLM; contrastive vs masked prediction
- [ ] **Layer-wise probing** — what lives at which depth, and how probes mislead
- [ ] **Few-shot** — matching nets, prototypical nets, episodic training
- [ ] **Low-resource, multilingual** — cross-lingual transfer, Indic corpora, code-switching
- [ ] **Synthesis** — neural TTS, vocoders, prosody control

> *Build:* layer-wise probe + recompute τᵢⱼ per layer → research direction 2 below.

## 4. Efficiency

- [ ] **Quantization** — PTQ vs QAT, per-channel scales, calibration, mixed precision
- [ ] **Pruning, distillation** — structured vs unstructured, student models
- [ ] **Running for real** — export, runtimes, RTF vs first-audio latency, resident memory

> *Build:* the §2 recognizer at INT8, exported and benchmarked. Every number measured, not estimated.

## 5. Ethics

- [ ] **Voice privacy, cloning** — consent, licensing of voices and checkpoints

---

## Reading

| source | for |
|---|---|
| [Aalto — Introduction to Speech Processing](https://speechprocessingbook.aalto.fi/) | §1, §2, evaluation, SSL, privacy |
| [Stanford CS224S](https://web.stanford.edu/class/cs224s/) | §2, non-English speech |
| [HuggingFace Audio Course](https://huggingface.co/learn/audio-course/) | implementation throughout §2–3 |
| [MIT 6.5940 / efficientml.ai](https://efficientml.ai) | §4 in full |

Papers alongside: wav2vec 2.0, HuBERT, WavLM, Matching Networks, Prototypical Networks, SmoothQuant, AWQ, Deep Compression.

---

## Research directions

**1. What does compression destroy in a speech model?** *(main one)*

WER and file size are too coarse to show what quantization actually breaks. I have a measure of phonetic resolution that works — τᵢⱼ predicts held-out confusions at ρ=−0.849.

- Quantize an SSL encoder: INT8/INT4, per-channel vs per-tensor, layers at mixed precision
- At each setting track *together*: error rate, τᵢⱼ over the inventory, few-shot accuracy
- Hypothesis: fine contrasts (`sh/ch`, `ay/aw`, `s/z`) collapse before WER reacts, and damage localises to specific layers
- Output: per-layer/per-bit-width map of which layers must stay high-precision
- Needs §3 + §4 done first. Null result is also publishable.

**2. Certificate on learned features** *(fallback, lower risk)*

Everything I have is about 12-dim MFCCs. Recompute τᵢⱼ and the few-shot protocol on wav2vec2/HuBERT layer activations — separability should be far better — and turn it into a layer-selection criterion. This is the control experiment the paper explicitly lacks; it would complete it with a positive result.

**3. Lightweight Indic speech on edge**

Per-module quantization sensitivity for a compact TTS: keep duration, pitch and vocoder heads at higher precision, quantize the linear/conv bulk, score on prosody (F0 error, duration error) not file size. Caveat: several strong Indic models are gated or non-commercially licensed — research-only scope, synthetic or licensed reference audio only.

---

## If I want the reading elective next time

Drafted a proposal for it — short and long versions in `resources/` (local only, not in git). Couldn't get hold of the prof before the elective window closed. Worth retrying; he's the only ASR person in the college.
