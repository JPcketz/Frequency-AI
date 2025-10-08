Frequency AI — Next-Gen Controllable Music Generation



Mission: Create a serious step up in controllable, high-fidelity music generation. Hierarchical musical intelligence, strict constraints (key/mode/tempo), expressive performance, and studio-grade audio/stems/MIDI.



Core Ideas



Hierarchical generation pipeline



Macro-Structure (Form): AABA, Verse/Chorus/Bridge with section lengths \& markers.



Harmony: Key/mode-aware chord progressions with voice-leading.



Melody \& Rhythm: Thematic lines, groove-aware phrasing, micro-timing.



Instrumentation \& Timbre: Up to 4 user-chosen instruments with expressive performance (velocity, micro-timing, CC).



Controls (“Better Music”):



Groove: Import a reference MIDI loop to extract humanized feel; impose with adjustable quantization/humanize.



Tonal Lock: key + mode (e.g., Lydian, Dorian) + tempo; hard constraints at every stage.



Harmonic Anchor: user 4-bar progression (e.g., Am-G-C-F) that the model must honor.



Instrument Set (≤4): e.g., “vintage analog synth bass”, “electric guitar big reverb”, “jazz kit”, “lush pad”.



Structure/Markers: total length (≤ 5 min) + timed events (e.g., 30s: introduce motif, 45s: filter sweep fade).



Outputs:



Audio: ≥ 44.1 kHz, 16-bit, stereo.



Stems: per-instrument WAVs.



MIDI: full note/velocity/CC for DAW editing.



Sheets/CSV: sections, chords, motifs, groove stats (importable to Google Sheets or notation tools).



Performance target: ≥ 15 s of high-quality audio in < 10 s on a modern consumer GPU (via chunked/streamed decoding).



Architecture (high level)

Inputs (key/mode/tempo, 4-bar anchor, groove MIDI, instruments, markers)

&nbsp;   │

&nbsp;   ├─► Structure Transformer  ──► section timeline (Verse/Chorus/Bridge)

&nbsp;   ├─► Harmony Transformer    ──► chord progression + voiced parts

&nbsp;   ├─► Melody/Rhythm Model    ──► multi-line symbolic score (melody/bass/inner voices/drums)

&nbsp;   └─► Timbre Renderer (Hybrid)

&nbsp;          • Latent Diffusion decoder for realism/space

&nbsp;          • DDSP/differentiable synth for explicit control

&nbsp;          • Optional sample layer (SFZ/SF2) for certain timbres

&nbsp;        ──► Per-stem audio + mixdown (44.1 kHz)





Conditioning encoder: embeds key, mode, tempo, groove template, instrument tokens, and markers into a controllable latent space shared across modules.



Realtime approach: fast symbolic planning; streamed audio decoding in 2–4 s windows with overlap-add and cached conditions.



Training \& Data (rights-respecting)



Start with public/royalty-free/academic sets (examples): Slakh, MUSDB (stems), URMP, MedleyDB, MAESTRO, Lakh MIDI, NSynth (for timbre pretraining).



Curriculum:



Symbolic pretrain (form/harmony/melody) on large MIDI.



Timbre pretrain (DDSP/neural synth) on note-level datasets.



Audio alignment (paired MIDI↔audio stems) for latent diffusion.



Finetune on real stems for realism \& mix robustness.



Custom losses (beyond vanilla recon)



Key/Scale loss: penalize out-of-key notes (key profile distance).



Chord-tone weighting: prioritize chord tones on strong beats; allow passing tones with resolution.



Voice-leading penalty: discourage parallel 5ths/8ves \& big inner-voice leaps.



Rhythm coherence: multi-resolution beat losses; groove-template alignment with bounded micro-timing.



Structure consistency: contrastive motif loss to enforce thematic relationships across markers.



Timbral distinctness: contrastive embedding separation between stems (avoid timbre collapse).



Perceptual audio: multi-res STFT, time-domain L1, loudness regularization, multi-scale adversarial discriminator.



Objective metrics



Tonal Coherence Score (TCS) — in-key % weighted by metrical salience + cadence conformity.



Rhythmic Complexity Index (RCI) — IOI entropy + nPVI + syncopation with groove DTW distance.



Timbral Distinctness (TD) — inter-stem vs intra-stem embedding separation (silhouette).



Structure Repetition Index (SRI) — motif recurrence via self-similarity/MI across sections.



Render Quality (RQ) — multi-band log-spectral distance + LUFS/crest/transient balance.



Planned layout

frequency-ai/

&nbsp; README.md

&nbsp; LICENSE

&nbsp; .gitignore

&nbsp; requirements.txt

&nbsp; src/freqai/

&nbsp;   \_\_init\_\_.py

&nbsp;   cli.py

&nbsp;   config.py

&nbsp;   conditioning/

&nbsp;     \_\_init\_\_.py

&nbsp;     features.py

&nbsp;   models/

&nbsp;     \_\_init\_\_.py

&nbsp;     structure\_transformer.py

&nbsp;     harmony\_transformer.py

&nbsp;     melody\_rhythm\_transformer.py

&nbsp;     timbre\_latent\_diffusion.py

&nbsp;     latent\_vae.py

&nbsp;   synthesis/

&nbsp;     \_\_init\_\_.py

&nbsp;     renderer.py

&nbsp;     instruments/

&nbsp;   data/

&nbsp;     \_\_init\_\_.py

&nbsp;     midi\_utils.py

&nbsp;     audio\_utils.py

&nbsp;     schemas.py

&nbsp;     datasets.py

&nbsp;   training/

&nbsp;     \_\_init\_\_.py

&nbsp;     losses.py

&nbsp;     metrics.py

&nbsp;     train.py

&nbsp;   inference/

&nbsp;     \_\_init\_\_.py

&nbsp;     generate.py

&nbsp;     groove\_imposer.py

&nbsp;   export/

&nbsp;     \_\_init\_\_.py

&nbsp;     stems.py

&nbsp;     midi\_export.py

&nbsp;     mixdown.py



Milestones (tiny steps)



M0 (now): README → .gitignore → requirements.txt → minimal cli.py that accepts your controls and prints a plan.



M1: Symbolic prototype (form→harmony→melody) + MIDI export + groove imposition.



M2: Audio prototype (DDSP/synth) → stems + mixdown (44.1 kHz).



M3: Latent diffusion decoder for realism; timbre contrastive loss.



M4: Performance \& UX: streamed decoding, config files, small GUI/Gradio.



License (tentative)



Apache-2.0 (finalize after dataset review).

