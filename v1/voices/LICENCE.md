# Reference voices

Six clips of recorded human speech, the voices of the cloning speech engine (`tropalm:qwen`). A
cloning engine has no voice of its own — handed nothing it picks a different one every sentence —
so a companion's voice IS one of these files.

A companion nobody gave a voice gets one of the two listed in `QwenTtsEngine.DefaultVoices`
(one female, one male, both LibriVox readings; owner ruling Q-tts1, 2026-09-24). The others stay
so a companion that already names one keeps it, and so a person can pick them.

Every clip was loaded at 24 kHz mono, trimmed at 35 dB below peak, padded with 80 ms of silence
at each end so a clone does not begin clipped, peak-normalised to 0.92, and written as 16-bit PCM.

## LibriVox (public domain)

Volunteer audiobook readings of public-domain books. LibriVox dedicates every recording to the
public domain; the archive.org items below carry the public domain dedication or mark. Nothing is
owed downstream. The readers are named here because the clip is their voice, not because the
licence asks for it. Each clip was cut out of the archive's 64 kbps mp3 of the chapter.

| file | work, place | reader | source | what is said | median pitch |
|---|---|---|---|---|---|
| `librivox-janeeyre-klett-f.wav` | *Jane Eyre* (version 3), chapter 1, at 6:25 | Elizabeth Klett (female) | [jane_eyre_ver03_0809_librivox](https://archive.org/details/jane_eyre_ver03_0809_librivox) | With Bewick on my knee, I was then happy: happy at least in my way. | 184 Hz |
| `librivox-jekyll-barnes-m.wav` | *The Strange Case of Dr Jekyll and Mr Hyde*, `jekyllandhyde_01-03_stevenson_64kb.mp3`, at about 1:57 | David Barnes (male) | [jekyll_and_hyde_librivox](https://archive.org/details/jekyll_and_hyde_librivox) | "I incline to Cain's heresy," he used to say quaintly. "I let my brother go to the devil in his own way." | 95 Hz |
| `librivox-scarlet-dixon-m.wav` | *A Study in Scarlet*, `studyinscarlet_01_doyle_64kb.mp3`, at about 1:38 | Robert Dixon (male) | [a_study_in_scarlet_1406_librivox](https://archive.org/details/a_study_in_scarlet_1406_librivox) | ... my orderly, who threw me across a pack-horse, and succeeded in bringing me safely to the British lines. | 111 Hz |

**Pitch decides a male clip, not the label.** The engine copies the register it is shown and
drifts up from it, most on short lines. A light male reader (Mark F. Smith's *Great Gatsby*,
174 Hz) cloned into a voice the owner heard as a woman's. So a male clip is measured before it is
kept: Barnes is the deep default, Dixon the less deep alternative, and both clone under 140 Hz
(`docs/plans/mid/voice_communication.md`, the TTS section).

## Mozilla Common Voice 17.0 (CC0)

Released under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) (public domain
dedication): no attribution is required and nothing is owed downstream. Reached through the
ungated parquet mirror [`fixie-ai/common_voice_17_0`](https://huggingface.co/datasets/fixie-ai/common_voice_17_0).
These were the defaults until 2026-09-24; the owner heard their clones as flat.

| file | clip | speaker | what is said | median pitch |
|---|---|---|---|---|
| `commonvoice-ja-f.wav` | ja validation, 6 up-votes | female, twenties | この試合に残された時間はわずかしかない | 235 Hz |
| `commonvoice-ja-m.wav` | ja validation, 2 up-votes | male, twenties | そんなに走って、ホームランの一本も出ないなら、勿体ないな。 | 146 Hz |
| `commonvoice-en-m.wav` | en validation, 2 up-votes | male | Crist gained increased success after the video went viral. | 139 Hz |

The reference's language does not bind the output's: the model transfers a speaker across
languages, so a clip recorded in English speaks Japanese in the same register
(`docs/records/investigations/2026-09-21-tts-survey.md`, section 4).

**Not shipped, and why.** The survey's most expressive Japanese reference is a JVNV corpus clip
under CC BY-SA 4.0. ShareAlike reaches an adaptation of the clip, and synthesized speech cloned
from it is arguable as one — so it stays out of the build. It remains the best-sounding
reference measured, which is the cost of this decision rather than an argument against it.
