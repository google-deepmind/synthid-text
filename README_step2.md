# SynthID Text: intermediate presentation

This is the current documentation for the **seven-slide presentation plan**, revised on 2026-10-24. All maintained documentation, code examples, and comments are in English. The existing filename is retained so previous references still lead to this entry point.

The project investigates a specific statistical text watermark using the Google DeepMind reference implementation and GPT-2. It is not a general AI-text detector.

## Read in presentation order

| Slide | Purpose | Preparation and speaker notes |
| --- | --- | --- |
| 1 | Show one prompt and two continuations; ask the audience to guess | [Audience challenge](docs/01_audience_challenge.md) |
| 2 | Reveal the generation conditions and show the project's current status | [Reveal and project status](docs/02_reveal_and_project_status.md) |
| 3 | Explain relevance using current OpenAI and Anthropic sources | [Relevance and sources](docs/03_relevance_and_sources.md) |
| 4 | Explain the watermark concept, course connections, and robustness limits | [SynthID and robustness](docs/04_synthid_and_robustness.md) |
| 5 | Show the actual generation/detection process and the external key file | [Generation and detection](docs/05_generation_and_detection.md) |
| 6 | Present the quality, meaning, detectability, and resource-cost study | [Evaluation plan](docs/06_evaluation_plan.md) |
| 7 | Finish with a short next-token animation | [Animation storyboard](docs/07_animation_storyboard.md) |

The theoretical slides can be shortened orally. Keep the distinction between completed work, illustrative examples, and future evaluation.

## What currently works

- Local GPT-2 generation with and without SynthID watermarking.
- Offline operation using the existing Hugging Face cache.
- Generation and detection with a configuration loaded from `keys/demo_key.txt`.
- Scoring a saved UTF-8 text file using the official Mean and Weighted Mean functions.
- Saved text, token IDs, configuration, and exploratory time/memory records.
- A tested live-demo script: `scripts/live_meeting_demo.sh`.

The last recorded full suite, on 2026-10-21, passed **35 tests with 6 performance warnings**. Documentation was reorganized on 2026-10-24; this is not a claim that the suite was rerun today. See [recorded validation](docs/evidence/validation_2026-10-21.txt).

The notebooks have not been executed end to end. Their corrected aggregation cells were tested separately. The quality study, calibrated detection thresholds, API, and final interface remain future work.

## Open the existing local environment

Run these commands in Ubuntu-24.04. They enter the repository and activate the existing environment; they do not install packages:

```bash
cd ~/contentProtection/synthid-text
source /home/joao/miniconda3/etc/profile.d/conda.sh
conda activate synthid-text
```

For slide 1, follow [the dedicated generation recipe](docs/01_audience_challenge.md). To rehearse the existing library example after the audience vote, run:

```bash
bash scripts/live_meeting_demo.sh
```

That script uses the previously recorded **library prompt**, not the garden prompt proposed for a new slide-1 pair. It prints the conditions, generates new files, and scores those new files. The exact steps and the role of `--key-file` are explained in [slide 5](docs/05_generation_and_detection.md).

## Evidence and historical material

`docs/evidence/` contains only the environment record, test log, and rehearsal results needed to support this presentation. Generated text, full logs, and old assets remain under `local-results/` and are excluded from Git by the existing ignore rule.

The earlier presentation rehearsal is at:

```text
local-results/meeting-rehearsal-2026-10-21/
```

Its six-page PDF and 79-second replay are **historical backup assets**, not an updated seven-slide deck. The replay contains real command output but is not a screen recording or a recording of the presenter. Earlier output ZIPs are historical snapshots and are not rewritten by this documentation update.

The previous README and complete `docs/` tree were archived outside the active documentation in `local-results/documentation-archive-.../` before replacement. This includes superseded setup narratives, exploratory prompt notes, Git-push instructions, and old manifests.

## Presentation boundaries

- The key influences statistical choices between tokens; it is not written inside the output.
- Scores are not authorship probabilities. No classification threshold or false-positive rate has been calibrated.
- The slide-2 reveal uses the known generation condition, not a newly validated detector decision.
- A small edit or a paraphrase can affect the score; one example does not establish general robustness.
- Current industry announcements motivate the research but do not establish legal compliance of this prototype.
- The final animation explains the mechanism. The production interface and API have not been built.
