# Slide 1: audience challenge

## Slide objective

Show the **same prompt** and two short continuations, labelled **Text X** and **Text Y**. Ask: "Which text was generated with a watermark?"

Use the same font, size, colour, and layout in both columns. Do not show scores, model-class names, generation labels, or filenames that disclose the conditions. Keep the answer for slide 2.

The audience vote is an introduction, not a controlled assessment of perceptibility or text quality.

## Prepare a new pair

This is a reproducible recipe to run; the garden outputs have not been generated as part of the documentation update. Do not attach old library scores to newly generated garden texts.

First activate the existing environment:

```bash
cd ~/contentProtection/synthid-text
source /home/joao/miniconda3/etc/profile.d/conda.sh
conda activate synthid-text
```

Create a unique output path and generate both conditions. The prompt anchors a scene and an activity without asking GPT-2 to reproduce a known story:

```bash
SLIDE_RUN="local-results/slide_01/$(date -u +%Y%m%dT%H%M%SZ)"
python -B scripts/prompt_demo.py generate \
  --prompt "In a quiet village, the residents decided to turn an empty piece of land into a community garden. On the first morning," \
  --key-file keys/demo_key.txt \
  --mode both \
  --max-new-tokens 128 \
  --temperature 0.7 \
  --top-k 40 \
  --seed 7 \
  --output-dir "$SLIDE_RUN/generation"
```

- `SLIDE_RUN` keeps each attempt in its own folder.
- `--mode both` generates the unwatermarked and watermarked conditions sequentially.
- `--key-file` loads the public demonstration configuration.
- `--max-new-tokens` is an upper limit, not a guaranteed length; EOS can stop generation early.
- Both conditions use the same prompt, seed, temperature, top-k, and checkpoint. Their continuations can still differ.
- Top-p remains 1.0 in this reference demo.

The terminal prints the generation labels, so prepare this material before the audience vote.

## Save neutral audience files and a private answer

Run the following block in the same terminal, while `SLIDE_RUN` still refers to this generation. It copies the recorded text without rewriting it. The mapping is fixed in advance: **X is watermarked and Y is unwatermarked**.

```bash
python -B - "$SLIDE_RUN" <<'PY'
import json
from pathlib import Path
import sys

run_dir = Path(sys.argv[1])
report_path = max((run_dir / "generation").glob("generate_*.json"))
report = json.loads(report_path.read_text(encoding="utf-8"))
audience = run_dir / "audience"
presenter = run_dir / "presenter"
audience.mkdir(exist_ok=True)
presenter.mkdir(exist_ok=True)
(audience / "prompt.txt").write_text(report["prompt"], encoding="utf-8")
answers = {}
for sample in report["samples"]:
    label = "X" if sample["watermarked"] else "Y"
    (audience / f"Text_{label}.txt").write_text(
        sample["generated_text"], encoding="utf-8"
    )
    answers[label] = {
        "watermarked": sample["watermarked"],
        "generated_tokens": sample["generated_tokens"],
        "weighted_mean_score": sample["detection_retokenized_text"]["weighted_mean_score"],
    }
(presenter / "answer_key.json").write_text(
    json.dumps(answers, indent=2) + "\n", encoding="utf-8"
)
print("Audience files:", audience)
print("Private answer:", presenter / "answer_key.json")
PY
```

Copy the contents of `prompt.txt`, `Text_X.txt`, and `Text_Y.txt` onto slide 1. Keep `answer_key.json` and the raw JSON away from the projected view until the reveal.

Read both outputs before presenting. GPT-2 may repeat phrases or invent details. If readability requires another prompt/seed, retain every attempt and explain the selection rule; do not select a pair because its detector scores look favourable. If you display excerpts, label them as excerpts and do not describe full-text scores as excerpt scores.

## Existing fallback pair

If a new pair cannot be prepared, use the recorded **library pair 02**:

```text
local-results/meeting-rehearsal-2026-10-21/audience/prompt.txt
local-results/meeting-rehearsal-2026-10-21/audience/pair_02_X.txt
local-results/meeting-rehearsal-2026-10-21/audience/pair_02_Y.txt
```

That pair also uses X = watermarked and Y = unwatermarked. It used a different prompt from the garden recipe. Its recorded scores and generation conditions are in [rehearsal evidence](evidence/rehearsal_results.json).

Do not use the highly repetitive fairy-tale pattern as the main challenge: the earlier 24-token example produced identical outputs and identical scores in both conditions.

## Spoken transition

"Keep your guess in mind. Next, we will reveal which generation used the watermark and examine what the method actually changes."

Continue with [slide 2](02_reveal_and_project_status.md).
