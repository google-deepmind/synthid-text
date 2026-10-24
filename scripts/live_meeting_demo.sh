#!/usr/bin/env bash
# Run the live demonstration with explicit prompt, key and text files.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="local-results/meeting-rehearsal-2026-10-21/live-$(date -u +%Y%m%dT%H%M%SZ)"

printf '\nPrompt file:\n'
cat prompts/meeting_library.txt
printf '\n\nPublic demonstration key file:\n'
cat keys/demo_key.txt
printf '\nGenerating both conditions...\n'
"$PYTHON_BIN" -B scripts/prompt_demo.py generate \
  --prompt "$(cat prompts/meeting_library.txt)" \
  --key-file keys/demo_key.txt \
  --max-new-tokens 256 --temperature 0.7 --top-k 40 --seed 42 \
  --output-dir "$RUN_DIR/generation"

# Give the two newly generated continuations stable names for the next commands.
"$PYTHON_BIN" -B - "$RUN_DIR" <<'PY'
import json
from pathlib import Path
import sys
run_dir = Path(sys.argv[1])
report_path = max((run_dir / 'generation').glob('generate_*.json'))
report = json.loads(report_path.read_text(encoding='utf-8'))
for sample in report['samples']:
    label = 'Y' if sample['watermarked'] else 'X'
    (run_dir / f'{label}.txt').write_text(sample['generated_text'], encoding='utf-8')
print('Saved the NEW continuations as X.txt (unwatermarked) and Y.txt (watermarked).')
PY

printf '\nChecking the newly generated X.txt...\n'
"$PYTHON_BIN" -B scripts/prompt_demo.py detect \
  --text-file "$RUN_DIR/X.txt" --key-file keys/demo_key.txt \
  --output-dir "$RUN_DIR/detect_X"
printf '\nChecking the newly generated Y.txt...\n'
"$PYTHON_BIN" -B scripts/prompt_demo.py detect \
  --text-file "$RUN_DIR/Y.txt" --key-file keys/demo_key.txt \
  --output-dir "$RUN_DIR/detect_Y"
printf '\nLive demonstration complete. Files: %s\n' "$RUN_DIR"
