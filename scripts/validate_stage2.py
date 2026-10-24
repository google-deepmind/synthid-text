"""Run local stage 2 scenarios and save evidence without new dependencies."""
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

root = Path(__file__).resolve().parents[1]
out = root / 'local-results/stage2-validation'
out.mkdir(parents=True, exist_ok=True)
script = root / 'scripts/prompt_demo.py'
checks = []

def run(name, arguments, stdin=None, expected=0):
    destination = out/name
    command = [sys.executable, '-B', str(script), *arguments, '--output-dir', str(destination)]
    p = subprocess.run(command, input=stdin, text=True, capture_output=True, cwd=root)
    (out/f'{name}.stdout.txt').write_text(p.stdout, encoding='utf-8')
    (out/f'{name}.stderr.txt').write_text(p.stderr, encoding='utf-8')
    result = {'name': name, 'arguments': arguments, 'stdin': stdin,
              'exit_code': p.returncode, 'expected_exit_code': expected, 'passed': p.returncode == expected}
    checks.append(result)
    print(name, 'OK' if result['passed'] else 'FAIL', 'exit', p.returncode, flush=True)
    if p.returncode != expected:
        print(p.stdout, p.stderr, flush=True)
        raise RuntimeError(name)
    files = sorted(destination.glob('*.json')) if destination.exists() else []
    return json.loads(files[-1].read_text()) if files else None

prompt = 'In a small village near the mountains, a young artist found'
common = ['generate', '--prompt', prompt, '--max-new-tokens', '128', '--seed', '42']
main = run('main_demo', common)
repeat = run('repeat', common)
identical = all(a['generated_token_ids'] == b['generated_token_ids'] for a,b in zip(main['samples'], repeat['samples']))
checks.append({'name':'same_seed_token_reproducibility', 'passed':identical})
assert identical
run('interactive', ['generate','--max-new-tokens','32','--seed','7'], 'The old library had one secret room\n')
text_file = sorted((out/'main_demo').glob('*_watermarked.txt'))[-1]
detection = run('detect_file', ['detect','--text-file',str(text_file)])
wm = next(s for s in main['samples'] if s['watermarked'])
same = abs(detection['detection']['weighted_mean_score']-wm['detection_retokenized_text']['weighted_mean_score']) < 1e-7
checks.append({'name':'file_detection_matches_generated_text_score','passed':same})
assert same
short = run('short_text', ['detect','--text','Hi'])
assert short['detection']['decision']=='insufficient_text'
run('invalid_top_k', ['generate','--prompt','Hello','--top-k','1'], expected=2)
run('empty_prompt', ['generate','--prompt','   '], expected=2)
run('context_limit_exceeded', ['generate','--prompt','Hello '*1024,'--max-new-tokens','128'], expected=1)
summary = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'checks':checks,
           'all_passed': all(c['passed'] for c in checks)}
(out/'validation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('ALL PASSED', summary['all_passed'], flush=True)
