# Slide 2: reveal and project status

## Slide content

Keep the positions and text from slide 1. Add explicit **Watermarked** and **Unwatermarked** labels, supported by colour. Do not rely on colour alone.

For the new slide-1 recipe and the recorded library pair 02:

| Audience label | Known generation condition |
| --- | --- |
| Text X | Watermarked |
| Text Y | Unwatermarked |

If you substitute any other pair, read its own answer key. The recorded library pair 01 has the opposite mapping, so its X/Y labels must not be reused blindly.

The reveal is based on the generation configuration. It is not a claim that the current detector has a calibrated decision threshold.

## Show what the project has already delivered

Use one small status line on the slide:

> Local GPT-2 generation | External key file | Saved-text scoring

Explain orally that the implementation runs on the local GPU, uses the cached model offline, and scores text with the compatible configuration. The last recorded suite passed 35 tests on 2026-10-21. Do not describe that as a fresh test run on 2026-10-24.

| Project milestone | Current status |
| --- | --- |
| Reproduce the repository | Existing environment and GPU validated; saved corrections tested. A clean installation from scratch was not performed. |
| Run in prompt mode | Implemented and exercised, including saved TXT verification and external keys. |
| Evaluate quality, detection, and cost | Protocol planned; only exploratory examples recorded so far. |
| Build the API and visual interface | Future work. |

## Evidence for questions

- [Recorded full test log](evidence/validation_2026-10-21.txt): 35 passed, 6 warnings in 91.09 seconds.
- [Recorded environment](evidence/environment_record.json): Python 3.10.21, PyTorch 2.10.0+cu128, Transformers 4.43.3, RTX 5060 Laptop GPU.
- [Recorded pairs and transformations](evidence/rehearsal_results.json): the actual generation labels, lengths, and scores.

The six warnings concern an existing PyTorch batching-performance limitation for `aten::take`. They were not test failures. The notebooks were not run end to end; the corrected aggregation cells were checked separately.

Keep the environment details and test log in speaker notes or available for questions; they are not another installation section in the talk.

## Interpretation

The mark is a statistical pattern in token choices, not a hidden string or special character. Reader guesses do not establish detection accuracy, equal quality, or general imperceptibility. Those are questions for the evaluation on slide 6.

## Spoken transition

"The generation settings tell us which output received the watermark. Now we will explain why this topic matters and how a verifier can search for its statistical signal."

Continue with [slide 3](03_relevance_and_sources.md).
