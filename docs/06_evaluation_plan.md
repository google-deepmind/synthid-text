# Slide 6: quality, meaning, detectability, and cost

## Slide objective

Present the next study as an open research question:

> How much detectable signal can we introduce while preserving useful content and acceptable computational cost?

The evaluation is planned. The earlier rehearsal demonstrates that the pipeline runs; it does not answer the study's questions.

## Questions and measurements

| Dimension | Research question | Planned measurements |
| --- | --- | --- |
| Quality and meaning | Does marking change relevance, coherence, facts, repetition, or the intended meaning? | Blind human ratings and preference; contradiction/repetition analysis; perplexity under the same base model. |
| Detectability | How often does the compatible detector identify marked samples at a known false-positive rate? | Length-aware calibration, TPR/FPR, ROC/AUC, and uncertainty. |
| Robustness | How much evidence survives edits, truncation, paraphrases, or Unicode changes? | Predeclared transformations and scores/detection rates by actual length. |
| Computational cost | What generation overhead is attributable to marking? | Synchronized generation time, tokens/second, allocated/reserved memory, and sampled GPU use where available. |

## The meaning-drift hypothesis

A changed token becomes part of the context, so subsequent probabilities and wording can diverge. That is normal autoregressive behaviour and is not itself evidence of semantic damage.

Compare repeated unwatermarked generations as well as matched watermarked/unwatermarked pairs. This provides a baseline for normal sampling variability. The same seed controls part of the randomness but does not make the two conditions equivalent after their first different choice.

Textual overlap is descriptive, not a measure of truth or preserved meaning. Human evaluation should consider the prompt, key information, contradictions, and usefulness. An embedding similarity score alone cannot establish factual correctness.

## Minimal experimental protocol

1. Freeze the checkpoint, tokenizer, prompts, sampling settings, precision, and watermark configuration.
2. Predeclare prompts, seeds, length limits, and transformations. Retain all outputs, including early EOS and unsuccessful examples.
3. Split calibration and test groups by prompt/source document so related samples do not leak between partitions.
4. Use generated unwatermarked text and separately sourced human text as distinct negative groups.
5. Calibrate thresholds using only calibration negatives, accounting for text length. Evaluate once on held-out samples and report sample counts and uncertainty.
6. Randomize A/B display order and hide generation labels, scores, and timings from human raters. Allow ties and explanations.
7. Score both conditions' continuations under the same unwatermarked base model for perplexity. Include the prompt as context and aggregate negative log-likelihood by valid token count before exponentiating.
8. Repeat performance runs with warm-up, CUDA synchronization, and alternated condition order. Separate loading, generation, detection, and quality-evaluation time.

Do not claim a very low false-positive rate from a few dozen samples or from observing zero false positives in a small set. A negative result does not establish human authorship.

## Performance controls

Record actual generated length and batch size. Keep device, precision, power conditions, and competing workloads documented. PyTorch peak allocated memory measures tensors, not every byte of process VRAM; reserved memory and device-level monitoring answer different questions.

Longer text and a larger model are possible extensions after compatibility and VRAM are checked. They do not guarantee a larger percentage overhead. No new model download is part of the present documentation update.

## Expected study outputs

A frozen protocol, prompts/seeds, per-sample results, a blind-rating form, detection curves by length, a cost table, and a limitations discussion. The API and final interface will reuse functions validated by this work.

For threshold guidance, see the [reference repository](https://github.com/google-deepmind/synthid-text#installation-and-usage). For comparable perplexity, see [Hugging Face's explanation](https://huggingface.co/docs/transformers/perplexity).

## Spoken transition

"We have a working generation and scoring pipeline. The next step is to quantify its trade-offs. This final animation summarizes the token-selection mechanism we will evaluate."

Continue with [slide 7](07_animation_storyboard.md).
