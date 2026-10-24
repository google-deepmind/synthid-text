# Slide 4: SynthID and robustness

## Slide objective

Explain the concept before showing the code. Keep three ideas visible:

1. A language model assigns probabilities to possible next tokens.
2. SynthID uses context and keys to influence those choices statistically.
3. A verifier with a compatible configuration checks the resulting token sequence.

The key itself is not stored inside the text. The output contains evidence of a selection pattern, not a recoverable secret sentence. A token can be a word, part of a word, punctuation, or whitespace-related unit.

## Course connections

Use only topics that have actually been covered in class:

| Topic | Connection |
| --- | --- |
| Conditional probability | Each next-token distribution depends on the current context. |
| Entropy | Highly constrained text leaves fewer plausible choices for marking. |
| Hash functions and keys | Context, candidate tokens, and keys produce reproducible scoring values. |
| Statistical testing | Detection requires comparing evidence with chance behaviour and controlling false positives. |

Do not describe watermarking as encrypting the message. The reference repository explicitly does not guarantee cryptographic security for its hashing function.

## Robustness: recorded local example

The following values come from the **256-token library pair 01**, not from a newly generated garden pair:

| Text/configuration | Weighted Mean | Token count |
| --- | ---: | ---: |
| Unwatermarked control, compatible key | 0.521429 | 256 |
| Watermarked original, compatible key | 0.584550 | 256 |
| Same marked text, different key | 0.497918 | 256 |
| Same bytes, different display font | 0.584550 | 256 |
| One word replaced | 0.583451 | 256 |
| One Unicode character replaced | 0.582637 | 258 |
| One paraphrase | 0.491896 | 189 |

The word edit replaced the first whole-word occurrence of `library` with `community`. The Unicode edit changed a Latin `a` (U+0061) to Cyrillic `а` (U+0430). That changes the text, even if the glyph looks similar.

The paraphrase was produced once by the assistant in the earlier chat. No additional local paraphrasing model was downloaded or executed. Meaning preservation has not been rated by human participants.

Source: [recorded rehearsal results](evidence/rehearsal_results.json). Original text, transformation manifests, and complete detection JSONs remain in `local-results/meeting-rehearsal-2026-10-21/`.

## What to conclude

> In this example, a small edit changed the score only slightly, while a more extensive rewrite reduced it. More samples and calibrated thresholds are needed to measure robustness.

There is no calibrated classification threshold here. Different lengths also affect how much evidence is available. Do not describe a score as an authorship probability, a false-positive rate, or proof that a watermark has been universally removed.

Changing font, colour, or size preserves the score only if the text and tokens remain the same. Copying text out of a PDF or another tool can introduce content changes; visual similarity alone is not a byte-level check.

General limitations are described by [Google's SynthID guidance](https://ai.google.dev/responsible/docs/safeguards/synthid#limitations) and the [original research paper](https://www.nature.com/articles/s41586-024-08025-4).

## Spoken transition

"We can now follow one generation step and see where the key enters the code, then verify a text saved in a normal file."

Continue with [slide 5](05_generation_and_detection.md).
