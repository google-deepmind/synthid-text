# Slide 7: next-token animation

## Purpose

Finish with a short visual explanation of the mechanism. This is the requested storyboard, not a claim that a new animation has already been produced. The original interface animation has not been supplied in this workspace.

Keep a single sentence visible:

```text
The princess was ____
```

Use the correctly spelled candidates `gorgeous`, `beautiful`, `elegant`, and `fancy`. Words are a teaching simplification; the actual model operates on tokens, and a displayed word is not guaranteed to be a single GPT-2 token.

## Suggested sequence: approximately 25 seconds

| Time | What appears | What changes or disappears |
| --- | --- | --- |
| 0-3 s | The incomplete sentence | No code, key values, or probability bars yet. |
| 3-8 s | Four candidates with base-probability bars | Keep the sentence visible. |
| 8-12 s | A small context/key indicator | Fade unrelated labels; show that context and configuration influence the update. |
| 12-17 s | Adjusted candidate probabilities | Animate the bar change rather than adding a second dense table. |
| 17-21 s | A sampling indicator selects `beautiful` | Make clear this is one possible draw, not an automatic maximum-probability rule. |
| 21-25 s | `The princess was beautiful` | Fade the candidates; briefly indicate the context is updated for the next step. |

Only elements relevant to the current step should appear. Use a consistent colour for the selected token and avoid floating labels that remain after their step ends.

## Illustrative numbers

These values are a simplified teaching example, **not measurements from GPT-2, ChatGPT, or Claude**:

| Displayed candidate | Before update | After illustrative update |
| --- | ---: | ---: |
| gorgeous | 45% | 22.5% |
| beautiful | 40% | 60% |
| elegant | 10% | 15% |
| fancy | 5% | 2.5% |

Each column sums to 100%. In this example `beautiful` overtakes `gorgeous`, which was initially first. It is not enough to say that `beautiful` overtakes `elegant`, which was already below it.

The sample selects `beautiful` in the animation. Another draw can select a different token. The full reference implementation applies several watermarking layers; these four bars illustrate a local effect and are not a complete simulation of that implementation.

## Narration

"The model has several plausible continuations. Context and the watermark configuration adjust how those candidates are sampled. In this illustrative draw, it selects 'beautiful'. That token is appended to the text, and the model proceeds to the next step. A compatible verifier later examines statistical evidence over a longer sequence."

The key does not get appended to the sentence. There is no hidden Unicode symbol, tracking identifier, or guaranteed visibly unusual word in this explanation.

## Closing statement

> The local generation and scoring mechanism already runs. Next we will measure quality, detectability, and cost, then connect the validated functions to the final API and interface.

If you show the separate original interface animation, label it as a future product concept. The old terminal-replay MP4 is a different asset and does not implement this storyboard.

Return to the [presentation entry point](../README_ETAPA2.md).
