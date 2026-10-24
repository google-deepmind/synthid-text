# Slide 3: relevance and sources

## Slide objective

Connect the research to current industry adoption and transparency requirements. Use two short official-source excerpts or screenshots with a title, source, and date. Avoid a wall of legal text.

Sources below were checked on **2026-10-24**. Recheck availability and wording before the final presentation.

## Two official examples

| Provider | What the source supports | Suggested slide wording |
| --- | --- | --- |
| Anthropic | Its 2026-08-14 announcement describes a SynthID Text variant and connects the rollout to the EU AI Act. | "Anthropic announced SynthID-based text watermarking for Claude." |
| OpenAI | Its current documentation describes textGrain for ChatGPT text in the EU and optional activation for supported API text outputs. Coverage varies. | "OpenAI documents its own text watermarking method, textGrain." |

- [Anthropic: How Claude's text watermark works](https://www.anthropic.com/news/claude-text-watermark). Publication date: 2026-08-14. Preserve the source's rollout qualifications.
- [OpenAI: Provenance signals in OpenAI-generated content](https://help.openai.com/en/articles/8912793-provenance-signals-in-openai-generated-content). This is a maintained official article; label the screenshot with the consultation date instead of inventing a publication date.

Do not say that every GPT/Claude model or every historical response is watermarked. Do not imply that OpenAI's text system uses the same algorithm as the local SynthID reference implementation.

Provider claims about quality or latency are their reported findings; they are not measurements of our GPT-2 setup.

## European context

Article 50(2) addresses machine-readable marking and detectability of synthetic outputs for the systems within its scope. It does not prescribe SynthID as the only implementation.

Suggested wording:

> This project investigates a technical approach related to transparency and synthetic-content marking discussed in the EU AI Act. The prototype is not a legal-compliance certification.

Primary reference: [European Commission AI Act Service Desk, Article 50](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-50).

Keep the scope, exceptions, and legal analysis out of unsupported blanket claims such as "all AI text must have our watermark."

## Our reference implementation

[Google DeepMind SynthID Text repository](https://github.com/google-deepmind/synthid-text) is the codebase being reproduced. It is a research/reference implementation with a static default configuration, not a production deployment of Claude or ChatGPT.

If you open its installation section during the talk, show the origin and instructions without reinstalling anything. The working environment already exists. GPT-2 is a continuation model, not a modern instruction-following assistant.

## Delivery

Allow roughly one minute for the two sources and the research motivation. If time is short, give one sentence per provider and move to the mechanism.

## Spoken transition

"These systems use different implementations, but they share the goal of leaving a machine-detectable signal. Our experiment studies the open SynthID Text reference code locally."

Continue with [slide 4](04_synthid_and_robustness.md).
