#!/usr/bin/env python3
"""Local, offline SynthID Text demonstration with GPT-2 (stage 2)."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

# Keep this PyTorch demonstration offline and avoid initializing TensorFlow.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["JAX_PLATFORMS"] = "cpu"
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import transformers
from synthid_text import detector_mean, logits_processing, synthid_mixin


def load_keys(key_file: Path | None) -> list[int]:
    """Read a JSON integer array from a UTF-8 text file, or use the public defaults."""
    if key_file is None:
        return list(synthid_mixin.DEFAULT_WATERMARKING_CONFIG["keys"])
    keys = json.loads(key_file.read_text(encoding="utf-8-sig"))
    if not isinstance(keys, list) or not keys:
        raise ValueError("The key file must contain a non-empty JSON array of integers.")
    if any(type(key) is not int or not -(2**63) <= key < 2**63 for key in keys):
        raise ValueError("Each watermarking key must be a signed 64-bit integer, not a boolean.")
    if len(set(keys)) != len(keys):
        raise ValueError("Watermarking keys must be unique.")
    return keys


def watermark_config(keys: list[int] | None = None) -> dict:
    """Use the generation configuration; CPU detection saves GPU memory."""
    return {
        **synthid_mixin.DEFAULT_WATERMARKING_CONFIG,
        "keys": load_keys(None) if keys is None else keys,
        "device": torch.device("cpu"),
    }


class ConfiguredSynthIDGPT2LMHeadModel(synthid_mixin.SynthIDGPT2LMHeadModel):
    """Use the caller's configuration without changing the upstream global defaults."""

    def _construct_warper_list(self, extra_params):
        return transformers.LogitsProcessorList([
            logits_processing.SynthIDLogitsProcessor(
                **self.watermarking_config, **extra_params
            )
        ])


def score_tokens(token_ids: list[int], eos_token_id: int, keys: list[int] | None = None) -> dict:
    config = watermark_config(keys)
    ngram_len = config["ngram_len"]
    result = {
        "token_count": len(token_ids),
        "valid_ngram_count": 0,
        "mean_score": None,
        "weighted_mean_score": None,
        "decision": "not_calibrated",
        "note": "A score is not a probability. No classification threshold has been calibrated.",
    }
    if len(token_ids) < ngram_len:
        result["decision"] = "insufficient_text"
        return result
    processor = logits_processing.SynthIDLogitsProcessor(**config, top_k=40, temperature=0.7)
    ids = torch.tensor([token_ids], dtype=torch.long)
    eos_mask = processor.compute_eos_token_mask(ids, eos_token_id)[:, ngram_len - 1:]
    repetition_mask = processor.compute_context_repetition_mask(ids)
    mask = eos_mask & repetition_mask
    result["valid_ngram_count"] = int(mask.sum().item())
    if result["valid_ngram_count"] == 0:
        result["decision"] = "insufficient_text"
        return result
    g_values = processor.compute_g_values(ids).numpy().astype(np.float32)
    mask_array = mask.numpy()
    # Use the official functions, with default weights decreasing from 10 to 1.
    result["mean_score"] = float(detector_mean.mean_score(g_values, mask_array)[0])
    result["weighted_mean_score"] = float(
        detector_mean.weighted_mean_score(g_values, mask_array)[0]
    )
    if not all(math.isfinite(result[k]) for k in ("mean_score", "weighted_mean_score")):
        raise ValueError("The detector returned a non-finite score.")
    return result


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def generate_one(tokenizer, prompt: str, watermarked: bool, args) -> dict:
    device = synthid_mixin.DEFAULT_WATERMARKING_CONFIG["device"]
    model_cls = ConfiguredSynthIDGPT2LMHeadModel if watermarked else transformers.GPT2LMHeadModel
    print("\nLoading " + ("watermarked GPT-2..." if watermarked else "unwatermarked GPT-2..."), flush=True)
    started = time.perf_counter()
    model = model_cls.from_pretrained("gpt2", local_files_only=True, torch_dtype=torch.float32)
    if watermarked:
        model.watermarking_config = {**watermark_config(args.watermark_keys), "device": device}
    model.to(device).eval()
    synchronize(device)
    load_seconds = time.perf_counter() - started
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    prompt_length = inputs["input_ids"].shape[1]
    context_limit = model.config.n_positions
    if prompt_length + args.max_new_tokens > context_limit:
        raise ValueError(
            f"Prompt ({prompt_length}) + requested output ({args.max_new_tokens}) exceeds "
            f"GPT-2's {context_limit}-token context. Shorten the prompt or reduce --max-new-tokens."
        )
    settings = dict(
        do_sample=True, temperature=args.temperature, top_k=args.top_k,
        top_p=1.0, pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id, use_cache=True,
    )
    with torch.inference_mode():
        # Warm up separately; exclude this step from the measured generation.
        torch.manual_seed(args.seed)
        warmup = model.generate(**inputs, max_new_tokens=min(8, args.max_new_tokens), **settings)
        del warmup
        synchronize(device)
        if device.type == "cuda":
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        else:
            baseline = None
        torch.manual_seed(args.seed)
        synchronize(device)
        started = time.perf_counter()
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, **settings)
        synchronize(device)
        generation_seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
    new_ids = output[0, prompt_length:].cpu().tolist()
    result = {
        "watermarked": watermarked,
        "model_class": model_cls.__name__,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "prompt_tokens": prompt_length,
        "generated_token_ids": new_ids,
        "generated_tokens": len(new_ids),
        "generated_text": tokenizer.decode(new_ids, skip_special_tokens=True),
        "metrics": {
            "model_load_seconds": load_seconds,
            "generation_seconds": generation_seconds,
            "tokens_per_second": len(new_ids) / generation_seconds,
            "cuda_baseline_allocated_mib": baseline / 2**20 if baseline is not None else None,
            "cuda_peak_allocated_mib": peak / 2**20 if peak is not None else None,
            "cuda_peak_increment_mib": (peak - baseline) / 2**20 if peak is not None else None,
            "cuda_peak_reserved_mib": reserved / 2**20 if reserved is not None else None,
        },
    }
    del model, inputs, output
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    result["detection_original_tokens"] = score_tokens(new_ids, tokenizer.eos_token_id, args.watermark_keys)
    # Also check the decoded text that can be copied and pasted into the demo.
    text_ids = tokenizer.encode(result["generated_text"], add_special_tokens=False)
    result["detection_retokenized_text"] = score_tokens(text_ids, tokenizer.eos_token_id, args.watermark_keys)
    non_eos = new_ids[:new_ids.index(tokenizer.eos_token_id)] if tokenizer.eos_token_id in new_ids else new_ids
    result["text_roundtrip_matches_non_eos_tokens"] = text_ids == non_eos
    return result


def environment_info() -> dict:
    packages = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "jax", "jaxlib")}
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    changes = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True)
    return {
        "python": platform.python_version(), "packages": packages,
        "cuda_build": torch.version.cuda,
        "device": str(synthid_mixin.DEFAULT_WATERMARKING_CONFIG["device"]),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_mib": torch.cuda.get_device_properties(0).total_memory / 2**20 if torch.cuda.is_available() else None,
        "git_head": git.stdout.strip() if git.returncode == 0 else None,
        "git_status": changes.stdout.strip() if changes.returncode == 0 else None,
        "synthid_source": str(Path(logits_processing.__file__).resolve()),
    }


def print_score(detection: dict) -> None:
    score = detection["weighted_mean_score"]
    value = f"{score:.6f}" if score is not None else "unavailable (insufficient text)"
    print(f"Weighted Mean: {value}; valid n-grams: {detection['valid_ngram_count']}")
    print("No automatic classification: the threshold has not been calibrated.")


def write_report(report: dict, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / f"{report['operation']}_{stamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if report["operation"] == "generate":
        for sample in report["samples"]:
            label = "watermarked" if sample["watermarked"] else "unwatermarked"
            path.with_name(f"{path.stem}_{label}.txt").write_text(sample["generated_text"], encoding="utf-8")
    print(f"\nResult saved to: {path.resolve()}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="Generate watermarked and/or unwatermarked continuations.")
    gen.add_argument("--prompt", help="If omitted, ask for a prompt in the terminal.")
    gen.add_argument("--mode", choices=("both", "watermarked", "unwatermarked"), default="both")
    gen.add_argument("--max-new-tokens", type=int, default=128)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--temperature", type=float, default=0.7)
    gen.add_argument("--top-k", type=int, default=40)
    det = sub.add_parser("detect", help="Score text without loading the GPT-2 model weights.")
    text = det.add_mutually_exclusive_group()
    text.add_argument("--text")
    text.add_argument("--text-file", type=Path, help="UTF-8 file containing only the text to evaluate.")
    for command in (gen, det):
        command.add_argument("--output-dir", type=Path, default=ROOT / "local-results" / "stage2")
        command.add_argument("--key-file", type=Path, help="UTF-8 .txt file containing a JSON array of watermarking integers.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.watermark_keys = load_keys(args.key_file)
    if args.command == "generate":
        if not 1 <= args.max_new_tokens <= 1023:
            parser.error("--max-new-tokens must be between 1 and 1023.")
        if not 0 < args.temperature <= 1:
            parser.error("--temperature must be in the interval (0, 1].")
        if not 2 <= args.top_k <= 50257:
            parser.error("--top-k must be between 2 and 50257 (the SynthID implementation requires top_k > 1).")
        prompt = args.prompt if args.prompt is not None else input("Enter the beginning of a text (preferably in English): ")
        if not prompt.strip():
            parser.error("The prompt cannot be empty.")
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    report = {
        "schema_version": 3,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "operation": args.command,
        "environment": environment_info(),
        "watermark_config": {k: str(v) if isinstance(v, torch.device) else v for k, v in watermark_config(args.watermark_keys).items()},
        "key_source": str(args.key_file.resolve()) if args.key_file else "public_upstream_defaults",
        "key_fingerprint_sha256": hashlib.sha256(json.dumps(args.watermark_keys, separators=(",", ":")).encode()).hexdigest(),
        "interpretation": "Technical demonstration; scores have no calibrated threshold. This is not a general AI-text detector.",
    }
    print(f"Key source: {report['key_source']}")
    print(f"Key fingerprint: {report['key_fingerprint_sha256'][:16]}")
    if args.command == "generate":
        report["prompt"] = prompt
        report["settings"] = {
            "model": "gpt2", "dtype": "float32", "seed": args.seed,
            "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
            "top_k": args.top_k, "top_p": 1.0, "warmup_max_new_tokens": min(8, args.max_new_tokens),
            "mode": args.mode,
        }
        modes = [False, True] if args.mode == "both" else [args.mode == "watermarked"]
        report["settings"]["generation_order"] = ["watermarked" if wm else "unwatermarked" for wm in modes]
        report["samples"] = []
        for watermarked in modes:
            sample = generate_one(tokenizer, prompt, watermarked, args)
            report["samples"].append(sample)
            print("\n" + ("WATERMARKED" if watermarked else "UNWATERMARKED"))
            print(sample["generated_text"])
            print_score(sample["detection_original_tokens"])
            print(f"Generation: {sample['metrics']['generation_seconds']:.3f} s; tokens: {sample['generated_tokens']}")
        print("\nTime and memory are exploratory measurements from one run, not performance conclusions.")
    else:
        if args.text_file:
            text = args.text_file.read_text(encoding="utf-8")
        else:
            text = args.text if args.text is not None else input("Paste the text to score: ")
        if not text.strip():
            parser.error("The text cannot be empty.")
        report["text"] = text
        report["detection"] = score_tokens(tokenizer.encode(text, add_special_tokens=False), tokenizer.eos_token_id, args.watermark_keys)
        print_score(report["detection"])
    write_report(report, args.output_dir)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except (KeyboardInterrupt, EOFError):
        print("\nExecution interrupted.", file=sys.stderr)
        sys.exit(130)
