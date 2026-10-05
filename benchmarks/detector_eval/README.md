# Weighted Mean detector calibration benchmark

This benchmark provides a deterministic baseline for calibrating the Weighted
Mean detector at fixed false-positive rates and comparing behavior across
effective token lengths.

It uses the repository's `weighted_mean_score` implementation on a synthetic
fixture. Unwatermarked g-values are sampled from Bernoulli(0.5), while the
watermarked fixture uses a configurable probability above 0.5. The fixture is
intended to exercise calibration and reporting reproducibly. It is not a model
of real generated text and should not be used to claim production detector
performance.

Run the default benchmark with:

```bash
python -m benchmarks.detector_eval.benchmark \
  --json-output /tmp/synthid_detector_benchmark.json
```

The default run evaluates effective token lengths 32, 64, 128, 256 and 512 at
false-positive-rate targets 0.001 and 0.01. It reports ROC-AUC, the calibrated
threshold and observed FPR, TPR at each target FPR, and deterministic bootstrap
95% confidence intervals.

The JSON artifact includes a schema version, complete benchmark configuration,
fixture parameters and per-length metrics so results can be compared across
changes. Use `--help` to override the lengths, FPR targets, sample count,
watermarking depth, fixture probability, bootstrap iterations or seed.

A future benchmark can replace the synthetic fixture with a fixed public text
corpus and add benign text transformations while retaining the same calibration
and reporting helpers.
