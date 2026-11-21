#!/usr/bin/env bash
set -euo pipefail

for n in $(seq 22 40); do
  scene=$(printf "scene-%03d" "$n")
  manifest="data/manifests/phone/${scene}.json"

  for mode in single multi; do
    output="outputs/phone_$(printf '%03d' "$n")_m0_${mode}"

    echo ""
    echo "======================================"
    echo "Scene: $scene | Mode: $mode"
    echo "======================================"

    if [[ -f "$output/metrics.json" && -f "$output/metrics.csv" ]]; then
      echo "SKIP: evaluation already completed"
      continue
    fi

    if [[ ! -f "$manifest" ]]; then
      echo "ERROR: missing manifest: $manifest"
      exit 1
    fi

    if [[ ! -f "$output/reference_prediction.npz" ]]; then
      bash scripts/local.sh scripts/run_baseline.py \
        --manifest "$manifest" \
        --mode "$mode" \
        --device cpu \
        --output "$output"
    else
      echo "Existing prediction found; running evaluation only"
    fi

    bash scripts/local.sh scripts/evaluate.py \
      --manifest "$manifest" \
      --run "$output"

    echo "DONE: $scene / $mode"
  done
done

echo ""
echo "All remaining phone baselines completed."
