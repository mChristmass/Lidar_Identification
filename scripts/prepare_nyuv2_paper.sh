#!/usr/bin/env bash
set -euo pipefail

RAW_DIR="${1:-data/public_semseg/nyuv2/raw}"
OUT_DIR="${2:-data/public_semseg/nyuv2/processed}"
mkdir -p "$RAW_DIR" "$OUT_DIR"

download() {
  local url="$1"
  local destination="$2"
  local minimum_bytes="$3"
  if [[ -f "$destination" ]] && [[ $(wc -c < "$destination") -ge "$minimum_bytes" ]]; then
    echo "DOWNLOAD_OK existing=$destination bytes=$(wc -c < "$destination")"
    return
  fi
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 5 --retry-delay 5 --continue-at - --output "$destination" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget --continue --output-document="$destination" "$url"
  else
    echo "curl or wget is required" >&2
    exit 2
  fi
  if [[ ! -f "$destination" ]] || [[ $(wc -c < "$destination") -lt "$minimum_bytes" ]]; then
    echo "Incomplete download: $destination" >&2
    exit 3
  fi
}

download \
  "https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat" \
  "$RAW_DIR/nyu_depth_v2_labeled.mat" \
  2972037809
download \
  "https://raw.githubusercontent.com/ankurhanda/nyuv2-meta-data/master/splits.mat" \
  "$RAW_DIR/splits.mat" \
  2626
download \
  "https://raw.githubusercontent.com/ankurhanda/nyuv2-meta-data/master/labels40.mat" \
  "$RAW_DIR/labels40.mat" \
  14539282

python -u -m experiments.paper_benchmark_v1.prepare_nyuv2 \
  --labeled-mat "$RAW_DIR/nyu_depth_v2_labeled.mat" \
  --splits-mat "$RAW_DIR/splits.mat" \
  --labels40-mat "$RAW_DIR/labels40.mat" \
  --output-dir "$OUT_DIR"

python -u -m experiments.paper_benchmark_v1.run audit \
  --dataset nyuv2 --stage tune --nyuv2-dir "$OUT_DIR"
