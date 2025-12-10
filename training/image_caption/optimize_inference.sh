#!/usr/bin/env bash
set -euo pipefail

# Wrapper for training.image_caption.optimize_inference
# - Names log files as optimize_inference-<limit>-<multi-grid-args>.log
# - If the log exists, appends a numeric suffix (-2, -3, ...)

args=("$@")

grid_args=(beam_width length_penalty repetition_penalty no_repeat_ngram_size max_length)
declare -A is_multi=()
limit_val=""

# Detect limit and which grid args have multiple values
i=0
while [[ $i -lt ${#args[@]} ]]; do
  token="${args[$i]}"
  case "$token" in
    --limit)
      if [[ $((i + 1)) -lt ${#args[@]} ]]; then
        limit_val="${args[$((i + 1))]}"
      fi
      i=$((i + 2))
      ;;
    --beam_width|--length_penalty|--repetition_penalty|--no_repeat_ngram_size|--max_length)
      name="${token#--}"
      count=0
      j=$((i + 1))
      while [[ $j -lt ${#args[@]} ]]; do
        next="${args[$j]}"
        if [[ "$next" == --* ]]; then
          break
        fi
        count=$((count + 1))
        j=$((j + 1))
      done
      if [[ $count -gt 1 ]]; then
        is_multi["$name"]=1
      fi
      i=$j
      ;;
    *)
      i=$((i + 1))
      ;;
  esac
done

# Build log filename parts
parts=("optimize_inference")
parts+=("${limit_val:-limit}")
for name in "${grid_args[@]}"; do
  if [[ -n "${is_multi[$name]:-}" ]]; then
    parts+=("$name")
  fi
done

IFS=- read -r -a _ <<<"" # reset any inherited IFS side effects
IFS=-; log_base="${parts[*]}"; unset IFS
log_file="${log_base}.log"

# If file exists, add numeric suffix
suffix=2
while [[ -e "$log_file" ]]; do
  log_file="${log_base}-${suffix}.log"
  suffix=$((suffix + 1))
done

raw_log="${log_file%.log}.raw.log"

# Run and capture raw output (with tqdm carriage returns)
python -m training.image_caption.optimize_inference "${args[@]}" 2>&1 | tee "$raw_log"

# Postprocess: normalize CR, drop tqdm lines and blanks
perl -pe 's/\r/\n/g' "$raw_log" \
  | sed '/^Inferencing:/d' \
  | sed '/^$/d' \
  > "$log_file"

echo "Cleaned log written to $log_file (raw log: $raw_log)"

