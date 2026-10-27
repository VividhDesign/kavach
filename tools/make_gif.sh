#!/usr/bin/env bash
# Usage: tools/make_gif.sh outputs/clip1/out.mp4 docs/demo.gif [start_s] [duration_s]
# Two-pass palette GIF (palettegen/paletteuse): far better colours than a direct conversion.
set -euo pipefail
in="$1"; out="$2"; ss="${3:-0}"; dur="${4:-8}"
filters="fps=8,scale=600:-1:flags=lanczos"
ffmpeg -loglevel error -y -ss "$ss" -t "$dur" -i "$in" -vf "$filters,palettegen=stats_mode=diff" /tmp/kavach_palette.png
ffmpeg -loglevel error -y -ss "$ss" -t "$dur" -i "$in" -i /tmp/kavach_palette.png \
  -lavfi "$filters [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5" "$out"
ls -lh "$out"
