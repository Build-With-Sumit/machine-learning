#!/usr/bin/env bash
# claude-video-eyes: give Claude vision over a video.
#   yt-dlp downloads it (or use a local file) -> ffmpeg rips N evenly-spaced
#   keyframes to a temp dir -> Claude Reads the frames (+ optional transcript).
# Zero API cost: the frames are read by YOUR Claude Code session.
#
#   extract_frames.sh <url|path> [num_frames]   (default 16 frames)
#
# Prints the frame dir + a newline list of frame paths for Claude to Read.
# YouTube / direct video URLs / local files work out of the box. Instagram
# needs auth — pass a local .mp4 (fetch it via Apify or browser cookies first).
set -uo pipefail

SRC="${1:?usage: extract_frames.sh <url|path> [num_frames]}"
N="${2:-16}"
WORK="$(mktemp -d 2>/dev/null || echo "${TMPDIR:-/tmp}/cve_$$")"; mkdir -p "$WORK"
VID="$WORK/video.mp4"

command -v ffmpeg >/dev/null 2>&1 || { echo "ERROR: ffmpeg not found on PATH" >&2; exit 3; }

if [ -f "$SRC" ]; then
  VID="$SRC"                                   # local file — use as-is
else
  command -v yt-dlp >/dev/null 2>&1 || YTDLP="python -m yt_dlp"
  YTDLP="${YTDLP:-yt-dlp}"
  echo "downloading via yt-dlp…" >&2
  if ! $YTDLP -q --no-warnings -f "mp4/best" -o "$VID" "$SRC" 2>"$WORK/ytdlp.err"; then
    echo "ERROR: download failed. If this is Instagram, it needs auth —" >&2
    echo "       fetch the .mp4 first (Apify / --cookies-from-browser) and pass the local path." >&2
    sed -n '1,3p' "$WORK/ytdlp.err" >&2
    exit 4
  fi
fi

# duration -> even frame spacing across the whole clip
DUR="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$VID" 2>/dev/null | cut -d. -f1)"
[ -z "${DUR:-}" ] || [ "$DUR" -lt 1 ] 2>/dev/null && DUR=1
FPS="$(python -c "print(max(0.05, min(2.0, $N/float($DUR))))" 2>/dev/null || echo 0.5)"

echo "extracting ~$N keyframes from ${DUR}s clip (fps=$FPS)…" >&2
ffmpeg -y -i "$VID" -vf "fps=$FPS,scale=768:-1" -qscale:v 3 "$WORK/frame_%03d.jpg" >/dev/null 2>&1

# optional audio track for transcription (whisper is not required by this skill)
ffmpeg -y -i "$VID" -vn -acodec libmp3lame -q:a 6 "$WORK/audio.mp3" >/dev/null 2>&1 || true

echo "FRAME_DIR=$WORK"
ls -1 "$WORK"/frame_*.jpg 2>/dev/null
[ -f "$WORK/audio.mp3" ] && echo "AUDIO=$WORK/audio.mp3"
exit 0
