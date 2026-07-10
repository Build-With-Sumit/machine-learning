# claude-video-eyes — give Claude real vision over a video

A small, self-contained [Claude Code](https://docs.anthropic.com/en/docs/claude-code)
skill that lets Claude **watch** a video instead of only reading its transcript.
`yt-dlp` downloads the clip (or you pass a local file), `ffmpeg` rips evenly-spaced
keyframes to a temp dir, and Claude **reads those frames** — so it sees the cuts,
the on-screen text, and the visual hook a transcript-only tool misses.

Zero API cost: the frames are read by your own Claude Code session, fully local.

## Quick start

```bash
# 1. extract ~16 evenly-spaced keyframes (and an audio track) into a temp dir
bash claude-video-eyes/scripts/extract_frames.sh <youtube-url | video-url | ./local.mp4> [num_frames]

# 2. in Claude Code, Read each printed frame_*.jpg in order — that's the vision
# 3. (optional) transcribe the printed audio.mp3 if the spoken words matter
```

The script prints `FRAME_DIR=…`, the list of frame paths, and `AUDIO=…`.

- **YouTube / direct video URLs / local `.mp4`** work out of the box.
- **Instagram** is auth-walled in yt-dlp — fetch the `.mp4` first (via Apify or
  `--cookies-from-browser`) and pass the local path.

## Requirements

`ffmpeg` / `ffprobe` and `yt-dlp` (or `python -m yt_dlp`) on your `PATH`.

## As a Claude Code skill

Drop this folder under `.claude/skills/` and invoke it by name
("watch this video with vision", "what's on screen in this video"). See
[`SKILL.md`](SKILL.md) for the full agent-facing procedure.
