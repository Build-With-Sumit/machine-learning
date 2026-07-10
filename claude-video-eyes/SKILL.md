---
name: claude-video-eyes
description: Give Claude real vision over a video (YouTube / direct URL / local .mp4) — yt-dlp downloads it, ffmpeg rips evenly-spaced keyframes, and Claude READS the frames alongside the transcript, so it sees the cuts, on-screen text, and visual hook that a transcript-only tool misses. Zero API cost — the frames are read by your own Claude Code session, fully local. Invoke on "watch/analyze this video with vision", "what's on screen in this video", or when handed a video URL to review. Instagram reels need auth — fetch the .mp4 first (Apify/cookies) and pass the local path.
---

# claude-video-eyes

Transcript-only "analyze this video" tools pull the words and miss half of what
a good video is — the cuts, the on-screen text, the visual hook. This skill gives
Claude **eyes**: download → keyframes → Claude reads the frames like a flip-book
while it reads the transcript. Runs local, uses your Claude Code session (no API
key, no per-token cost).

This is a generalized, self-contained video-frame-vision skill — the reusable
core of "let Claude actually watch the video." Point it at a URL or a local file
and it hands your Claude Code session an evenly-spaced set of keyframes to read.

## Procedure

1. **Extract frames** (and audio) into a temp dir:
   ```bash
   bash claude-video-eyes/scripts/extract_frames.sh <url|path> [num_frames]
   ```
   Default 16 frames, evenly spaced across the clip, scaled to 768px. It prints
   `FRAME_DIR=…` then the frame paths (and `AUDIO=…` if extracted).
   - **YouTube / direct video URL / local `.mp4`** → works out of the box.
   - **Instagram** → yt-dlp is auth-walled. Fetch the `.mp4` first (via Apify, or
     `--cookies-from-browser`) and pass the **local path**.

2. **Read the frames.** Use the Read tool on each `frame_*.jpg` (they render as
   images — this is the vision). Read them in order so you perceive the sequence.

3. **Transcribe if the words matter.** If `AUDIO=…` was printed and you need the
   spoken track, transcribe `audio.mp3` (faster-whisper locally, or ask the user).
   Frames alone are enough for "what's on screen / what's the visual hook".

4. **Answer against BOTH.** Combine what you saw (frames) with what was said
   (transcript). Explicitly flag anything that's on screen but *not* in the words —
   that gap is the whole point of having eyes.

## Notes
- Needs `ffmpeg`/`ffprobe` and `yt-dlp` (or `python -m yt_dlp`) on PATH.
- Frame count: 12–20 is plenty for a reel; bump to 30–40 for a long video (cost
  is just Read calls). More frames = finer temporal resolution.
- Temp dirs aren't cleaned automatically — they're under the system temp dir.
- To use it as a Claude Code skill, drop this folder under `.claude/skills/` and
  invoke it by name; standalone, run the script directly as shown above.
