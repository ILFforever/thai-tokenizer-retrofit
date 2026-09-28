"""Build the explainer video page and, optionally, render it to MP4.

    python src/video_data.py                 # refresh video/data.json
    python src/build_video.py                # video/index.html (playable page)
    python src/build_video.py --render       # + video/thai_token_retrofit.mp4
    python src/build_video.py --check        # screenshot one frame per scene
    python src/build_video.py --audio        # soundtrack (src/video_audio.py), muxed into the MP4

The page draws every frame from a time value (`__render(t)`), so rendering is
exact: step t at the frame rate, screenshot, pipe to ffmpeg. Needs playwright
(installed Chrome channel) and ffmpeg for --render.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
VID = ROOT / "video"


def build() -> Path:
    data = (VID / "data.json").read_text(encoding="utf-8")
    json.loads(data)  # fail early on a broken file
    html = (VID / "template.html").read_text(encoding="utf-8")
    assert "/*__DATA__*/null" in html
    # "</" inside JSON would close the <script> tag early
    html = html.replace("/*__DATA__*/null", data.replace("</", "<\\/"))
    out = VID / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size // 1024} KB)")
    return out


def _page(p, width: int, height: int):
    browser = p.chromium.launch(channel="chrome")
    page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.goto((VID / "index.html").as_uri())
    page.evaluate("window.__ready")
    # frame-capture mode: canvas only, full viewport
    page.evaluate("document.body.classList.add('capture')")
    return browser, page, errors


def check(out_dir: Path) -> None:
    from playwright.sync_api import sync_playwright

    out_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser, page, errors = _page(p, 1920, 1080)
        scenes = page.evaluate("window.__scenes")
        total = page.evaluate("window.__TOTAL")
        print(f"{len(scenes)} scenes, {total:.1f} s")
        canvas = page.locator("canvas").first
        for i, s in enumerate(scenes):
            for frac in (0.85,):
                page.evaluate(f"window.__render({s['start'] + s['dur'] * frac})")
                canvas.screenshot(path=str(out_dir / f"s{i + 1:02d}.jpg"), type="jpeg", quality=70)
        browser.close()
    print("errors:", errors or "none")


def render(path: Path, fps: int, width: int, height: int, ffmpeg: str) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser, page, errors = _page(p, width, height)
        total = page.evaluate("window.__TOTAL")
        n = int(total * fps)
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(fps), "-c:v", "mjpeg",
               "-i", "-", "-vf", f"scale={width}:{height}", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
        ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        canvas = page.locator("canvas").first
        t0 = time.time()
        for f in range(n):
            page.evaluate(f"window.__render({f / fps})")
            ff.stdin.write(canvas.screenshot(type="jpeg", quality=92))
            if f % (fps * 10) == 0:
                el = time.time() - t0
                print(f"  frame {f}/{n}  {el:.0f}s elapsed, ~{el / max(f, 1) * (n - f):.0f}s left", flush=True)
        ff.stdin.close()
        ff.wait()
        browser.close()
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB); errors: {errors or 'none'}")


def audio(ffmpeg: str, mp4: Path) -> None:
    """Soundtrack from the page's cue list -> video/soundtrack.mp3 (for the
    page) and, if the MP4 exists, a copy of it with sound."""
    from playwright.sync_api import sync_playwright

    from video_audio import synth, write_wav

    with sync_playwright() as p:
        browser, page, errors = _page(p, 1280, 720)
        cues, acts, total = page.evaluate("[window.__cues, window.__acts, window.__TOTAL]")
        browser.close()
    t0 = time.time()
    x = synth(cues, acts, total)
    wav = VID / "soundtrack.wav"
    write_wav(x, wav)
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(wav), "-c:a", "libmp3lame", "-b:a", "128k",
                    str(VID / "soundtrack.mp3")], check=True)
    print(f"soundtrack: {len(cues)} cues, {total:.0f} s, synthesised in {time.time() - t0:.0f}s")
    if mp4.exists():
        out = mp4.with_name(mp4.stem + "_sound.mp4")
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(mp4), "-i", str(wav), "-map", "0:v", "-map", "1:a",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out)],
                       check=True)
        print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--audio", action="store_true", help="soundtrack; muxed into the MP4 if it exists")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--out", default=str(VID / "thai_token_retrofit.mp4"))
    ap.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "W:/ffmpeg/bin/ffmpeg.exe")
    a = ap.parse_args()
    build()
    w, h = map(int, a.size.split("x"))
    if a.check:
        check(VID / "_check")
    if a.render:
        render(Path(a.out), a.fps, w, h, a.ffmpeg)
    if a.audio:
        audio(a.ffmpeg, Path(a.out))


if __name__ == "__main__":
    sys.exit(main())
