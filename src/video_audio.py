"""Synthesise the explainer video's soundtrack from its cue list.

Every sound is generated here from sine waves and filtered noise, so there are
no third-party audio files. The cues (time, sound, variant) come from the page
itself (`window.__cues` in video/template.html), timed to the animations.

    python src/video_audio.py            # video/soundtrack.wav + .mp3
    (build_video.py --audio calls this and muxes it into the MP4)
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SR = 44100
RNG = np.random.default_rng(7)
# A major pentatonic, two octaves: every tone in the film sits in one key
SCALE = 440 * 2 ** (np.array([0, 2, 4, 7, 9, 12, 14, 16, 19, 21, 24]) / 12)


def _t(sec: float) -> np.ndarray:
    return np.arange(int(sec * SR)) / SR


def _env(n: int, attack: float, decay: float) -> np.ndarray:
    t = np.arange(n) / SR
    return np.minimum(1, t / max(attack, 1e-4)) * np.exp(-t / decay)


def _lowpass(x: np.ndarray, cutoff) -> np.ndarray:
    """One-pole low-pass; `cutoff` may be an array (a sweep)."""
    cutoff = np.broadcast_to(np.asarray(cutoff, dtype=float), x.shape)
    a = 1 - np.exp(-2 * np.pi * cutoff / SR)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc += a[i] * (x[i] - acc)
        y[i] = acc
    return y


def _bell(f: float, sec: float, vol: float) -> np.ndarray:
    t = _t(sec)
    parts = [(1, 1, 1.0), (2.0, .5, .6), (3.01, .25, .35), (4.2, .12, .2)]  # (ratio, level, decay scale)
    return vol * sum(l * np.sin(2 * np.pi * f * r * t) * np.exp(-t / (sec * .35 * d)) for r, l, d in parts) \
        * np.minimum(1, t / .004)


def tick(v):
    f = SCALE[5 + int(v) % 4] * 2 * (1 + RNG.uniform(-.03, .03))
    t = _t(.06)
    return .10 * np.sin(2 * np.pi * f * t) * _env(len(t), .002, .012)


def pop(v):
    f = SCALE[int(v) % len(SCALE)]
    t = _t(.18)
    glide = f * (0.7 + 0.3 * np.minimum(1, t / .03))
    return .22 * np.sin(2 * np.pi * np.cumsum(glide) / SR) * _env(len(t), .003, .05)


def bar(v):
    f = SCALE[int(v) % len(SCALE)] / 2
    t = _t(.7)
    return .13 * (np.sin(2 * np.pi * f * t) + .3 * np.sin(4 * np.pi * f * t)) * _env(len(t), .02, .18)


def ding(v):
    return _bell(SCALE[5 + int(v) % 3], 1.6, .16)


def chime(v):
    a = _bell(SCALE[5 + int(v) % 3], 1.2, .12)
    b = _bell(SCALE[7 + int(v) % 3], 1.2, .12)
    out = np.zeros(len(a) + int(.09 * SR))
    out[:len(a)] += a
    out[int(.09 * SR):] += b
    return out


def chord(v):
    root = [220, 196, 247][int(v) % 3]
    t = _t(3.5)
    notes = root * np.array([1, 1.25, 1.5, 1.875, 2.25])  # major 9
    x = sum(np.sin(2 * np.pi * f * t + i) for i, f in enumerate(notes)) / len(notes)
    return .20 * x * np.minimum(1, t / .5) * np.exp(-t / 1.4)


def swell(v):
    t = _t(2.4)
    noise = _lowpass(RNG.standard_normal(len(t)), 200 + 2500 * (t / t[-1]) ** 2)
    env = np.sin(np.pi * t / t[-1]) ** 2
    return .10 * noise * env / (np.abs(noise).max() + 1e-9) * 2.5


def whoosh(v):
    sec = .45 if int(v) else .6
    t = _t(sec)
    sweep = 300 + 3500 * np.sin(np.pi * t / sec)
    n = RNG.standard_normal(len(t))
    band = _lowpass(n, sweep) - _lowpass(n, sweep * .3)
    band /= np.abs(band).max() + 1e-9
    return (.07 if int(v) else .11) * band * np.sin(np.pi * t / sec) ** 2


def thud(v):
    t = _t(.4)
    f = 55 + 45 * np.exp(-t / .05)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(len(t), .002, .12)
    click = _lowpass(RNG.standard_normal(len(t)), 1500) * _env(len(t), .0005, .006)
    return .40 * body + .12 * click


def clunk(v):
    t = _t(.25)
    body = np.sin(2 * np.pi * 180 * t) * _env(len(t), .001, .04)
    noise = _lowpass(RNG.standard_normal(len(t)), 900) * _env(len(t), .001, .025)
    return .22 * body + .15 * noise


def snip(v):
    t = _t(.05)
    n = RNG.standard_normal(len(t))
    hi = n - _lowpass(n, 3000)
    return .10 * hi * _env(len(t), .0005, .008)


def fall(v):
    t = _t(3.6)
    f = 700 * (200 / 700) ** (t / t[-1])
    return .07 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / t[-1]) * (1 + .3 * np.sin(2 * np.pi * 6 * t))


def fall2(v):
    t = _t(1.0)
    f = 520 * (260 / 520) ** np.minimum(1, t / .8)
    return .12 * np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(len(t), .01, .3)


def click(v):
    t = _t(.12)
    f = SCALE[int(v) % len(SCALE)]
    return .16 * np.sin(2 * np.pi * f * t) * _env(len(t), .001, .03)


def type_(v):
    t = _t(.03)
    n = RNG.standard_normal(len(t))
    tone = (2200 if int(v) else 1700) * (1 + RNG.uniform(-.05, .05))
    return .045 * (_lowpass(n, tone) * 3 + .3 * np.sin(2 * np.pi * tone * t)) * _env(len(t), .0005, .006)


SOUNDS = {"tick": tick, "pop": pop, "bar": bar, "ding": ding, "chime": chime, "chord": chord, "swell": swell,
          "whoosh": whoosh, "thud": thud, "clunk": clunk, "snip": snip, "fall": fall, "fall2": fall2,
          "click": click, "type": type_}


def pad(total: float, acts: list[tuple[float, int]]) -> np.ndarray:
    """A very quiet drone under everything; its root moves with each act."""
    roots = [110.0, 98.0, 123.47, 110.0, 130.81]
    t = _t(total)
    # glide over 1.5 s into each new act's root instead of jumping
    xs, ys = [0.0], [roots[acts[0][1]]]
    for start, act in acts[1:]:
        if roots[act] != ys[-1]:
            xs += [start, start + 1.5]
            ys += [ys[-1], roots[act]]
    root = np.interp(t, xs, ys)
    phase = 2 * np.pi * np.cumsum(root) / SR
    lfo = 1 + .25 * np.sin(2 * np.pi * t / 9)
    x = (np.sin(phase) + .6 * np.sin(1.5 * phase) + .35 * np.sin(2 * phase + 1)) * lfo
    fade = np.minimum(1, t / 2) * np.minimum(1, (total - t) / 2)
    return .022 * x * fade


def synth(cues: list, acts: list, total: float) -> np.ndarray:
    out = pad(total + 1, acts)
    for time, kind, variant in cues:
        s = SOUNDS[kind](variant)
        i = int(max(0, time) * SR)
        j = min(len(out), i + len(s))
        out[i:j] += s[:j - i]
    out = np.tanh(out * 1.3) / np.tanh(1.3)  # soft limiter
    return out / (np.abs(out).max() + 1e-9) * 0.89


def write_wav(x: np.ndarray, path: Path) -> None:
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
