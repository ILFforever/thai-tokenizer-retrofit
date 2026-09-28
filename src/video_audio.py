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
    return .07 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / t[-1])


def fall2(v):
    t = _t(1.0)
    f = 520 * (260 / 520) ** np.minimum(1, t / .8)
    return .12 * np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(len(t), .01, .3)


def type_(v):
    t = _t(.03)
    n = RNG.standard_normal(len(t))
    tone = (2200 if int(v) else 1700) * (1 + RNG.uniform(-.05, .05))
    return .045 * (_lowpass(n, tone) * 3 + .3 * np.sin(2 * np.pi * tone * t)) * _env(len(t), .0005, .006)


def tick(v):
    """A short note in one of five timbres (v % 5); pitch is a random scale note."""
    fam = int(v) % 5
    f = SCALE[5 + RNG.integers(0, 5)] * (1 + RNG.uniform(-.01, .01))
    if fam == 1:  # marimba: fundamental + a quick 4th partial
        t = _t(.3)
        return .13 * (np.sin(2 * np.pi * f / 2 * t) * _env(len(t), .001, .06)
                      + .4 * np.sin(2 * np.pi * 2 * f * t) * _env(len(t), .001, .012))
    if fam == 2:  # wood block: band of noise + a resonant knock
        t = _t(.09)
        n = RNG.standard_normal(len(t))
        band = _lowpass(n, 2600) - _lowpass(n, 700)
        return .10 * band * _env(len(t), .0005, .01) + .10 * np.sin(2 * np.pi * (700 + f / 2) * t) * _env(len(t), .0005, .02)
    if fam == 3:  # glass: high, inharmonic, short ring
        t = _t(.25)
        g = f * 3
        return .05 * (np.sin(2 * np.pi * g * t) + .5 * np.sin(2 * np.pi * 2.76 * g * t)) * _env(len(t), .001, .06)
    if fam == 4:  # bubble: fast upward glide
        t = _t(.09)
        glide = f * (1 + np.minimum(1, t / .04))
        return .11 * np.sin(2 * np.pi * np.cumsum(glide) / SR) * _env(len(t), .002, .025)
    t = _t(.06)  # soft blip
    return .10 * np.sin(2 * np.pi * 2 * f * t) * _env(len(t), .002, .012)


def pluck(v):
    """Karplus-Strong plucked string."""
    f = SCALE[int(v) % len(SCALE)] / 2
    n, N = int(.9 * SR), int(SR / f)
    y = np.zeros(n)
    y[:N] = RNG.uniform(-1, 1, N)
    for i in range(N, n):
        y[i] = .996 * .5 * (y[i - N] + y[i - N - 1])
    return .18 * _lowpass(y, 3000)


def blip(v):
    """Soft square-wave synth note."""
    f = SCALE[int(v) % len(SCALE)]
    t = _t(.35)
    return .07 * _lowpass(np.sign(np.sin(2 * np.pi * f * t)), 1800) * _env(len(t), .005, .09)


def snap(v):
    """Magnetic snap for a merge: a click and a note that drops into place."""
    f = SCALE[int(v) % len(SCALE)]
    t = _t(.16)
    glide = f * (1 + .6 * np.exp(-t / .012))
    tone = np.sin(2 * np.pi * np.cumsum(glide) / SR) * _env(len(t), .001, .045)
    n = RNG.standard_normal(len(t))
    return .15 * tone + .06 * (n - _lowpass(n, 2500)) * _env(len(t), .0003, .004)


def click(v):
    return snap(v)


def trans(v):
    """Scene change, four styles so consecutive cuts differ."""
    style = int(v) % 4
    if style == 1:  # riser
        t = _t(.35)
        n = RNG.standard_normal(len(t))
        x = _lowpass(n, 300 + 5000 * (t / t[-1]) ** 2)
        return .09 * x / (np.abs(x).max() + 1e-9) * (t / t[-1]) ** 1.5
    if style == 2:  # page turn: two soft brushes
        out = np.zeros(int(.25 * SR))
        for start in (0, .09):
            m = int(.07 * SR)
            n = _lowpass(RNG.standard_normal(m), 2200)
            i = int(start * SR)
            out[i:i + m] += .12 * n / (np.abs(n).max() + 1e-9) * np.sin(np.pi * np.arange(m) / m)
        return out
    if style == 3:  # airy low tom
        t = _t(.3)
        f = 90 + 60 * np.exp(-t / .04)
        return .16 * np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(len(t), .002, .09)
    return whoosh(1)


def impact(v):
    """New act: a low boom with a bright shimmer on top."""
    t = _t(1.4)
    f = 45 + 40 * np.exp(-t / .06)
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(len(t), .003, .35)
    air = _lowpass(RNG.standard_normal(len(t)), 900) * _env(len(t), .002, .08)
    return .38 * boom + .06 * air + _bell(SCALE[8], 1.4, .05)


def roll(v):
    """Counter rolling up: clicks that speed up, then stop."""
    out = np.zeros(int(3.2 * SR))
    t, gap, k = 0.0, .09, 0
    while t < 2.9:
        c = tick(2)
        c = c * (.5 + .5 * t / 2.9)
        i = int(t * SR)
        out[i:i + len(c)] += c[:len(out) - i]
        gap = max(.022, gap * .94)
        t += gap
        k += 1
    return .7 * out


def sparkle(v):
    """Scattered high glints, for the vocabulary grid lighting up."""
    out = np.zeros(int(1.8 * SR))
    for _ in range(36):
        f = RNG.uniform(2200, 5200)
        g = _t(.05)
        grain = .035 * np.sin(2 * np.pi * f * g) * _env(len(g), .003, .012)
        i = int(RNG.uniform(0, 1.7) * SR)
        out[i:i + len(grain)] += grain
    return out


def buzz(v):
    """'Wrong' - two short low buzzes."""
    out = np.zeros(int(.34 * SR))
    for start in (0, .16):
        t = _t(.13)
        x = _lowpass(np.sign(np.sin(2 * np.pi * 110 * t)), 900) * _env(len(t), .004, .06)
        i = int(start * SR)
        out[i:i + len(x)] += .10 * x
    return out


SOUNDS = {"tick": tick, "pop": pop, "bar": bar, "ding": ding, "chime": chime, "chord": chord, "swell": swell,
          "whoosh": whoosh, "thud": thud, "clunk": clunk, "snip": snip, "fall": fall, "fall2": fall2,
          "click": click, "type": type_, "pluck": pluck, "blip": blip, "snap": snap, "trans": trans,
          "impact": impact, "roll": roll, "sparkle": sparkle, "buzz": buzz}


def synth(cues: list, acts: list, total: float) -> np.ndarray:
    out = np.zeros(int((total + 1) * SR))
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
