"""学習時データ拡張。画像とソフト教師の幾何変換を同期し、画質変動は画像だけに適用する。

変換ごとの確率・強度は spec（dict）で持ち、乱数は呼び出し側の Generator から取る。
同じ (seed, sample index) なら worker 数に依らず同じ切片・同じ変換になる。
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import albumentations as A  # AlbumentationsX（pip: albumentationsx）
import cv2
import numpy as np
from scipy import ndimage as ndi

# 変換名。幾何（画像と教師に同じ変換）／画質（画像のみ）。AlbumentationsX 2.5系（import名はalbumentations）
GEOMETRIC = ("d4", "affine", "elastic", "grid", "optical", "tps")
PHOTOMETRIC = ("illum", "bc", "gamma", "tonecurve", "blur", "downscale", "noise", "jpeg", "pixeldrop", "saltpepper")
NAMES = GEOMETRIC + PHOTOMETRIC

# 切片の選び方 (sampling): fg=前景画素を含む切片、disc=太陽円盤内の任意位置、bg=円盤内で前景ゼロ、uniform=画像全体で一様
SAMPLING = {
    "legacy": {"fg": 0.7, "disc": 0.0, "bg": 0.0, "min_disc": 0.0},
    "mixed": {"fg": 0.6, "disc": 0.3, "bg": 0.0, "min_disc": 0.3},
    # bg=円盤内で前景ゼロの切片。過検出（誤検出が多い）対策として負例だけの切片を明示的に混ぜる
    "bg": {"fg": 0.5, "disc": 0.0, "bg": 0.3, "min_disc": 0.5},
}

_LIN = dict(interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_LINEAR)

# 変換名 -> albumentationsの変換を作る関数。強度はspec側（PRESETS）で持つ。
_FACTORY = {
    "affine": lambda q: A.Affine(scale=tuple(q["scale"]), rotate=tuple(q["rotate"]), **_LIN, p=1.0),
    "elastic": lambda q: A.ElasticTransform(displacement_range=tuple(q["displacement"]),
                                            control_grid_shape=tuple(q.get("grid", (7, 7))), **_LIN, p=1.0),
    "grid": lambda q: A.GridDistortion(num_steps=q.get("steps", 5), distort_range=tuple(q["distort"]), **_LIN, p=1.0),
    "optical": lambda q: A.OpticalDistortion(distort_range=tuple(q["distort"]), **_LIN, p=1.0),
    "tps": lambda q: A.ThinPlateSpline(scale_range=tuple(q["scale"]), **_LIN, p=1.0),
    "illum": lambda q: A.Illumination(mode=q.get("mode", "linear"), intensity_range=tuple(q["intensity"]),
                                      effect_type="both", p=1.0),
    "gamma": lambda q: A.RandomGamma(gamma_range=tuple(g * 100 for g in q["gamma"]), p=1.0),
    "tonecurve": lambda q: A.RandomToneCurve(scale=q["scale"], p=1.0),
    "blur": lambda q: A.GaussianBlur(blur_range=(0, 0), sigma_range=tuple(q["sigma"]), p=1.0),
    "downscale": lambda q: A.Downscale(scale_range=tuple(q["scale"]),
                                       interpolation_pair={"downscale": cv2.INTER_AREA, "upscale": cv2.INTER_LINEAR}, p=1.0),
    "noise": lambda q: A.GaussNoise(std_range=tuple(q["std"]), per_channel=False, p=1.0),
    "jpeg": lambda q: A.ImageCompression(quality_range=tuple(int(x) for x in q["quality"]), p=1.0),
    "pixeldrop": lambda q: A.PixelDropout(dropout_prob=q["prob"], drop_value=0, p=1.0),
    "saltpepper": lambda q: A.SaltAndPepper(amount_range=tuple(q["amount"]), salt_vs_pepper_range=(0.5, 0.5), p=1.0),
}

PRESETS: dict[str, dict] = {
    # 既存実装（exp001/a1と同じ内容）：90度回転・反転・明るさ/コントラストのみ
    "legacy": {
        "d4": {"p": 1.0},
        "bc": {"p": 1.0, "contrast": (0.8, 1.25), "brightness": (-0.08, 0.08)},
    },
    # 最初の本格構成：ぼけ・ノイズ・JPEG・形状変形・画素欠損を含める。強度は scripts/visualize_augmentation.py で
    # 実画像と正解を見て決めた初期値（細い枝が消えない範囲）。比較実験で個別に強弱を振る。
    "full": {
        "d4": {"p": 1.0},
        "affine": {"p": 0.5, "scale": (0.8, 1.25), "rotate": (-15.0, 15.0)},
        "elastic": {"p": 0.2, "displacement": (0.005, 0.02)},
        "grid": {"p": 0.1, "distort": (-0.1, 0.1)},
        "optical": {"p": 0.1, "distort": (-0.05, 0.05)},
        "tps": {"p": 0.1, "scale": (0.0, 0.03)},
        "illum": {"p": 0.2, "intensity": (0.01, 0.1)},
        "bc": {"p": 0.8, "contrast": (0.8, 1.25), "brightness": (-0.08, 0.08)},
        "gamma": {"p": 0.3, "gamma": (0.7, 1.4)},
        "tonecurve": {"p": 0.15, "scale": 0.05},
        "blur": {"p": 0.3, "sigma": (0.3, 1.2)},
        "downscale": {"p": 0.1, "scale": (0.6, 0.9)},
        "noise": {"p": 0.3, "std": (0.005, 0.03)},
        "jpeg": {"p": 0.2, "quality": (60, 95)},
        "pixeldrop": {"p": 0.15, "prob": 0.002},
        "saltpepper": {"p": 0.1, "amount": (0.0005, 0.003)},
    },
}


def resolve(aug: dict | str | None) -> dict[str, dict]:
    """設定 → 変換spec。形式: {preset: full, off: [blur], set: {noise: {p: 0.5}}} か preset名。"""
    if aug is None:
        aug = "legacy"
    if isinstance(aug, str):
        aug = {"preset": aug}
    aug = {("off" if k is False else k): v for k, v in aug.items()}  # YAML 1.1 は裸の off を False にする
    if "drop" in aug:
        aug = {**{k: v for k, v in aug.items() if k != "drop"}, "off": aug["drop"]}
    unknown = set(aug) - {"preset", "off", "set"}
    if unknown:
        raise ValueError(f"unknown aug keys {sorted(unknown)}")
    spec = copy.deepcopy(PRESETS[aug.get("preset", "legacy")])
    for k in aug.get("off", []):
        spec.pop(k, None)
    for k, v in aug.get("set", {}).items():
        if k not in NAMES:
            raise ValueError(f"unknown transform {k}")
        if k not in spec and k not in _FACTORY and k not in ("d4", "bc"):
            raise ValueError(f"unknown transform {k}")
        spec.setdefault(k, {"p": 1.0}).update(v)
    for k in spec:
        if k not in NAMES:
            raise ValueError(f"unknown transform {k}")
    return spec


def window_size(spec: dict, crop: int) -> int:
    """幾何変換後に中央を切り出すので、切出し窓を余裕を持って大きくする。"""
    if any(k in spec for k in ("affine", "elastic", "grid", "optical", "tps")):
        return int(round(crop * 1.5 / 2)) * 2
    return crop


class Pipeline:
    """画像uint8[w,w]と教師float32[w,w]を受け、幾何→中央crop→画質の順に適用する。

    AlbumentationsX は2次元配列だとAffine等が内部で失敗するので、(H,W,1)で渡す。
    """

    def __init__(self, spec: dict[str, dict], crop: int):
        self.spec, self.crop = spec, crop
        self.win = window_size(spec, crop)
        self.geo = [(n, _FACTORY[n](spec[n])) for n in GEOMETRIC if n in spec and n in _FACTORY]
        self.photo = [(n, _FACTORY[n](spec[n])) for n in PHOTOMETRIC if n in spec and n in _FACTORY]

    @staticmethod
    def _seed(t, rng):
        t.set_random_seed(int(rng.integers(2**31 - 1)))

    def __call__(self, img: np.ndarray, tgt: np.ndarray, rng: np.random.Generator):
        """戻り値は中央crop後の (img uint8, tgt float32, applied)。applied は変換名→適用したか。"""
        applied = dict.fromkeys(NAMES, False)

        def hit(name):
            sp = self.spec.get(name)
            ok = sp is not None and rng.random() < sp["p"]
            applied[name] = bool(ok)
            return ok

        if hit("d4"):
            k = int(rng.integers(4))
            img, tgt = np.rot90(img, k), np.rot90(tgt, k)
            if rng.random() < 0.5:
                img, tgt = img[:, ::-1], tgt[:, ::-1]
        img, tgt = np.ascontiguousarray(img)[..., None], np.ascontiguousarray(tgt)[..., None]
        for name, t in self.geo:
            if hit(name):
                self._seed(t, rng)
                o = t(image=img, mask=tgt)
                img, tgt = o["image"], o["mask"]
        if img.shape[0] != self.crop:
            o = (img.shape[0] - self.crop) // 2
            img, tgt = img[o : o + self.crop, o : o + self.crop], tgt[o : o + self.crop, o : o + self.crop]
        img = img.reshape(img.shape[:2])
        if hit("bc"):
            sp = self.spec["bc"]
            x = img.astype(np.float32) / 255
            x = (x - 0.5) * rng.uniform(*sp["contrast"]) + 0.5 + rng.uniform(*sp["brightness"])
            img = np.clip(x * 255 + 0.5, 0, 255).astype(np.uint8)
        img = img[..., None]
        for name, t in self.photo:
            if hit(name):
                self._seed(t, rng)
                img = t(image=img)["image"]
        img = np.asarray(img).reshape(img.shape[:2])
        return (np.ascontiguousarray(img, dtype=np.uint8),
                np.ascontiguousarray(np.asarray(tgt).reshape(img.shape), dtype=np.float32), applied)


@dataclass
class Disc:
    """太陽円盤の画素集合。切片の中心を円盤内から選ぶために使う。"""

    mask: np.ndarray  # bool[H,W]
    centers: np.ndarray  # int[N,2] 円盤を縁から少し削った内側の画素（間引き済み）


def find_disc(img: np.ndarray, thr: int = 12, stride: int = 4) -> Disc:
    """円盤外はほぼ黒。軽く平滑化→しきい値→最大連結成分→穴埋め。"""
    sm = cv2.GaussianBlur(img, (0, 0), 3)
    lab, n = ndi.label(sm > thr)
    if n == 0:
        return Disc(np.zeros(img.shape, bool), np.zeros((0, 2), int))
    size = ndi.sum(np.ones_like(lab), lab, np.arange(1, n + 1))
    m = ndi.binary_fill_holes(lab == (1 + int(np.argmax(size))))
    r = int(np.sqrt(m.sum() / np.pi))
    inner = ndi.binary_erosion(m, iterations=max(1, r // 20), border_value=0)
    ys, xs = np.nonzero(inner[::stride, ::stride])
    return Disc(m, np.stack([ys * stride, xs * stride], 1))
