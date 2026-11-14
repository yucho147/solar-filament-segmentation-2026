"""位置チャネル（exp028）: 画像の画素から決まる「円盤中心からの距離」と「縦位置」を、画像の追加入力チャネルにする。

- rho: 円盤中心からの距離 ρ = sqrt((x-cx)^2 + (y-cy)^2) / R（縁で約1、画像の角で約1.5）。
- lat: 縦位置 |y-cy| / R。赤道を通る水平線からの距離の代理（画像が太陽の北を上にしているかは未検証。
       学習画像の正解の |y-cy|/R の分布が |x-cx|/R より偏るため、向きはおおむね共通と見ているが、緯度そのものではない）。
- 円盤の幾何（cx, cy, R）は固定値。学習887枚+テスト180枚の画素から縁（半径方向の輝度勾配が最大の円）を測り、
  99%の画像で中心は±数画素以内、半径は908.7±数画素だったため、画像ごとの推定はしない（scripts/posch_limb.py）。
- 位置マップは切片を切る前の画像全体（2048×2048）の座標で作り、切片の切出し・90度回転・反転を画像・教師と同じ幾何変換で適用する。
  明度/コントラスト変換は画像チャネルだけで、位置チャネルには掛けない。
- ファイル名の時刻・観測所は使わない。

有効化: Config.extra["position_channels"] が真のときだけ（Configのフィールドは増やさない）。
  true                                  … 既定（rho, lat と上の固定幾何）
  {channels: [rho, lat], geometry: {cx: 1023.0, cy: 1024.0, R: 904.0}}  … 省略した項目は既定
無効（キー無し・false・空）のときは既存の動作と同じ（このmoduleは何も変えない）。
モデルの入力は x[B, 1+k, H, W]（ch0=画像0〜1、ch1以降=位置チャネル。並びは channels の順）。
"""
from __future__ import annotations

import functools
import hashlib
import json

import numpy as np

NAMES = ("rho", "lat")
FORMULA = {"rho": "hypot(x - cx, y - cy) / R", "lat": "abs(y - cy) / R"}  # x,y は画素番号（最初の画素の中心=0）
DEFAULT_GEOMETRY = {"cx": 1023.0, "cy": 1024.0, "R": 904.0}
KEY = "position_channels"


def parse(extra: dict | None) -> dict | None:
    """Config.extra から位置チャネルの仕様 {channels, geometry} を返す。無効なら None。"""
    v = (extra or {}).get(KEY)
    if not v:
        return None
    v = {} if v is True else v
    if not isinstance(v, dict) or set(v) - {"channels", "geometry"}:
        raise ValueError(f"{KEY} must be true or a dict with keys channels/geometry; got {v!r}")
    names = list(v.get("channels", NAMES))
    if not names or len(set(names)) != len(names) or set(names) - set(NAMES):
        raise ValueError(f"{KEY}.channels must be unique names from {NAMES}; got {names}")
    geo = {**DEFAULT_GEOMETRY, **(v.get("geometry") or {})}
    if set(geo) != set(DEFAULT_GEOMETRY) or geo["R"] <= 0:
        raise ValueError(f"{KEY}.geometry needs cx, cy, R>0; got {geo}")
    return {"channels": names, "geometry": {k: float(geo[k]) for k in ("cx", "cy", "R")}}


def spec(cfg) -> dict | None:
    return parse(cfg.extra)


def n_channels(cfg) -> int:
    s = spec(cfg)
    return len(s["channels"]) if s else 0


@functools.lru_cache(maxsize=8)
def _maps(names: tuple, cx: float, cy: float, R: float, h: int, w: int) -> np.ndarray:
    yy = (np.arange(h, dtype=np.float64) - cy)[:, None]
    xx = (np.arange(w, dtype=np.float64) - cx)[None, :]
    out = []
    for n in names:
        a = np.hypot(yy, xx) / R if n == "rho" else np.broadcast_to(np.abs(yy) / R, (h, w))
        out.append(a.astype(np.float32))
    m = np.stack(out)
    m.setflags(write=False)
    return m


def maps(s: dict, h: int, w: int) -> np.ndarray:
    """仕様 s の位置マップ float32[k,h,w]（読み取り専用・キャッシュ済み）。"""
    g = s["geometry"]
    return _maps(tuple(s["channels"]), g["cx"], g["cy"], g["R"], int(h), int(w))


def full_map(cfg, shape) -> np.ndarray | None:
    """画像全体（shape=(H,W)）の位置マップ。無効なら None。"""
    s = spec(cfg)
    if s is None:
        return None
    if cfg.factor != 1:
        raise ValueError("position channels are defined in the native 2048 pixel frame (factor=1)")
    return maps(s, shape[0], shape[1])


def d4(a: np.ndarray, k: int, flip: int) -> np.ndarray:
    """[C,h,w] に native.forward の画像・教師と同じ順序の変換（90度回転k回→左右反転）を掛ける。"""
    if k:
        a = np.rot90(a, k, axes=(1, 2))
    if flip:
        a = a[:, :, ::-1]
    return np.ascontiguousarray(a, dtype=np.float32)


def crop_d4(pm: np.ndarray, y0: int, x0: int, w: int, k: int, flip: int) -> np.ndarray:
    """画像全体の位置マップから切片(y0,x0,w)を切り出し、画像・教師と同じD4変換を掛ける。"""
    return d4(pm[:, y0:y0 + w, x0:x0 + w], k, flip)


def model_opts(cfg) -> dict:
    """engine.build がモデルへ渡す追加の構成。無効なら空。"""
    s = spec(cfg)
    if s is None:
        return {}
    if cfg.arch != "dinov3_seg":
        raise ValueError(f"{KEY} is only supported for arch=dinov3_seg (got {cfg.arch})")
    return {"pos_channels": len(s["channels"])}


def check_state(state: dict, cfg) -> None:
    """重みの位置チャネル有無・本数が設定と一致することを確認する（不一致は読み込まずに拒否）。"""
    want = n_channels(cfg)
    w = state.get("pos_in.weight")
    have = 0 if w is None else int(w.shape[1])
    if have != want:
        raise ValueError(f"position channel mismatch: weights have {have}, config {KEY} requires {want}")


def formula_hash(names: list[str]) -> str:
    """計算式の識別子: チャネル名・式の文字列・基準幾何での64×64の数値のSHA-256（先頭16桁）。"""
    ref = _maps(tuple(names), 31.5, 31.5, 30.0, 64, 64)
    h = hashlib.sha256(json.dumps([names, [FORMULA[n] for n in names]]).encode())
    h.update(np.ascontiguousarray(ref).tobytes())
    return h.hexdigest()[:16]


def record(cfg) -> dict | None:
    """manifest・.aug.json へ残す記録。無効なら None。"""
    s = spec(cfg)
    if s is None:
        return None
    return {"channels": s["channels"], "geometry": s["geometry"], "formula": {n: FORMULA[n] for n in s["channels"]},
            "formula_hash": formula_hash(s["channels"]), "channel_order": "x[:,0]=image, x[:,1:]=channels in listed order",
            "model_input_normalization": "(p - 0.5) / 0.3 (inside the model)",
            "applied_transform": "same crop/rot90/flip as image and target; brightness/contrast never applied",
            "geometry_source": "scripts/posch_limb.py (radial-gradient limb fit on train+test pixels)"}
