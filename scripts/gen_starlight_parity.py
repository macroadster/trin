#!/usr/bin/env python3
"""Parity: load starlight_fixture.gguf into BalancedStarlightDetector (eval) and compare heads.

Minimal GGUF F32 reader + architecture mirrored from starlight trainer.py.
Usage (from trin root):
  python3 scripts/gen_starlight_parity.py
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
GGUF = ROOT / "demos/weights/starlight_fixture.gguf"
IN_DIR = ROOT / "demos/weights/starlight_inputs"
H = W = 32


class BalancedStarlightDetector(nn.Module):
    def __init__(self, meta_weight=0.3, proj_dim=64):
        super().__init__()
        self.meta_weight = meta_weight
        self.proj_dim = proj_dim
        self.meta_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.meta_proj = nn.Linear(128 * 16, proj_dim)
        def conv2d_stack(in_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
                nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.AdaptiveAvgPool2d(4),
            )
        self.alpha_conv = conv2d_stack(2)
        self.alpha_proj = nn.Linear(64 * 4 * 4, proj_dim)
        self.lsb_conv = conv2d_stack(3)
        self.lsb_proj = nn.Linear(64 * 4 * 4, proj_dim)
        self.palette_fc = nn.Sequential(
            nn.Linear(768, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, proj_dim),
        )
        self.format_features_fc = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 32))
        self.content_features_fc = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 32))
        self.pixel_conv = conv2d_stack(3)
        self.pixel_proj = nn.Linear(64 * 4 * 4, proj_dim)
        self.palette_lsb_conv = conv2d_stack(1)
        self.palette_lsb_proj = nn.Linear(64 * 4 * 4, proj_dim)
        self.fusion_dim = 6 * proj_dim + 2 * 32
        self.stream_gate = nn.Sequential(nn.Linear(self.fusion_dim, 8), nn.Sigmoid())
        self.fusion = nn.Sequential(
            nn.Linear(self.fusion_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 128 + 64),
        )
        self.stego_head = nn.Linear(64, 1)
        self.method_head = nn.Linear(64, 5)
        self.bit_order_head = nn.Linear(64, 3)
        self.embedding_head = nn.Linear(64, 64)

    def forward(self, pixel_tensor, meta, alpha, lsb, palette, palette_lsb, format_features, content_features):
        pixel_tensor = self.pixel_proj(self.pixel_conv(pixel_tensor).reshape(1, -1))
        meta = self.meta_proj(self.meta_conv(meta.unsqueeze(1)).reshape(1, -1)) * self.meta_weight
        alpha = self.alpha_proj(self.alpha_conv(alpha).reshape(1, -1))
        lsb = self.lsb_proj(self.lsb_conv(lsb).reshape(1, -1))
        palette = self.palette_fc(palette)
        palette_lsb = self.palette_lsb_proj(self.palette_lsb_conv(palette_lsb).reshape(1, -1))
        format_features = self.format_features_fc(format_features)
        content_features = self.content_features_fc(content_features)
        streams = [pixel_tensor, meta, alpha, lsb, palette, palette_lsb, format_features, content_features]
        all_streams = torch.cat(streams, dim=1)
        gate = self.stream_gate(all_streams)
        gated = [s * gate[:, i : i + 1] for i, s in enumerate(streams)]
        fused = self.fusion(torch.cat(gated, dim=1))
        cls = fused[:, 128:]
        return (
            self.stego_head(cls),
            self.method_head(cls),
            self.bit_order_head(cls),
            self.embedding_head(cls),
        )


def read_gguf_f32(path: Path) -> dict[str, torch.Tensor]:
    data = path.read_bytes()
    magic, ver = struct.unpack_from("<II", data, 0)
    if magic != 0x46554747:
        raise ValueError("bad magic")
    n_tensors, n_kv = struct.unpack_from("<QQ", data, 8)
    off = 24

    def read_str(o):
        (n,) = struct.unpack_from("<Q", data, o)
        o += 8
        s = data[o : o + n].decode("utf-8")
        return s, o + n

    # skip KV
    for _ in range(n_kv):
        _, off = read_str(off)
        (vt,) = struct.unpack_from("<I", data, off)
        off += 4
        if vt == 8:  # string
            _, off = read_str(off)
        elif vt == 4:  # u32
            off += 4
        elif vt == 6:  # f32
            off += 4
        elif vt == 7:  # bool
            off += 1
        elif vt == 5:  # i32
            off += 4
        elif vt == 9:  # u64
            off += 8
        elif vt == 10:  # i64
            off += 8
        else:
            # try common: STRING already handled; UINT32 handled
            raise ValueError(f"unsupported kv type {vt} at {off}")

    infos = []
    for _ in range(n_tensors):
        name, off = read_str(off)
        (ndims,) = struct.unpack_from("<I", data, off)
        off += 4
        dims = list(struct.unpack_from("<" + "Q" * ndims, data, off))
        off += 8 * ndims
        (typ,) = struct.unpack_from("<I", data, off)
        off += 4
        (toff,) = struct.unpack_from("<Q", data, off)
        off += 8
        infos.append((name, dims, typ, toff))

    # data section starts at aligned off
    align = 32
    data_base = (off + align - 1) // align * align
    out = {}
    for name, dims, typ, toff in infos:
        if typ != 0:
            raise ValueError(f"{name} not F32")
        n = 1
        for d in dims:
            n *= d
        start = data_base + toff
        vals = struct.unpack_from("<" + "f" * n, data, start)
        # GGUF dims[0] is first written dim = our shape[0] (PyTorch out, in, ...)
        t = torch.tensor(vals, dtype=torch.float32).reshape(*[int(d) for d in dims])
        out[name] = t
    return out


def load_csv(path: Path) -> torch.Tensor:
    text = path.read_text().strip().replace("\n", ",")
    parts = [p for p in text.replace(";", ",").split(",") if p.strip() and not p.strip().startswith("#")]
    return torch.tensor([float(p) for p in parts], dtype=torch.float32)


def main():
    if not GGUF.exists():
        print("missing", GGUF, file=sys.stderr)
        sys.exit(1)
    tensors = read_gguf_f32(GGUF)
    model = BalancedStarlightDetector(meta_weight=0.3)
    sd = model.state_dict()
    # map GGUF tensors into state_dict; BN num_batches_tracked left default
    loaded = 0
    with torch.no_grad():
        for k in list(sd.keys()):
            if k in tensors:
                if sd[k].shape != tensors[k].shape:
                    print("shape mismatch", k, sd[k].shape, tensors[k].shape)
                    sys.exit(1)
                sd[k].copy_(tensors[k])
                loaded += 1
    model.load_state_dict(sd)
    model.eval()
    print(f"loaded {loaded} tensors from GGUF into state_dict")

    pixel = load_csv(IN_DIR / "pixel.csv").reshape(1, 3, H, W)
    meta = load_csv(IN_DIR / "meta.csv").reshape(1, 2048)
    alpha = load_csv(IN_DIR / "alpha.csv").reshape(1, 2, H, W)
    lsb = load_csv(IN_DIR / "lsb.csv").reshape(1, 3, H, W)
    palette = load_csv(IN_DIR / "palette.csv").reshape(1, 768)
    plsb = load_csv(IN_DIR / "palette_lsb.csv").reshape(1, 1, H, W)
    fmt = load_csv(IN_DIR / "format_features.csv").reshape(1, 6)
    content = load_csv(IN_DIR / "content_features.csv").reshape(1, 6)

    with torch.no_grad():
        stego, method, bit, emb = model(pixel, meta, alpha, lsb, palette, plsb, fmt, content)

    exp_path = ROOT / "demos/weights/starlight_expected.csv"
    # skip comment lines
    raw = exp_path.read_text().strip().splitlines()
    line = [ln for ln in raw if not ln.startswith("#")][-1]
    exp = [float(x) for x in line.split(",") if x]
    got = torch.cat([stego.reshape(-1), method.reshape(-1), bit.reshape(-1), emb.reshape(-1)]).tolist()
    assert len(got) == len(exp) == 73
    max_abs = max(abs(a - b) for a, b in zip(got, exp))
    print(f"torch_stego={got[0]:.8g} go_stego={exp[0]:.8g}")
    print(f"max_abs_diff={max_abs:.6g}")
    thr = 1e-4
    ok = max_abs < thr
    print("parity_ok" if ok else "parity_FAIL", f"threshold={thr}")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
