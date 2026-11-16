"""Patch models/qwen3.5-9b-q4s.bin into an abliterated Qwen3.5-9B model (default: huihui-ai/Huihui-Qwen3.5-9B-abliterated).

Because qwen3.5-9b-q4s.bin uses Garlic LLM's custom PS4 GPU tensor layout (PMQ5 / q4s + q3s + q4row), raw GGUF or
safetensors files from Hugging Face cannot be copied directly to the PS4.

However, comparing huihui-ai/Huihui-Qwen3.5-9B-abliterated against Qwen/Qwen3.5-9B shows that:
  - qwen3.5-9b-mtp.bin (all mtp.* tensors and embed_tokens) is 100% identical (no change needed).
  - In qwen3.5-9b-q4s.bin, all 3-bit gate/up (wgu), wqkv, w_in, norms, lm_head and embed_tokens are 100% identical.
  - Only the 62 output projection matrices (layers 1..31 wo / wout and wdown, all in q4s format) are modified.

This script streams ONLY those 62 tensors via HTTP Range requests from Hugging Face, quantizes them to q4s, and patches
them at their exact byte offsets in models/qwen3.5-9b-q4s.bin (saving a backup or writing in place). No Modal account
and no 18 GB full checkpoint download are required.

Usage:
  python pc/abliterate_9b.py
  python pc/abliterate_9b.py --repo=huihui-ai/Huihui-Qwen3.5-9B-abliterated
  python pc/upload_model.py --ip 192.168.1.29 models/qwen3.5-9b-q4s.bin
"""

import json
import shutil
import struct
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from q4 import quantize_q4  # noqa: E402

REPO = "huihui-ai/Huihui-Qwen3.5-9B-abliterated"
for a in sys.argv[1:]:
    if a.startswith("--repo="):
        REPO = a.split("=", 1)[1]

MODEL_BIN = ROOT / "models" / "qwen3.5-9b-q4s.bin"
BACKUP_BIN = ROOT / "models" / "qwen3.5-9b-q4s.orig.bin"
MODEL_IDX = ROOT / "chat" / "assets_9b" / "model.idx"


def to_q4s(q, d):
    rows, cols = q.shape
    T, nb = rows // 128, cols // 32
    q = q.astype(np.uint32).reshape(T, 2, 64, nb, 8, 4)
    half = q[..., 0] | (q[..., 1] << 4) | (q[..., 2] << 8) | (q[..., 3] << 12)
    words = half[:, 0] | (half[:, 1] << 16)
    words = words.reshape(T, 64, nb, 2, 4).transpose(0, 2, 3, 1, 4)
    d = d.astype(np.float16).view(np.uint16).astype(np.uint32).reshape(T, 2, 64, nb)
    sc = (d[:, 0] | (d[:, 1] << 16)).transpose(0, 2, 1)
    return np.ascontiguousarray(words).tobytes(), np.ascontiguousarray(sc).tobytes()


def q4_of(w):
    packed, d16 = quantize_q4(w)
    rows = w.shape[0]
    p = packed.reshape(rows, -1, 16)
    return np.concatenate([p & 15, p >> 4], axis=2).reshape(rows, -1), d16


def parse_idx():
    d = MODEL_IDX.read_bytes()
    hdr = struct.unpack("<4s10I2f128sQI", d[:192])
    nl, n_rec = hdr[1], hdr[15]
    ext = struct.unpack("<8I", d[192:224])
    mask = ext[5]
    order = ["embed", "final_norm"]
    for i in range(nl):
        full = (mask >> i) & 1
        mixer = ("q_norm", "k_norm", "wqkv", "wo") if full else ("w_in", "conv", "A", "dt_bias", "gnorm", "wout")
        order += [f"{i}.{k}" for k in ("attn_norm", "mlp_norm", *mixer, "wgu", "wdown")]
    order += ["lm_head", "ids"]
    assert len(order) == n_rec
    recs = {}
    for idx, name in enumerate(order):
        recs[name] = struct.unpack("<QQIIQI", d[224 + idx * 36 : 224 + (idx + 1) * 36])
    return nl, mask, recs


_headers = {}


def fetch_bf16(wmap, name):
    shard = wmap[name]
    base_url = f"https://huggingface.co/{REPO}/resolve/main/{shard}"
    if shard not in _headers:
        b = urllib.request.urlopen(urllib.request.Request(base_url, headers={"Range": "bytes=0-7"})).read()
        n = struct.unpack("<Q", b)[0]
        h = urllib.request.urlopen(urllib.request.Request(base_url, headers={"Range": f"bytes=8-{8 + n - 1}"})).read()
        _headers[shard] = (8 + n, json.loads(h))
    data_start, h = _headers[shard]
    e = h[name]
    a, b = e["data_offsets"]
    for attempt in range(5):
        try:
            req = urllib.request.Request(base_url, headers={"Range": f"bytes={data_start + a}-{data_start + b - 1}"})
            raw = urllib.request.urlopen(req, timeout=600).read()
            if len(raw) == b - a:
                break
        except Exception as ex:
            print(f"  retry {name}: {ex}", flush=True)
    u16 = np.frombuffer(raw, np.uint16).astype(np.uint32) << 16
    return u16.view(np.float32).reshape(e["shape"])


def main():
    if not MODEL_BIN.exists():
        sys.exit(f"Missing {MODEL_BIN}")
    if not BACKUP_BIN.exists() and "--no-backup" not in sys.argv:
        print(f"Creating backup copy at {BACKUP_BIN.name} (pass --no-backup to skip)...", flush=True)
        shutil.copy2(MODEL_BIN, BACKUP_BIN)

    idx_url = f"https://huggingface.co/{REPO}/resolve/main/model.safetensors.index.json"
    wmap = json.loads(urllib.request.urlopen(idx_url).read())["weight_map"]
    nl, mask, recs = parse_idx()

    with open(MODEL_BIN, "r+b") as f:
        for i in range(1, nl):
            t0 = time.time()
            full = (mask >> i) & 1
            attn_key = f"{i}.wo" if full else f"{i}.wout"
            hf_attn = (
                f"model.language_model.layers.{i}.self_attn.o_proj.weight"
                if full
                else f"model.language_model.layers.{i}.linear_attn.out_proj.weight"
            )
            hf_down = f"model.language_model.layers.{i}.mlp.down_proj.weight"

            for key, hf_name in ((attn_key, hf_attn), (f"{i}.wdown", hf_down)):
                off, sz, rows, cols, scales, fmt = recs[key]
                assert fmt == 6, (key, fmt)
                w = fetch_bf16(wmap, hf_name)
                wb, sb = to_q4s(*q4_of(w))
                assert len(wb) == scales and len(wb) + len(sb) == sz, (key, len(wb), len(sb), sz)
                f.seek(off)
                f.write(wb)
                f.write(sb)
            print(f"layer {i:2d}/31 patched ({attn_key}, {i}.wdown) in {time.time() - t0:.1f}s", flush=True)

    print(f"\nDone! {MODEL_BIN} is now patched with {REPO}.")
    print("Upload to PS4 with: python pc/upload_model.py --ip <PS4_IP> models/qwen3.5-9b-q4s.bin")


if __name__ == "__main__":
    main()
