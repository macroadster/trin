# Starlight steg assist + tiny trit CNN scaffold

**Relationship to production detector:** Starlight’s production scan model is
**BalancedStarlightDetector** (Trin v1.19) — see `docs/STARLIGHT_INFER.md` and
`pkg/starlight`. This document describes the **sidecar** path only:

1. **v1.20** — classical / trit LSB embed–extract + residual markers  
2. **v1.21** — tiny trit CNN (`tiny_steg_cnn.gguf`) used as a prepack / kernel
   scaffold — **not** a drop-in for the multi-stream detector

**Architecture target (scaffold):** export a small linear-im2col net to GGUF;
run **Trin ternary CPU inference** without a Go→Python proxy on that path.

```
  [train/export offline]          [stargate process — pure Go]
  Starlight CNN (PyTorch/etc.)
        │ export im2col/linear
        ▼
   tiny_steg_cnn.gguf  ──────────►  OpenSession / prepack trit
                                    ForwardStegCNN (L0→SiLU→L1→SiLU→Head)
                                    scan_image path (MCP) calls this
```

## Why not a proxy

| Old pattern | Target |
|-------------|--------|
| Go stargate → HTTP/gRPC → Python torch model | **GGUF weights in-process** |
| Python GIL + venv + torch wheels | single static/linkable Go binary surface |
| re-export on every scan host | ship one `.gguf` artifact with the deploy |

Trin’s job is the **inference kernel**: absmean trit pack, bitplane matvec,
SiLU, residual feature construction. Official wish inscription / PSBT still
uses Starlight MCP tools; only **scan inference** moves fully in-process.

## Export contract (CNN → GGUF)

Conv layers are lowered to row-major **im2col / linear** tensors (what most
export tooling already emits for GGUF):

| Tensor | Shape (out×in) | Notes |
|--------|----------------|-------|
| `L0` | `H0 × K` | first patch/linear bank |
| `L0_b` | `H0` | optional bias |
| `L1` | `H1 × H0` | hidden |
| `L1_b` | `H1` | optional bias |
| `Head` | `2 × H1` | logits: `[clean, stego]` |
| `Head_b` | `2` | optional bias |

Metadata keys (string):

- `general.architecture` = `starlight_steg_cnn`
- `starlight.steg.runtime` = `trin_in_stargate`
- `starlight.steg.no_python` = `1`
- `starlight.steg.in_k` / `hidden0` / `hidden1` / `n_class`
- `starlight.steg.act` = `silu`
- `starlight.steg.export` = `im2col_linear`

Toy fixture generator (stands in for the real CNN export job):

```bash
go run scripts/gen_tiny_steg_cnn_gguf.go .
# → demos/weights/tiny_steg_cnn.gguf
```

Real production export will replace this generator with the Starlight training
pipeline’s GGUF writer; **tensor names + meta keys stay stable**.

## Inference (Trin / stargate)

Go package API (linkable from stargate):

```go
s, _ := weights.OpenSession("path/to/model.gguf")
cfg := steg.CNNConfigFromSession(s)
steg.PrepackCNNTrit(s, cfg)                 // once per process
x := steg.FeaturesFromImage(pixels, cfg.InK)
out, _ := steg.ForwardStegCNN(s, x, cfg, true) // bitplane=true
// out.Decision: "clean" | "stego"; out.Margin = stego - clean
```

Trin language demo (same graph):

```bash
./trin run examples/starlight_steg_cnn.trin
./trin build examples/starlight_steg_cnn.trin -o build/starlight_steg_cnn --backend go
```

Serve CLI (local agent / stargate smoke):

```bash
go run scripts/starlight_steg_serve.go cnn
```

## Efficiency note

Packed trit weights are **~8–16× smaller** than f32 for the filter banks.
On this host pure-Go bitplane accumulate can still lose latency to dense f32
for small mats; the win for stargate is **RAM + no Python dependency +
prepack reuse across scans**. Large M/K production CNNs lean harder into the
memory win.

## Files

| Path | Role |
|------|------|
| `internal/steg/cnn.go` | GGUF CNN forward + feature builder |
| `internal/steg/steg.go` | LSB trit codec + residual markers |
| `demos/weights/tiny_steg_cnn.gguf` | toy export fixture |
| `scripts/gen_tiny_steg_cnn_gguf.go` | fixture writer |
| `scripts/starlight_steg_serve.go` | `demo` / `cnn` / `bench` CLI |
| `examples/starlight_steg_cnn.trin` | language surface e2e |
| `examples/starlight_steg_trit.trin` | residual + linear_trit only |

## Non-goals (current)

- Training the production Starlight CNN (export only)
- Full stargate repo merge (this tree is the inference library)
- Pixel-perfect bit-for-bit match to an old Python model until export lands
- GPU path for scan (same table as rest of Trin CUDA)

## Acceptance

- `go run scripts/gen_tiny_steg_cnn_gguf.go .`
- `go test ./internal/steg -count=1`
- `go run scripts/starlight_steg_serve.go cnn` → `starlight_steg_serve_cnn_ok=1` + `python_proxy=0`
- `./trin run examples/starlight_steg_cnn.trin` → `starlight_steg_cnn_ok` + L0/L1/Head parity
