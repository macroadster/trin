# Starlight CNN Inference (Trin v1.19)

Path A: run **BalancedStarlightDetector** from a GGUF weight session inside Trin
(and expose a public Go package for Stargate).

## Architecture contract

Matches Project Starlight `trainer.py` / `BalancedStarlightDetector` **eval** path:

| Item | Contract |
|------|----------|
| Batch | 1 |
| Spatial | `H×W` (fixture `32`; production typically `256`) |
| Pixel | `3·H·W` NCHW flat |
| Meta | `2048` |
| Alpha | `2·H·W` |
| LSB | `3·H·W` |
| Palette | `768` |
| Palette LSB | `1·H·W` |
| Format / content features | `6` each |
| BN eps | `1e-5` (PyTorch default) |
| Dropout | **identity** at eval |
| `meta_weight` | default **0.3** (if `0` is passed, runtime uses `0.3`) |
| Outputs | stego `[1]`, method `[5]`, bit_order `[3]`, embedding `[64]` |
| Concat | `73` floats: stego‖method‖bit_order‖embedding |

Stacks: im2col `Conv2d` + BatchNorm + ReLU, adaptive avg pool → 64-d stream
projections, fusion MLP, four linear heads.

## How to run

```bash
# Unit tests (CNN ops + fixture session forward)
go test ./internal/weights/ -run 'Starlight|Im2col|Conv2d|BatchNorm|ReLU|Linear|Adaptive' -count=1

# Full weights package
go test ./internal/weights/ -count=1

# Build CLI
go build -o trin ./cmd/trin

# Default: fixture GGUF + CSV parity (H=W=32)
./trin run examples/starlight_infer.trin

# Production GGUF (2a head smoke + demo@32; full 256 only if STARLIGHT_FULL=1)
STARLIGHT_GGUF=/path/to/starlight.gguf ./trin run examples/starlight_infer.trin
STARLIGHT_GGUF=/path/to/starlight.gguf STARLIGHT_FULL=1 ./trin run examples/starlight_infer.trin
```

Env resolution for the example (and recommended for hosts):

1. `STARLIGHT_GGUF` if non-empty  
2. else `STARLIGHT_TRIN_MODEL` if non-empty  
3. else `demos/weights/starlight_fixture.gguf`

### Trin builtins (v1.19)

```
getenv(key) → str
load_f32_csv(path) → span[f32]   // skips # comments
starlight_forward(id, pixel, meta, alpha, lsb, palette, palette_lsb,
                  format, content, H, W [, meta_weight]) → span[f32]  // 73
starlight_demo_forward(id, H, W, seed [, meta_weight]) → span[f32]   // LCG inputs
relu_f32(x)  sigmoid_f32(x)  linear_bias(W, x, b, M, K)
```

Also uses existing `open_weight_session` / `weight_session_*` / `close_weight_session`.

## Fixture generation

```bash
go run scripts/gen_starlight_fixture_gguf.go
# writes demos/weights/starlight_fixture.gguf
#      demos/weights/starlight_inputs/*.csv
#      demos/weights/starlight_expected.csv
# seed 42 weights, seed 99 inputs, H=W=32, meta_weight=0.3
```

## Parity (Go vs PyTorch)

```bash
python3 scripts/gen_starlight_parity.py
# threshold 1e-4 on concat outputs (requires torch)
```

Notes:

- Fixture path is the golden H=32 parity surface (Go ↔ PyTorch reference in the script).
- Production `starlight.gguf` may be random-init or trained; use it for **shape / smoke**
  (`demo@32` always; `demo@256` gated by `STARLIGHT_FULL=1`).
- Trin expected CSV is produced by the Go fixture generator; Python script re-implements
  the same LCG and compares its PyTorch module to that contract.

## emit-go / Stargate path

### Preferred: public package (no `internal` import)

```go
import "github.com/ericchien/trin/pkg/starlight"

sess, err := starlight.Open(ggufPath)
if err != nil { return err }
defer sess.Close()

out, err := sess.Forward(starlight.Inputs{
    Pixel: u.Pixel, Meta: u.Meta, Alpha: u.Alpha,
    LSB: u.LSB, Palette: u.Palette, PaletteLSB: u.PaletteLSB,
    FormatFeatures: u.FormatFeatures, ContentFeatures: u.ContentFeatures,
    H: u.H, W: u.W,
    MetaWeight: 0.3, // 0 ⇒ 0.3
})
// out.Stego, out.Method, out.BitOrder, out.Embedding
// out.Concat() → []float32 length 73
```

**Replace Stargate `ForwardStarlight` body** with the above. Do **not** import
`github.com/ericchien/trin/internal/*` from Stargate.

Mapping `UnifiedInput` → `starlight.Inputs` (typical field names):

| UnifiedInput | starlight.Inputs |
|--------------|------------------|
| Pixel / PixelFlat | Pixel |
| Meta / MetaFlat | Meta |
| Alpha / AlphaFlat | Alpha |
| LSB / LSBFlat | LSB |
| Palette / PaletteFlat | Palette |
| PaletteLSB | PaletteLSB |
| FormatFeatures | FormatFeatures |
| ContentFeatures | ContentFeatures |
| H, W | H, W |

Keep `MinimalGGUF` open for other tensors if needed, or switch the Starlight path
entirely to `starlight.Open`.

### Alternative: `trin emit-go`

```bash
trin emit-go examples/starlight_infer.trin
```

Produces a standalone Go main useful for demos. **Note:** the full CNN lives in the
Go runtime natives / `pkg/starlight`, not as pure Trin IR ops — emit-go demos still
depend on the weight-session / Starlight natives when run via the interpreter path.

## Acceptance checklist

- [x] load `starlight.gguf` via weight session  
- [x] linear heads + fusion path (`StarlightForward`)  
- [x] substantial CNN path (im2col conv stacks)  
- [x] `pkg/starlight` documented for Stargate  
- [x] parity notes + Python script  
- [x] `examples/starlight_infer.trin` works on fixture  
