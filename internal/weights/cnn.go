package weights

import (
	"fmt"
	"math"
)

// ---- Elementwise activations (eval / inference) ----

// ReLU applies max(0, x) in-place-safe (returns new slice).
func ReLU(x []float32) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		if v > 0 {
			out[i] = v
		}
	}
	return out
}

// ReLUInPlace clamps negatives to zero.
func ReLUInPlace(x []float32) {
	for i, v := range x {
		if v < 0 {
			x[i] = 0
		}
	}
}

// Sigmoid applies 1/(1+exp(-x)) elementwise.
func Sigmoid(x []float32) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = float32(1.0 / (1.0 + math.Exp(float64(-v))))
	}
	return out
}

// ---- Linear ----

// LinearBias computes y = W @ x + b with W row-major MxK.
// b may be nil (treated as zero).
func LinearBias(W, x, b []float32, M, K int) ([]float32, error) {
	if M <= 0 || K <= 0 {
		return nil, fmt.Errorf("LinearBias: bad M/K")
	}
	if len(W) < M*K {
		return nil, fmt.Errorf("LinearBias: W short need %d have %d", M*K, len(W))
	}
	if len(x) < K {
		return nil, fmt.Errorf("LinearBias: x short need %d have %d", K, len(x))
	}
	out := make([]float32, M)
	for i := 0; i < M; i++ {
		acc := float32(0)
		base := i * K
		for t := 0; t < K; t++ {
			acc += W[base+t] * x[t]
		}
		if b != nil && i < len(b) {
			acc += b[i]
		}
		out[i] = acc
	}
	return out, nil
}

// ---- im2col ----

// Im2col2d unfolds NCHW single-batch input x (flat C*H*W) into columns
// of shape [outH*outW, C*kH*kW] (row-major). outH/outW use floor formula
// matching PyTorch Conv2d with given stride and symmetric pad.
func Im2col2d(x []float32, C, H, W, kH, kW, stride, pad int) ([]float32, int, int, error) {
	if C <= 0 || H <= 0 || W <= 0 || kH <= 0 || kW <= 0 || stride <= 0 {
		return nil, 0, 0, fmt.Errorf("Im2col2d: bad dims")
	}
	if len(x) < C*H*W {
		return nil, 0, 0, fmt.Errorf("Im2col2d: x short")
	}
	outH := (H+2*pad-kH)/stride + 1
	outW := (W+2*pad-kW)/stride + 1
	if outH <= 0 || outW <= 0 {
		return nil, 0, 0, fmt.Errorf("Im2col2d: non-positive out spatial")
	}
	colDim := C * kH * kW
	cols := make([]float32, outH*outW*colDim)
	for oh := 0; oh < outH; oh++ {
		for ow := 0; ow < outW; ow++ {
			row := oh*outW + ow
			base := row * colDim
			col := 0
			for c := 0; c < C; c++ {
				for kh := 0; kh < kH; kh++ {
					ih := oh*stride - pad + kh
					for kw := 0; kw < kW; kw++ {
						iw := ow*stride - pad + kw
						var v float32
						if ih >= 0 && ih < H && iw >= 0 && iw < W {
							v = x[c*H*W+ih*W+iw]
						}
						cols[base+col] = v
						col++
					}
				}
			}
		}
	}
	return cols, outH, outW, nil
}

// Im2col1d unfolds NCW (C,L) input into [outL, C*k] columns.
func Im2col1d(x []float32, C, L, k, stride, pad int) ([]float32, int, error) {
	if C <= 0 || L <= 0 || k <= 0 || stride <= 0 {
		return nil, 0, fmt.Errorf("Im2col1d: bad dims")
	}
	if len(x) < C*L {
		return nil, 0, fmt.Errorf("Im2col1d: x short")
	}
	outL := (L+2*pad-k)/stride + 1
	if outL <= 0 {
		return nil, 0, fmt.Errorf("Im2col1d: non-positive outL")
	}
	colDim := C * k
	cols := make([]float32, outL*colDim)
	for ol := 0; ol < outL; ol++ {
		base := ol * colDim
		col := 0
		for c := 0; c < C; c++ {
			for ki := 0; ki < k; ki++ {
				il := ol*stride - pad + ki
				var v float32
				if il >= 0 && il < L {
					v = x[c*L+il]
				}
				cols[base+col] = v
				col++
			}
		}
	}
	return cols, outL, nil
}

// ---- Convolutions via im2col + matmul ----

// Conv2dNCHW: weight OIHW flat Cout*Cin*kH*kW, bias optional length Cout.
// Returns y flat Cout*outH*outW.
func Conv2dNCHW(x, weight, bias []float32, Cin, H, W, Cout, kH, kW, stride, pad int) ([]float32, int, int, error) {
	cols, outH, outW, err := Im2col2d(x, Cin, H, W, kH, kW, stride, pad)
	if err != nil {
		return nil, 0, 0, err
	}
	colDim := Cin * kH * kW
	if len(weight) < Cout*colDim {
		return nil, 0, 0, fmt.Errorf("Conv2dNCHW: weight short")
	}
	nPos := outH * outW
	y := make([]float32, Cout*nPos)
	// y[c, pos] = weight[c,:] · cols[pos,:]
	for c := 0; c < Cout; c++ {
		wBase := c * colDim
		for p := 0; p < nPos; p++ {
			acc := float32(0)
			cBase := p * colDim
			for t := 0; t < colDim; t++ {
				acc += weight[wBase+t] * cols[cBase+t]
			}
			if bias != nil && c < len(bias) {
				acc += bias[c]
			}
			y[c*nPos+p] = acc
		}
	}
	return y, outH, outW, nil
}

// Conv1d: weight [Cout, Cin, K] flat Cout*Cin*K, bias optional.
// x flat Cin*L; returns y flat Cout*outL.
func Conv1d(x, weight, bias []float32, Cin, L, Cout, k, stride, pad int) ([]float32, int, error) {
	cols, outL, err := Im2col1d(x, Cin, L, k, stride, pad)
	if err != nil {
		return nil, 0, err
	}
	colDim := Cin * k
	if len(weight) < Cout*colDim {
		return nil, 0, fmt.Errorf("Conv1d: weight short")
	}
	y := make([]float32, Cout*outL)
	for c := 0; c < Cout; c++ {
		wBase := c * colDim
		for p := 0; p < outL; p++ {
			acc := float32(0)
			cBase := p * colDim
			for t := 0; t < colDim; t++ {
				acc += weight[wBase+t] * cols[cBase+t]
			}
			if bias != nil && c < len(bias) {
				acc += bias[c]
			}
			y[c*outL+p] = acc
		}
	}
	return y, outL, nil
}

// ---- BatchNorm inference ----

// BatchNorm2dInfer applies per-channel affine BN on NCHW flat (C,H,W).
func BatchNorm2dInfer(x, mean, variance, gamma, beta []float32, C, H, W int, eps float32) ([]float32, error) {
	if len(x) < C*H*W {
		return nil, fmt.Errorf("BatchNorm2dInfer: x short")
	}
	if len(mean) < C || len(variance) < C {
		return nil, fmt.Errorf("BatchNorm2dInfer: mean/var short")
	}
	out := make([]float32, C*H*W)
	n := H * W
	for c := 0; c < C; c++ {
		inv := float32(1.0 / math.Sqrt(float64(variance[c]+eps)))
		g := float32(1)
		if gamma != nil && c < len(gamma) {
			g = gamma[c]
		}
		b := float32(0)
		if beta != nil && c < len(beta) {
			b = beta[c]
		}
		scale := g * inv
		shift := b - mean[c]*scale
		base := c * n
		for i := 0; i < n; i++ {
			out[base+i] = x[base+i]*scale + shift
		}
	}
	return out, nil
}

// BatchNorm1dInfer applies per-channel BN on (C,L) flat.
func BatchNorm1dInfer(x, mean, variance, gamma, beta []float32, C, L int, eps float32) ([]float32, error) {
	if len(x) < C*L {
		return nil, fmt.Errorf("BatchNorm1dInfer: x short")
	}
	if len(mean) < C || len(variance) < C {
		return nil, fmt.Errorf("BatchNorm1dInfer: mean/var short")
	}
	out := make([]float32, C*L)
	for c := 0; c < C; c++ {
		inv := float32(1.0 / math.Sqrt(float64(variance[c]+eps)))
		g := float32(1)
		if gamma != nil && c < len(gamma) {
			g = gamma[c]
		}
		b := float32(0)
		if beta != nil && c < len(beta) {
			b = beta[c]
		}
		scale := g * inv
		shift := b - mean[c]*scale
		base := c * L
		for i := 0; i < L; i++ {
			out[base+i] = x[base+i]*scale + shift
		}
	}
	return out, nil
}

// FuseConvBN2d fuses Conv2d + BatchNorm2d into a single conv weight/bias.
// w is OIHW flat Cout*Cin*kH*kW; b may be nil.
func FuseConvBN2d(w, b, mean, variance, gamma, beta []float32, Cout, Cin, kH, kW int, eps float32) (w2, b2 []float32, err error) {
	k := Cin * kH * kW
	if len(w) < Cout*k {
		return nil, nil, fmt.Errorf("FuseConvBN2d: w short")
	}
	if len(mean) < Cout || len(variance) < Cout {
		return nil, nil, fmt.Errorf("FuseConvBN2d: mean/var short")
	}
	w2 = make([]float32, Cout*k)
	b2 = make([]float32, Cout)
	for c := 0; c < Cout; c++ {
		inv := float32(1.0 / math.Sqrt(float64(variance[c]+eps)))
		g := float32(1)
		if gamma != nil && c < len(gamma) {
			g = gamma[c]
		}
		bet := float32(0)
		if beta != nil && c < len(beta) {
			bet = beta[c]
		}
		scale := g * inv
		bc := float32(0)
		if b != nil && c < len(b) {
			bc = b[c]
		}
		base := c * k
		for t := 0; t < k; t++ {
			w2[base+t] = w[base+t] * scale
		}
		b2[c] = scale*(bc-mean[c]) + bet
	}
	return w2, b2, nil
}

// FuseConvBN1d fuses Conv1d + BatchNorm1d.
func FuseConvBN1d(w, b, mean, variance, gamma, beta []float32, Cout, Cin, k int, eps float32) (w2, b2 []float32, err error) {
	return FuseConvBN2d(w, b, mean, variance, gamma, beta, Cout, Cin, k, 1, eps)
}

// ---- Adaptive average pooling ----

// AdaptiveAvgPool2d pools NCHW (C,H,W) → (C,outH,outW) matching PyTorch.
func AdaptiveAvgPool2d(x []float32, C, H, W, outH, outW int) ([]float32, error) {
	if C <= 0 || H <= 0 || W <= 0 || outH <= 0 || outW <= 0 {
		return nil, fmt.Errorf("AdaptiveAvgPool2d: bad dims")
	}
	if len(x) < C*H*W {
		return nil, fmt.Errorf("AdaptiveAvgPool2d: x short")
	}
	out := make([]float32, C*outH*outW)
	for c := 0; c < C; c++ {
		srcBase := c * H * W
		dstBase := c * outH * outW
		for oh := 0; oh < outH; oh++ {
			hStart := (oh * H) / outH
			hEnd := ((oh + 1) * H) / outH
			if hEnd <= hStart {
				hEnd = hStart + 1
			}
			for ow := 0; ow < outW; ow++ {
				wStart := (ow * W) / outW
				wEnd := ((ow + 1) * W) / outW
				if wEnd <= wStart {
					wEnd = wStart + 1
				}
				sum := float32(0)
				cnt := 0
				for ih := hStart; ih < hEnd; ih++ {
					for iw := wStart; iw < wEnd; iw++ {
						sum += x[srcBase+ih*W+iw]
						cnt++
					}
				}
				if cnt > 0 {
					out[dstBase+oh*outW+ow] = sum / float32(cnt)
				}
			}
		}
	}
	return out, nil
}

// AdaptiveAvgPool1d pools (C,L) → (C,outL).
func AdaptiveAvgPool1d(x []float32, C, L, outL int) ([]float32, error) {
	if C <= 0 || L <= 0 || outL <= 0 {
		return nil, fmt.Errorf("AdaptiveAvgPool1d: bad dims")
	}
	if len(x) < C*L {
		return nil, fmt.Errorf("AdaptiveAvgPool1d: x short")
	}
	out := make([]float32, C*outL)
	for c := 0; c < C; c++ {
		srcBase := c * L
		dstBase := c * outL
		for ol := 0; ol < outL; ol++ {
			start := (ol * L) / outL
			end := ((ol + 1) * L) / outL
			if end <= start {
				end = start + 1
			}
			sum := float32(0)
			cnt := 0
			for i := start; i < end; i++ {
				sum += x[srcBase+i]
				cnt++
			}
			if cnt > 0 {
				out[dstBase+ol] = sum / float32(cnt)
			}
		}
	}
	return out, nil
}

// ---- High-level Starlight stacks ----

const bnEpsDefault = float32(1e-5)

// StarlightConv2dStack runs 3×(Conv2d k=3 s=2 p=1 + BN + ReLU) + AdaptiveAvgPool2d(4).
// Channel progression: inC → 16 → 32 → 64. Weights/BN provided as length-3 slices.
// Returns flat 64*4*4 = 1024 features.
func StarlightConv2dStack(
	x []float32, inC, H, W int,
	w0, b0, mean0, var0, g0, beta0 []float32,
	w1, b1, mean1, var1, g1, beta1 []float32,
	w2, b2, mean2, var2, g2, beta2 []float32,
	eps float32,
) ([]float32, error) {
	if eps <= 0 {
		eps = bnEpsDefault
	}
	// fuse BN into each conv
	fw0, fb0, err := FuseConvBN2d(w0, b0, mean0, var0, g0, beta0, 16, inC, 3, 3, eps)
	if err != nil {
		return nil, err
	}
	fw1, fb1, err := FuseConvBN2d(w1, b1, mean1, var1, g1, beta1, 32, 16, 3, 3, eps)
	if err != nil {
		return nil, err
	}
	fw2, fb2, err := FuseConvBN2d(w2, b2, mean2, var2, g2, beta2, 64, 32, 3, 3, eps)
	if err != nil {
		return nil, err
	}
	y, h, w, err := Conv2dNCHW(x, fw0, fb0, inC, H, W, 16, 3, 3, 2, 1)
	if err != nil {
		return nil, err
	}
	ReLUInPlace(y)
	y, h, w, err = Conv2dNCHW(y, fw1, fb1, 16, h, w, 32, 3, 3, 2, 1)
	if err != nil {
		return nil, err
	}
	ReLUInPlace(y)
	y, h, w, err = Conv2dNCHW(y, fw2, fb2, 32, h, w, 64, 3, 3, 2, 1)
	if err != nil {
		return nil, err
	}
	ReLUInPlace(y)
	return AdaptiveAvgPool2d(y, 64, h, w, 4, 4)
}

// StarlightMetaConvStack: meta (2048) unsqueezed to (1,2048) → 3×Conv1d+BN+ReLU + pool16 → 2048.
func StarlightMetaConvStack(
	meta []float32,
	w0, b0, mean0, var0, g0, beta0 []float32, // Conv1d 1→32 k7 s2 p3
	w1, b1, mean1, var1, g1, beta1 []float32, // 32→64 k5 s2 p2
	w2, b2, mean2, var2, g2, beta2 []float32, // 64→128 k3 s2 p1
	eps float32,
) ([]float32, error) {
	if eps <= 0 {
		eps = bnEpsDefault
	}
	if len(meta) < 2048 {
		return nil, fmt.Errorf("StarlightMetaConvStack: meta short")
	}
	x := meta[:2048]
	fw0, fb0, err := FuseConvBN1d(w0, b0, mean0, var0, g0, beta0, 32, 1, 7, eps)
	if err != nil {
		return nil, err
	}
	fw1, fb1, err := FuseConvBN1d(w1, b1, mean1, var1, g1, beta1, 64, 32, 5, eps)
	if err != nil {
		return nil, err
	}
	fw2, fb2, err := FuseConvBN1d(w2, b2, mean2, var2, g2, beta2, 128, 64, 3, eps)
	if err != nil {
		return nil, err
	}
	y, L, err := Conv1d(x, fw0, fb0, 1, 2048, 32, 7, 2, 3)
	if err != nil {
		return nil, err
	}
	ReLUInPlace(y)
	y, L, err = Conv1d(y, fw1, fb1, 32, L, 64, 5, 2, 2)
	if err != nil {
		return nil, err
	}
	ReLUInPlace(y)
	y, L, err = Conv1d(y, fw2, fb2, 64, L, 128, 3, 2, 1)
	if err != nil {
		return nil, err
	}
	ReLUInPlace(y)
	return AdaptiveAvgPool1d(y, 128, L, 16)
}

// TensorMap is a name→float32 tensor lookup (session or fixture).
type TensorMap interface {
	LoadF32(name string) ([]float32, error)
}

// sessionTensorMap adapts WeightSession to TensorMap.
type sessionTensorMap struct{ s *WeightSession }

func (m sessionTensorMap) LoadF32(name string) ([]float32, error) {
	v, _, err := m.s.LoadF32(name)
	return v, err
}

// mapTensorMap wraps a plain map.
type mapTensorMap map[string][]float32

func (m mapTensorMap) LoadF32(name string) ([]float32, error) {
	v, ok := m[name]
	if !ok {
		return nil, fmt.Errorf("tensor %q not found", name)
	}
	return v, nil
}

func mustLoad(tm TensorMap, name string) ([]float32, error) {
	return tm.LoadF32(name)
}

// StarlightOutputs holds detector head outputs (batch=1).
type StarlightOutputs struct {
	Stego     []float32 // [1]
	Method    []float32 // [5]
	BitOrder  []float32 // [3]
	Embedding []float32 // [64]
	// Intermediate (optional debug)
	ClsFeatures []float32 // [64]
	Fused       []float32 // [192]
}

// StarlightInputs holds NCHW/flat inputs for batch=1.
type StarlightInputs struct {
	Pixel           []float32 // 3*H*W
	Meta            []float32 // 2048
	Alpha           []float32 // 2*H*W
	LSB             []float32 // 3*H*W
	Palette         []float32 // 768
	PaletteLSB      []float32 // 1*H*W
	FormatFeatures  []float32 // 6
	ContentFeatures []float32 // 6
	H, W            int       // spatial (default 256)
	MetaWeight      float32   // default 0.3
}

// StarlightForward runs BalancedStarlightDetector eval path from a tensor map.
func StarlightForward(tm TensorMap, in StarlightInputs) (*StarlightOutputs, error) {
	if tm == nil {
		return nil, fmt.Errorf("StarlightForward: nil tensor map")
	}
	H, W := in.H, in.W
	if H <= 0 {
		H = 256
	}
	if W <= 0 {
		W = 256
	}
	mw := in.MetaWeight
	if mw == 0 {
		mw = 0.3
	}
	eps := bnEpsDefault

	load3 := func(prefix string, idxs []int) (ws, bs, means, vars, gs, betas [][]float32, err error) {
		// idxs are Sequential indices for Conv modules: 0,3,6 and BN: 1,4,7
		ws = make([][]float32, 3)
		bs = make([][]float32, 3)
		means = make([][]float32, 3)
		vars = make([][]float32, 3)
		gs = make([][]float32, 3)
		betas = make([][]float32, 3)
		convIdx := []int{0, 3, 6}
		bnIdx := []int{1, 4, 7}
		for i := 0; i < 3; i++ {
			ci, bi := convIdx[i], bnIdx[i]
			if ws[i], err = mustLoad(tm, fmt.Sprintf("%s.%d.weight", prefix, ci)); err != nil {
				return
			}
			if bs[i], err = mustLoad(tm, fmt.Sprintf("%s.%d.bias", prefix, ci)); err != nil {
				return
			}
			if gs[i], err = mustLoad(tm, fmt.Sprintf("%s.%d.weight", prefix, bi)); err != nil {
				return
			}
			if betas[i], err = mustLoad(tm, fmt.Sprintf("%s.%d.bias", prefix, bi)); err != nil {
				return
			}
			if means[i], err = mustLoad(tm, fmt.Sprintf("%s.%d.running_mean", prefix, bi)); err != nil {
				return
			}
			if vars[i], err = mustLoad(tm, fmt.Sprintf("%s.%d.running_var", prefix, bi)); err != nil {
				return
			}
		}
		return
	}

	// --- pixel stream ---
	pw, pb, pm, pv, pg, pbe, err := load3("pixel_conv", nil)
	if err != nil {
		return nil, fmt.Errorf("pixel_conv: %w", err)
	}
	pixFeat, err := StarlightConv2dStack(in.Pixel, 3, H, W,
		pw[0], pb[0], pm[0], pv[0], pg[0], pbe[0],
		pw[1], pb[1], pm[1], pv[1], pg[1], pbe[1],
		pw[2], pb[2], pm[2], pv[2], pg[2], pbe[2], eps)
	if err != nil {
		return nil, err
	}
	ppW, err := mustLoad(tm, "pixel_proj.weight")
	if err != nil {
		return nil, err
	}
	ppB, err := mustLoad(tm, "pixel_proj.bias")
	if err != nil {
		return nil, err
	}
	pixel, err := LinearBias(ppW, pixFeat, ppB, 64, 1024)
	if err != nil {
		return nil, err
	}

	// --- meta stream ---
	mw0, mb0, mm0, mv0, mg0, mbe0, err := load3("meta_conv", nil)
	if err != nil {
		return nil, fmt.Errorf("meta_conv: %w", err)
	}
	// meta uses different kernels; load3 only loads by Sequential index — shapes differ.
	// StarlightMetaConvStack expects the actual tensors; channel progression is 1→32→64→128.
	metaFeat, err := StarlightMetaConvStack(in.Meta,
		mw0[0], mb0[0], mm0[0], mv0[0], mg0[0], mbe0[0],
		mw0[1], mb0[1], mm0[1], mv0[1], mg0[1], mbe0[1],
		mw0[2], mb0[2], mm0[2], mv0[2], mg0[2], mbe0[2], eps)
	if err != nil {
		return nil, err
	}
	mpW, err := mustLoad(tm, "meta_proj.weight")
	if err != nil {
		return nil, err
	}
	mpB, err := mustLoad(tm, "meta_proj.bias")
	if err != nil {
		return nil, err
	}
	meta, err := LinearBias(mpW, metaFeat, mpB, 64, 2048)
	if err != nil {
		return nil, err
	}
	for i := range meta {
		meta[i] *= mw
	}

	// --- alpha stream ---
	aw, ab, am, av, ag, abe, err := load3("alpha_conv", nil)
	if err != nil {
		return nil, fmt.Errorf("alpha_conv: %w", err)
	}
	alphaFeat, err := StarlightConv2dStack(in.Alpha, 2, H, W,
		aw[0], ab[0], am[0], av[0], ag[0], abe[0],
		aw[1], ab[1], am[1], av[1], ag[1], abe[1],
		aw[2], ab[2], am[2], av[2], ag[2], abe[2], eps)
	if err != nil {
		return nil, err
	}
	apW, err := mustLoad(tm, "alpha_proj.weight")
	if err != nil {
		return nil, err
	}
	apB, err := mustLoad(tm, "alpha_proj.bias")
	if err != nil {
		return nil, err
	}
	alpha, err := LinearBias(apW, alphaFeat, apB, 64, 1024)
	if err != nil {
		return nil, err
	}

	// --- lsb stream ---
	lw, lb, lm, lv, lg, lbe, err := load3("lsb_conv", nil)
	if err != nil {
		return nil, fmt.Errorf("lsb_conv: %w", err)
	}
	lsbFeat, err := StarlightConv2dStack(in.LSB, 3, H, W,
		lw[0], lb[0], lm[0], lv[0], lg[0], lbe[0],
		lw[1], lb[1], lm[1], lv[1], lg[1], lbe[1],
		lw[2], lb[2], lm[2], lv[2], lg[2], lbe[2], eps)
	if err != nil {
		return nil, err
	}
	lpW, err := mustLoad(tm, "lsb_proj.weight")
	if err != nil {
		return nil, err
	}
	lpB, err := mustLoad(tm, "lsb_proj.bias")
	if err != nil {
		return nil, err
	}
	lsb, err := LinearBias(lpW, lsbFeat, lpB, 64, 1024)
	if err != nil {
		return nil, err
	}

	// --- palette_fc (Dropout skipped at eval) ---
	pfc0w, err := mustLoad(tm, "palette_fc.0.weight")
	if err != nil {
		return nil, err
	}
	pfc0b, err := mustLoad(tm, "palette_fc.0.bias")
	if err != nil {
		return nil, err
	}
	pfc3w, err := mustLoad(tm, "palette_fc.3.weight")
	if err != nil {
		return nil, err
	}
	pfc3b, err := mustLoad(tm, "palette_fc.3.bias")
	if err != nil {
		return nil, err
	}
	ph, err := LinearBias(pfc0w, in.Palette, pfc0b, 128, 768)
	if err != nil {
		return nil, err
	}
	ReLUInPlace(ph)
	palette, err := LinearBias(pfc3w, ph, pfc3b, 64, 128)
	if err != nil {
		return nil, err
	}

	// --- palette_lsb stream ---
	plw, plb, plm, plv, plg, plbe, err := load3("palette_lsb_conv", nil)
	if err != nil {
		return nil, fmt.Errorf("palette_lsb_conv: %w", err)
	}
	plFeat, err := StarlightConv2dStack(in.PaletteLSB, 1, H, W,
		plw[0], plb[0], plm[0], plv[0], plg[0], plbe[0],
		plw[1], plb[1], plm[1], plv[1], plg[1], plbe[1],
		plw[2], plb[2], plm[2], plv[2], plg[2], plbe[2], eps)
	if err != nil {
		return nil, err
	}
	plpW, err := mustLoad(tm, "palette_lsb_proj.weight")
	if err != nil {
		return nil, err
	}
	plpB, err := mustLoad(tm, "palette_lsb_proj.bias")
	if err != nil {
		return nil, err
	}
	paletteLSB, err := LinearBias(plpW, plFeat, plpB, 64, 1024)
	if err != nil {
		return nil, err
	}

	// --- format / content fc ---
	ff := func(prefix string, x []float32) ([]float32, error) {
		w0, err := mustLoad(tm, prefix+".0.weight")
		if err != nil {
			return nil, err
		}
		b0, err := mustLoad(tm, prefix+".0.bias")
		if err != nil {
			return nil, err
		}
		w2, err := mustLoad(tm, prefix+".2.weight")
		if err != nil {
			return nil, err
		}
		b2, err := mustLoad(tm, prefix+".2.bias")
		if err != nil {
			return nil, err
		}
		h, err := LinearBias(w0, x, b0, 32, 6)
		if err != nil {
			return nil, err
		}
		ReLUInPlace(h)
		return LinearBias(w2, h, b2, 32, 32)
	}
	formatF, err := ff("format_features_fc", in.FormatFeatures)
	if err != nil {
		return nil, err
	}
	contentF, err := ff("content_features_fc", in.ContentFeatures)
	if err != nil {
		return nil, err
	}

	// concat all streams → 448
	all := make([]float32, 0, 448)
	all = append(all, pixel...)
	all = append(all, meta...)
	all = append(all, alpha...)
	all = append(all, lsb...)
	all = append(all, palette...)
	all = append(all, paletteLSB...)
	all = append(all, formatF...)
	all = append(all, contentF...)

	// stream_gate
	sgW, err := mustLoad(tm, "stream_gate.0.weight")
	if err != nil {
		return nil, err
	}
	sgB, err := mustLoad(tm, "stream_gate.0.bias")
	if err != nil {
		return nil, err
	}
	gate, err := LinearBias(sgW, all, sgB, 8, 448)
	if err != nil {
		return nil, err
	}
	gate = Sigmoid(gate)

	// apply gate per stream chunk
	streams := [][]float32{pixel, meta, alpha, lsb, palette, paletteLSB, formatF, contentF}
	gated := make([]float32, 0, 448)
	for i, s := range streams {
		g := gate[i]
		for _, v := range s {
			gated = append(gated, v*g)
		}
	}

	// fusion: Linear 448→256 → ReLU → 256→128 → ReLU → 128→192
	// Dropout is identity at eval (indices 2,5)
	f0w, err := mustLoad(tm, "fusion.0.weight")
	if err != nil {
		return nil, err
	}
	f0b, err := mustLoad(tm, "fusion.0.bias")
	if err != nil {
		return nil, err
	}
	f3w, err := mustLoad(tm, "fusion.3.weight")
	if err != nil {
		return nil, err
	}
	f3b, err := mustLoad(tm, "fusion.3.bias")
	if err != nil {
		return nil, err
	}
	f6w, err := mustLoad(tm, "fusion.6.weight")
	if err != nil {
		return nil, err
	}
	f6b, err := mustLoad(tm, "fusion.6.bias")
	if err != nil {
		return nil, err
	}
	h, err := LinearBias(f0w, gated, f0b, 256, 448)
	if err != nil {
		return nil, err
	}
	ReLUInPlace(h)
	h, err = LinearBias(f3w, h, f3b, 128, 256)
	if err != nil {
		return nil, err
	}
	ReLUInPlace(h)
	fused, err := LinearBias(f6w, h, f6b, 192, 128)
	if err != nil {
		return nil, err
	}
	cls := fused[128:192]

	// heads
	loadHead := func(name string, M, K int, x []float32) ([]float32, error) {
		w, err := mustLoad(tm, name+".weight")
		if err != nil {
			return nil, err
		}
		b, err := mustLoad(tm, name+".bias")
		if err != nil {
			return nil, err
		}
		return LinearBias(w, x, b, M, K)
	}
	stego, err := loadHead("stego_head", 1, 64, cls)
	if err != nil {
		return nil, err
	}
	method, err := loadHead("method_head", 5, 64, cls)
	if err != nil {
		return nil, err
	}
	bitOrder, err := loadHead("bit_order_head", 3, 64, cls)
	if err != nil {
		return nil, err
	}
	emb, err := loadHead("embedding_head", 64, 64, cls)
	if err != nil {
		return nil, err
	}

	return &StarlightOutputs{
		Stego:       stego,
		Method:      method,
		BitOrder:    bitOrder,
		Embedding:   emb,
		ClsFeatures: append([]float32(nil), cls...),
		Fused:       fused,
	}, nil
}

// StarlightForwardSession runs StarlightForward on an open weight session.
func StarlightForwardSession(s *WeightSession, in StarlightInputs) (*StarlightOutputs, error) {
	if s == nil {
		return nil, fmt.Errorf("nil session")
	}
	return StarlightForward(sessionTensorMap{s}, in)
}

// ConcatOutputs flattens heads to [stego(1)|method(5)|bit_order(3)|embedding(64)] = 73.
func (o *StarlightOutputs) ConcatOutputs() []float32 {
	out := make([]float32, 0, 73)
	out = append(out, o.Stego...)
	out = append(out, o.Method...)
	out = append(out, o.BitOrder...)
	out = append(out, o.Embedding...)
	return out
}
