package weights

import (
	"math"
	"path/filepath"
	"testing"
)

func maxAbsDiffCNN(a, b []float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var m float32
	for i := 0; i < n; i++ {
		d := a[i] - b[i]
		if d < 0 {
			d = -d
		}
		if d > m {
			m = d
		}
	}
	return m
}

func TestReLUSigmoid(t *testing.T) {
	x := []float32{-2, -0.5, 0, 0.5, 2}
	r := ReLU(x)
	wantR := []float32{0, 0, 0, 0.5, 2}
	if maxAbsDiffCNN(r, wantR) > 1e-6 {
		t.Fatalf("ReLU got %v want %v", r, wantR)
	}
	s := Sigmoid([]float32{0})
	if math.Abs(float64(s[0]-0.5)) > 1e-6 {
		t.Fatalf("Sigmoid(0)=%v", s[0])
	}
}

func TestLinearBias(t *testing.T) {
	// W [2,3] = [[1,0,0],[0,1,1]], x=[1,2,3], b=[0.5,-0.5]
	W := []float32{1, 0, 0, 0, 1, 1}
	x := []float32{1, 2, 3}
	b := []float32{0.5, -0.5}
	y, err := LinearBias(W, x, b, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	// y0 = 1+0.5=1.5; y1 = 2+3-0.5=4.5
	if math.Abs(float64(y[0]-1.5)) > 1e-5 || math.Abs(float64(y[1]-4.5)) > 1e-5 {
		t.Fatalf("got %v", y)
	}
}

func TestIm2col2dAndConv2d(t *testing.T) {
	// C=1 H=3 W=3, k=2 s=1 p=0 → out 2x2
	// x = [[1,2,3],[4,5,6],[7,8,9]]
	x := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}
	cols, oh, ow, err := Im2col2d(x, 1, 3, 3, 2, 2, 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if oh != 2 || ow != 2 {
		t.Fatalf("out %dx%d", oh, ow)
	}
	// row0 (top-left): 1,2,4,5
	want0 := []float32{1, 2, 4, 5}
	if maxAbsDiffCNN(cols[0:4], want0) > 1e-6 {
		t.Fatalf("col0 %v want %v", cols[0:4], want0)
	}
	// identity-ish conv: single filter all ones → sum of each window
	w := []float32{1, 1, 1, 1}
	y, oh2, ow2, err := Conv2dNCHW(x, w, nil, 1, 3, 3, 1, 2, 2, 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if oh2 != 2 || ow2 != 2 {
		t.Fatalf("conv out %dx%d", oh2, ow2)
	}
	// sums: 1+2+4+5=12, 2+3+5+6=16, 4+5+7+8=24, 5+6+8+9=28
	wantY := []float32{12, 16, 24, 28}
	if maxAbsDiffCNN(y, wantY) > 1e-5 {
		t.Fatalf("conv y=%v want %v", y, wantY)
	}
}

func TestConv2dWithBiasAndPad(t *testing.T) {
	// C=1 H=2 W=2, k=3 s=1 p=1 → out 2x2 (same)
	x := []float32{1, 2, 3, 4}
	// center weight 1, rest 0 → identity with pad
	w := []float32{0, 0, 0, 0, 1, 0, 0, 0, 0}
	b := []float32{0.5}
	y, oh, ow, err := Conv2dNCHW(x, w, b, 1, 2, 2, 1, 3, 3, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if oh != 2 || ow != 2 {
		t.Fatalf("%dx%d", oh, ow)
	}
	want := []float32{1.5, 2.5, 3.5, 4.5}
	if maxAbsDiffCNN(y, want) > 1e-5 {
		t.Fatalf("got %v want %v", y, want)
	}
}

func TestConv1d(t *testing.T) {
	// C=1 L=5 k=3 s=2 p=1 → outL = (5+2-3)/2+1 = 3
	x := []float32{1, 2, 3, 4, 5}
	w := []float32{1, 0, -1} // Cout=1 Cin=1 K=3
	y, outL, err := Conv1d(x, w, []float32{0}, 1, 5, 1, 3, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	if outL != 3 {
		t.Fatalf("outL=%d", outL)
	}
	// positions: ol=0 il center at -1+1=0.. pad: [0,1,2] → 0*1+1*0+2*(-1)=-2
	// ol=1: [2,3,4] → 2-4=-2
	// ol=2: [4,5,0] → 4-0=4
	want := []float32{-2, -2, 4}
	if maxAbsDiffCNN(y, want) > 1e-5 {
		t.Fatalf("got %v want %v", y, want)
	}
}

func TestBatchNormAndFuse(t *testing.T) {
	// C=2 H=1 W=2
	x := []float32{1, 3, 2, 4} // c0:[1,3] c1:[2,4]
	mean := []float32{2, 3}
	variance := []float32{1, 1} // sqrt=1
	gamma := []float32{2, 0.5}
	beta := []float32{0, 1}
	y, err := BatchNorm2dInfer(x, mean, variance, gamma, beta, 2, 1, 2, 0)
	if err != nil {
		t.Fatal(err)
	}
	// c0: scale=2, (1-2)*2+0=-2, (3-2)*2=2
	// c1: scale=0.5, (2-3)*0.5+1=0.5, (4-3)*0.5+1=1.5
	want := []float32{-2, 2, 0.5, 1.5}
	if maxAbsDiffCNN(y, want) > 1e-5 {
		t.Fatalf("bn %v want %v", y, want)
	}

	// Fuse identity conv + BN should match BN of conv
	// Cout=1 Cin=1 k=1: w=[2], b=[1], mean=0, var=1, gamma=3, beta=-1
	// fused scale = 3/1=3; w2=6; b2=3*(1-0)+(-1)=2
	w2, b2, err := FuseConvBN2d([]float32{2}, []float32{1}, []float32{0}, []float32{1}, []float32{3}, []float32{-1}, 1, 1, 1, 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(w2[0]-6)) > 1e-5 || math.Abs(float64(b2[0]-2)) > 1e-5 {
		t.Fatalf("fuse w=%v b=%v", w2, b2)
	}
}

func TestAdaptiveAvgPool2d(t *testing.T) {
	// C=1 H=4 W=4 → 2x2, each bin 2x2
	x := make([]float32, 16)
	for i := range x {
		x[i] = float32(i + 1)
	}
	y, err := AdaptiveAvgPool2d(x, 1, 4, 4, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	// TL: 1,2,5,6 avg=3.5; TR: 3,4,7,8 avg=5.5; BL: 9,10,13,14 avg=11.5; BR: 11,12,15,16 avg=13.5
	want := []float32{3.5, 5.5, 11.5, 13.5}
	if maxAbsDiffCNN(y, want) > 1e-5 {
		t.Fatalf("got %v want %v", y, want)
	}
}

func TestAdaptiveAvgPool1d(t *testing.T) {
	x := []float32{1, 2, 3, 4, 5, 6} // C=1 L=6 → outL=3
	y, err := AdaptiveAvgPool1d(x, 1, 6, 3)
	if err != nil {
		t.Fatal(err)
	}
	// bins [0:2]=1.5, [2:4]=3.5, [4:6]=5.5
	want := []float32{1.5, 3.5, 5.5}
	if maxAbsDiffCNN(y, want) > 1e-5 {
		t.Fatalf("got %v want %v", y, want)
	}
}

// Deterministic LCG matching fixture generator.



func TestStarlightForwardTinySpatial(t *testing.T) {
	// Full architecture but H=W=32 for speed; still uses AdaptiveAvgPool→4 so projs match.
	const H, W = 32, 32
	tensors, _ := GenStarlightFixtureTensors(42)
	tm := mapTensorMap(tensors)

	g := &StarlightLCG{S: 99}
	in := StarlightInputs{
		Pixel:           g.Fill( 3*H*W),
		Meta:            g.Fill( 2048),
		Alpha:           g.Fill( 2*H*W),
		LSB:             g.Fill( 3*H*W),
		Palette:         g.Fill( 768),
		PaletteLSB:      g.Fill( 1*H*W),
		FormatFeatures:  g.Fill( 6),
		ContentFeatures: g.Fill( 6),
		H:               H,
		W:               W,
		MetaWeight:      0.3,
	}
	out, err := StarlightForward(tm, in)
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Stego) != 1 || len(out.Method) != 5 || len(out.BitOrder) != 3 || len(out.Embedding) != 64 {
		t.Fatalf("bad output lens stego=%d method=%d bit=%d emb=%d",
			len(out.Stego), len(out.Method), len(out.BitOrder), len(out.Embedding))
	}
	// Determinism: re-run should match
	out2, err := StarlightForward(tm, in)
	if err != nil {
		t.Fatal(err)
	}
	if maxAbsDiffCNN(out.ConcatOutputs(), out2.ConcatOutputs()) > 0 {
		t.Fatal("non-deterministic")
	}
	// Finite values
	for i, v := range out.ConcatOutputs() {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("non-finite at %d: %v", i, v)
		}
	}
	t.Logf("stego=%v method=%v bit=%v", out.Stego, out.Method, out.BitOrder)
}

func TestStarlightForwardSessionGGUF(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "starlight_fixture.gguf")
	tensors, shapes := GenStarlightFixtureTensors(42)
	meta := map[string]string{
		"general.name":         "starlight-fixture",
		"general.architecture": "starlight",
		"trin.meta_weight":     "0.3",
		"trin.input_h":         "32",
		"trin.input_w":         "32",
	}
	if err := WriteGGUFF32(path, tensors, shapes, meta); err != nil {
		t.Fatal(err)
	}
	s, err := OpenSession(path)
	if err != nil {
		t.Fatal(err)
	}
	defer CloseSession(s.ID)
	names, err := s.List()
	if err != nil {
		t.Fatal(err)
	}
	if len(names) < 50 {
		t.Fatalf("expected many tensors, got %d", len(names))
	}
	const H, W = 32, 32
	g := &StarlightLCG{S: 99}
	in := StarlightInputs{
		Pixel: g.Fill( 3*H*W), Meta: g.Fill( 2048), Alpha: g.Fill( 2*H*W),
		LSB: g.Fill( 3*H*W), Palette: g.Fill( 768), PaletteLSB: g.Fill( 1*H*W),
		FormatFeatures: g.Fill( 6), ContentFeatures: g.Fill( 6),
		H: H, W: W, MetaWeight: 0.3,
	}
	out, err := StarlightForwardSession(s, in)
	if err != nil {
		t.Fatal(err)
	}
	// Compare vs map path
	out2, err := StarlightForward(mapTensorMap(tensors), in)
	if err != nil {
		t.Fatal(err)
	}
	d := maxAbsDiffCNN(out.ConcatOutputs(), out2.ConcatOutputs())
	if d > 1e-5 {
		t.Fatalf("session vs map max abs diff %v", d)
	}
	t.Logf("session parity maxAbs=%g stego=%v", d, out.Stego)
}

func TestFuseConvBNMatchesSeparate(t *testing.T) {
	// Random small conv+bn vs fused
	g := &StarlightLCG{S: 7}
	Cin, Cout, H, W, k := 2, 3, 8, 8, 3
	x := g.Fill( Cin*H*W)
	w := g.Fill( Cout*Cin*k*k)
	b := g.Fill( Cout)
	mean := g.Fill( Cout)
	variance := g.Fill( Cout)
	for i := range variance {
		variance[i] = float32(math.Abs(float64(variance[i]))) + 0.1
	}
	gamma := g.Fill( Cout)
	beta := g.Fill( Cout)
	// separate
	y1, oh, ow, err := Conv2dNCHW(x, w, b, Cin, H, W, Cout, k, k, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	y1, err = BatchNorm2dInfer(y1, mean, variance, gamma, beta, Cout, oh, ow, 1e-5)
	if err != nil {
		t.Fatal(err)
	}
	// fused
	fw, fb, err := FuseConvBN2d(w, b, mean, variance, gamma, beta, Cout, Cin, k, k, 1e-5)
	if err != nil {
		t.Fatal(err)
	}
	y2, _, _, err := Conv2dNCHW(x, fw, fb, Cin, H, W, Cout, k, k, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	d := maxAbsDiffCNN(y1, y2)
	if d > 1e-4 {
		t.Fatalf("fuse mismatch maxAbs=%g", d)
	}
}
