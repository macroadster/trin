package cuda

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestEmitPTXContainsEntry(t *testing.T) {
	for _, mod := range AllPTXModules("sm_75") {
		if !strings.Contains(mod.Source, ".entry "+mod.Name) {
			t.Fatalf("missing entry in %s", mod.Name)
		}
		if !strings.Contains(mod.Source, ".target sm_75") {
			t.Fatalf("bad target in %s", mod.Name)
		}
		if len(mod.Source) < 200 {
			t.Fatalf("PTX too short for %s", mod.Name)
		}
	}
}

func TestCompilePTXIfAvailable(t *testing.T) {
	if !PTXASAvailable() {
		t.Skip("ptxas not on host")
	}
	for _, mod := range AllPTXModules("") {
		cub, err := CompilePTX(mod)
		if err != nil {
			t.Fatalf("%s: %v\n%s", mod.Name, err, cub.Stderr)
		}
		if len(cub.Bytes) < 64 {
			t.Fatalf("%s cubin too small: %d", mod.Name, len(cub.Bytes))
		}
		t.Logf("%s cubin=%dB arch=%s tool=%s", mod.Name, len(cub.Bytes), cub.Arch, cub.Tool)
	}
}

func TestPlanLaunchNoDevice(t *testing.T) {
	os.Setenv("TRIN_CUDA", "0")
	ResetProbeForTest()
	st := PlanLaunch(PTXLinearF32Entry)
	if st.WouldLaunch {
		t.Fatalf("should not launch without device: %+v", st)
	}
	if st.Backend != BackendCPUFallback {
		t.Fatalf("backend=%s", st.Backend)
	}
	if st.PTXBytes == 0 {
		t.Fatalf("expected PTX bytes")
	}
	// If ptxas present, cubin should assemble
	if PTXASAvailable() && st.CubinBytes == 0 && !strings.Contains(st.Reason, "ptxas compile") {
		t.Fatalf("expected cubin or compile reason: %+v", st)
	}
}

func TestDeviceBufferRoundTrip(t *testing.T) {
	b, err := AllocF32(4)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Free()
	src := []float32{1, -2, 0.5, 3.25}
	if err := b.CopyHtoD(src); err != nil {
		t.Fatal(err)
	}
	dst := make([]float32, 4)
	if err := b.CopyDtoH(dst); err != nil {
		t.Fatal(err)
	}
	for i := range src {
		if dst[i] != src[i] {
			t.Fatalf("dst=%v src=%v", dst, src)
		}
	}
}

func TestWritePTXFile(t *testing.T) {
	dir := t.TempDir()
	mod := EmitLinearF32PTX("sm_75")
	p := filepath.Join(dir, "linear.ptx")
	if err := WritePTXFile(mod, p); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), PTXLinearF32Entry) {
		t.Fatalf("file missing entry")
	}
}
