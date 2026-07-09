package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/ericchien/trin/internal/ast"
	"github.com/ericchien/trin/internal/codegen"
	"github.com/ericchien/trin/internal/interp"
	"github.com/ericchien/trin/internal/ir"
	"github.com/ericchien/trin/internal/parser"
)

const version = "1.15.0-go"

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	switch os.Args[1] {
	case "version", "-v", "--version":
		fmt.Println("trin", version)
	case "run":
		if len(os.Args) < 3 {
			fatal("usage: trin run <file.trin>")
		}
		src, err := os.ReadFile(os.Args[2])
		if err != nil {
			fatal(err.Error())
		}
		if err := interp.RunSource(string(src), os.Stdin, os.Stdout); err != nil {
			fatal(err.Error())
		}
	case "build":
		if len(os.Args) < 3 {
			fatal("usage: trin build <file.trin> [-o out] [--backend go|c|llvm] [--emit-obj] [--link-llvm]")
		}
		in := os.Args[2]
		out := strings.TrimSuffix(filepath.Base(in), filepath.Ext(in))
		backend := defaultBackend()
		emitObject := false
		linkLLVM := false
		llcPath := os.Getenv("TRIN_LLC")
		ccPath := os.Getenv("TRIN_CC")
		entry := os.Getenv("TRIN_LLVM_ENTRY")
		for i := 3; i < len(os.Args); i++ {
			switch {
			case os.Args[i] == "-o" && i+1 < len(os.Args):
				out = os.Args[i+1]
				i++
			case os.Args[i] == "--backend" && i+1 < len(os.Args):
				backend = os.Args[i+1]
				i++
			case strings.HasPrefix(os.Args[i], "--backend="):
				backend = strings.TrimPrefix(os.Args[i], "--backend=")
			case os.Args[i] == "--emit-obj":
				emitObject = true
			case os.Args[i] == "--link-llvm":
				linkLLVM = true
			case os.Args[i] == "--llc" && i+1 < len(os.Args):
				llcPath = os.Args[i+1]
				i++
			case strings.HasPrefix(os.Args[i], "--llc="):
				llcPath = strings.TrimPrefix(os.Args[i], "--llc=")
			case os.Args[i] == "--cc" && i+1 < len(os.Args):
				ccPath = os.Args[i+1]
				i++
			case strings.HasPrefix(os.Args[i], "--cc="):
				ccPath = strings.TrimPrefix(os.Args[i], "--cc=")
			case os.Args[i] == "--entry" && i+1 < len(os.Args):
				entry = os.Args[i+1]
				i++
			case strings.HasPrefix(os.Args[i], "--entry="):
				entry = strings.TrimPrefix(os.Args[i], "--entry=")
			}
		}
		if emitObject {
			if !strings.HasSuffix(out, ".o") {
				out = out + ".o"
			}
			if err := emitObjectFile(in, out, llcPath); err != nil {
				fatal(err.Error())
			}
			fmt.Printf("wrote %s (emit-obj)\n", out)
			return
		}
		if linkLLVM || backend == "llvm" {
			if err := linkLLVMBinary(in, out, llcPath, ccPath, entry); err != nil {
				fatal(err.Error())
			}
			fmt.Printf("wrote %s (backend=llvm)\n", out)
			return
		}
		if err := build(in, out, backend); err != nil {
			fatal(err.Error())
		}
		fmt.Printf("wrote %s (backend=%s)\n", out, backend)
	case "emit-c":
		emitIR(os.Args[2:], "c")
	case "emit-go":
		emitIR(os.Args[2:], "go")
	case "emit-ir":
		emitTrinIR(os.Args[2:])
	case "emit-llvm":
		emitLLVM(os.Args[2:])
	case "emit-mlir":
		emitMLIR(os.Args[2:])
	case "emit-mlir-llvm":
		emitMLIRLLVM(os.Args[2:])
	case "lower-mlir":
		lowerMLIR(os.Args[2:])
	case "rewrite-mlir":
		rewriteMLIR(os.Args[2:])
	case "emit-obj":
		emitObj(os.Args[2:])
	case "link-llvm":
		linkLLVMCmd(os.Args[2:])
	case "help", "-h", "--help":
		usage()
	default:
		if strings.HasSuffix(os.Args[1], ".trin") {
			src, err := os.ReadFile(os.Args[1])
			if err != nil {
				fatal(err.Error())
			}
			if err := interp.RunSource(string(src), os.Stdin, os.Stdout); err != nil {
				fatal(err.Error())
			}
			return
		}
		usage()
		os.Exit(2)
	}
}

func emitIR(args []string, kind string) {
	if len(args) < 1 {
		fatal(fmt.Sprintf("usage: trin emit-%s <file.trin> [-o out]", kind))
	}
	in := args[0]
	ext := "." + kind
	out := strings.TrimSuffix(in, filepath.Ext(in)) + ext
	for i := 1; i < len(args); i++ {
		if args[i] == "-o" && i+1 < len(args) {
			out = args[i+1]
			i++
		}
	}
	src, err := os.ReadFile(in)
	if err != nil {
		fatal(err.Error())
	}
	prog, err := parser.ParseSource(string(src))
	if err != nil {
		fatal(err.Error())
	}
	var code string
	switch kind {
	case "c":
		code, err = codegen.EmitC(prog)
	case "go":
		code, err = codegen.EmitGo(prog)
	}
	if err != nil {
		fatal(err.Error())
	}
	if err := os.WriteFile(out, []byte(code), 0o644); err != nil {
		fatal(err.Error())
	}
	fmt.Printf("wrote %s\n", out)
}

func emitMLIR(args []string) {
	if len(args) < 1 {
		fatal("usage: trin emit-mlir <file.trin> [-o out.mlir]")
	}
	in := args[0]
	out := strings.TrimSuffix(in, filepath.Ext(in)) + ".mlir"
	toStdout := false
	for i := 1; i < len(args); i++ {
		switch {
		case args[i] == "-o" && i+1 < len(args):
			out = args[i+1]
			i++
		case args[i] == "--stdout":
			toStdout = true
		}
	}
	src, err := os.ReadFile(in)
	if err != nil {
		fatal(err.Error())
	}
	prog, err := parser.ParseSource(string(src))
	if err != nil {
		fatal(err.Error())
	}
	text, err := codegen.EmitMLIR(prog)
	if err != nil {
		fatal(err.Error())
	}
	if toStdout || out == "-" {
		fmt.Print(text)
		return
	}
	if err := os.WriteFile(out, []byte(text), 0o644); err != nil {
		fatal(err.Error())
	}
	fmt.Printf("wrote %s\n", out)
}

func emitMLIRLLVM(args []string) {
	if len(args) < 1 {
		fatal("usage: trin emit-mlir-llvm <file.trin> [-o out.ll]")
	}
	in := args[0]
	out := strings.TrimSuffix(in, filepath.Ext(in)) + ".mlir.ll"
	toStdout := false
	for i := 1; i < len(args); i++ {
		switch {
		case args[i] == "-o" && i+1 < len(args):
			out = args[i+1]
			i++
		case args[i] == "--stdout":
			toStdout = true
		}
	}
	src, err := os.ReadFile(in)
	if err != nil {
		fatal(err.Error())
	}
	prog, err := parser.ParseSource(string(src))
	if err != nil {
		fatal(err.Error())
	}
	text, err := codegen.EmitMLIRLLVM(prog)
	if err != nil {
		fatal(err.Error())
	}
	if toStdout || out == "-" {
		fmt.Print(text)
		return
	}
	if err := os.WriteFile(out, []byte(text), 0o644); err != nil {
		fatal(err.Error())
	}
	fmt.Printf("wrote %s\n", out)
}

func lowerMLIR(args []string) {
	if len(args) < 1 {
		fatal("usage: trin lower-mlir <file.mlir> [-o out.ll]")
	}
	in := args[0]
	out := strings.TrimSuffix(in, filepath.Ext(in)) + ".ll"
	toStdout := false
	for i := 1; i < len(args); i++ {
		switch {
		case args[i] == "-o" && i+1 < len(args):
			out = args[i+1]
			i++
		case args[i] == "--stdout":
			toStdout = true
		}
	}
	src, err := os.ReadFile(in)
	if err != nil {
		fatal(err.Error())
	}
	text, err := codegen.LowerMLIRToLLVM(string(src))
	if err != nil {
		fatal(err.Error())
	}
	if toStdout || out == "-" {
		fmt.Print(text)
		return
	}
	if err := os.WriteFile(out, []byte(text), 0o644); err != nil {
		fatal(err.Error())
	}
	fmt.Printf("wrote %s\n", out)
}

func rewriteMLIR(args []string) {
	if len(args) < 1 {
		fatal("usage: trin rewrite-mlir <file.mlir> [-o out.mlir]")
	}
	in := args[0]
	out := strings.TrimSuffix(in, filepath.Ext(in)) + ".rw.mlir"
	toStdout := false
	for i := 1; i < len(args); i++ {
		switch {
		case args[i] == "-o" && i+1 < len(args):
			out = args[i+1]
			i++
		case args[i] == "--stdout":
			toStdout = true
		}
	}
	src, err := os.ReadFile(in)
	if err != nil {
		fatal(err.Error())
	}
	text, err := codegen.RewriteMLIR(string(src))
	if err != nil {
		fatal(err.Error())
	}
	if toStdout || out == "-" {
		fmt.Print(text)
		return
	}
	if err := os.WriteFile(out, []byte(text), 0o644); err != nil {
		fatal(err.Error())
	}
	fmt.Printf("wrote %s\n", out)
}

func emitLLVM(args []string) {
	if len(args) < 1 {
		fatal("usage: trin emit-llvm <file.trin> [-o out.ll]")
	}
	in := args[0]
	out := strings.TrimSuffix(in, filepath.Ext(in)) + ".ll"
	toStdout := false
	for i := 1; i < len(args); i++ {
		switch {
		case args[i] == "-o" && i+1 < len(args):
			out = args[i+1]
			i++
		case args[i] == "--stdout":
			toStdout = true
		}
	}
	src, err := os.ReadFile(in)
	if err != nil {
		fatal(err.Error())
	}
	prog, err := parser.ParseSource(string(src))
	if err != nil {
		fatal(err.Error())
	}
	text, err := codegen.EmitLLVM(prog)
	if err != nil {
		fatal(err.Error())
	}
	if toStdout || out == "-" {
		fmt.Print(text)
		return
	}
	if err := os.WriteFile(out, []byte(text), 0o644); err != nil {
		fatal(err.Error())
	}
	fmt.Printf("wrote %s\n", out)
}

func emitTrinIR(args []string) {
	if len(args) < 1 {
		fatal("usage: trin emit-ir <file.trin> [-o out.ir]")
	}
	in := args[0]
	out := strings.TrimSuffix(in, filepath.Ext(in)) + ".ir"
	toStdout := false
	for i := 1; i < len(args); i++ {
		switch {
		case args[i] == "-o" && i+1 < len(args):
			out = args[i+1]
			i++
		case args[i] == "--stdout":
			toStdout = true
		}
	}
	src, err := os.ReadFile(in)
	if err != nil {
		fatal(err.Error())
	}
	prog, err := parser.ParseSource(string(src))
	if err != nil {
		fatal(err.Error())
	}
	mod, err := ir.Emit(prog)
	if err != nil {
		fatal(err.Error())
	}
	text := mod.String()
	if toStdout || out == "-" {
		fmt.Print(text)
		return
	}
	if err := os.WriteFile(out, []byte(text), 0o644); err != nil {
		fatal(err.Error())
	}
	fmt.Printf("wrote %s\n", out)
}

func defaultBackend() string {
	if b := os.Getenv("TRIN_BACKEND"); b != "" {
		return b
	}
	if _, err := exec.LookPath(goBin()); err == nil {
		return "go"
	}
	return "c"
}

func goBin() string {
	if g := os.Getenv("TRIN_GO"); g != "" {
		return g
	}
	return "go"
}

func build(in, out, backend string) error {
	src, err := os.ReadFile(in)
	if err != nil {
		return err
	}
	prog, err := parser.ParseSource(string(src))
	if err != nil {
		return err
	}
	switch backend {
	case "go", "golang":
		return buildGo(prog, out)
	case "c", "cc":
		return buildC(prog, out)
	default:
		return fmt.Errorf("unknown backend %q (want go or c)", backend)
	}
}

func buildGo(prog *ast.Program, out string) error {
	code, err := codegen.EmitGo(prog)
	if err != nil {
		return err
	}
	// go build needs a directory module or a lone file outside modules.
	// Put generated file in a temp dir with go.mod; replace points at Trin source
	// so GGUF/safetensors readers resolve via github.com/ericchien/trin/internal/weights.
	dir, err := os.MkdirTemp("", "trin-build-*")
	if err != nil {
		return err
	}
	keep := os.Getenv("TRIN_KEEP_GO") == "1"
	if !keep {
		defer os.RemoveAll(dir)
	} else {
		fmt.Fprintf(os.Stderr, "trin: kept go workspace at %s\n", dir)
	}
	gofile := filepath.Join(dir, "main.go")
	if err := os.WriteFile(gofile, []byte(code), 0o644); err != nil {
		return err
	}
	// GGUF/safetensors readers are embedded into the generated main package
	// (see codegen.emitPrelude); no extra module deps required.
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module trinprog\n\ngo 1.22\n"), 0o644); err != nil {
		return err
	}
	absOut, err := filepath.Abs(out)
	if err != nil {
		return err
	}
	cmd := exec.Command(goBin(), "build", "-o", absOut, ".")
	cmd.Dir = dir
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	// Ensure PATH has go if caller used full path
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("go build failed: %w — source at %s", err, gofile)
	}
	return nil
}

// copyGoPackage copies *.go (non-test) sources from srcDir into dstDir.
func copyGoPackage(srcDir, dstDir string) error {
	if err := os.MkdirAll(dstDir, 0o755); err != nil {
		return err
	}
	ents, err := os.ReadDir(srcDir)
	if err != nil {
		return err
	}
	for _, e := range ents {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		b, err := os.ReadFile(filepath.Join(srcDir, name))
		if err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(dstDir, name), b, 0o644); err != nil {
			return err
		}
	}
	return nil
}

// trinModuleRoot returns the absolute path of the Trin module (directory of this binary's module).
func trinModuleRoot() (string, error) {
	// Prefer TRIN_ROOT if set (CI/reproducible builds)
	if r := os.Getenv("TRIN_ROOT"); r != "" {
		return filepath.Abs(r)
	}
	// Walk from executable, then from cwd, for go.mod containing the module path
	candidates := []string{}
	if exe, err := os.Executable(); err == nil {
		candidates = append(candidates, filepath.Dir(exe))
	}
	if wd, err := os.Getwd(); err == nil {
		candidates = append(candidates, wd)
	}
	// also try relative parents from cwd
	for _, c := range candidates {
		dir := c
		for i := 0; i < 8; i++ {
			gm := filepath.Join(dir, "go.mod")
			if b, err := os.ReadFile(gm); err == nil {
				if strings.Contains(string(b), "module github.com/ericchien/trin") {
					return filepath.Abs(dir)
				}
			}
			parent := filepath.Dir(dir)
			if parent == dir {
				break
			}
			dir = parent
		}
	}
	return "", fmt.Errorf("cannot locate Trin module root (set TRIN_ROOT)")
}

func buildC(prog *ast.Program, out string) error {
	c, err := codegen.EmitC(prog)
	if err != nil {
		return err
	}
	tmp := out + ".trin.c"
	if err := os.WriteFile(tmp, []byte(c), 0o644); err != nil {
		return err
	}
	cc := os.Getenv("TRIN_CC")
	if cc == "" {
		cc = "cc"
	}
	cmd := exec.Command(cc, "-O2", "-std=gnu11", "-o", out, tmp, "-lm")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("C compile failed (%s): %w — generated source left at %s", cc, err, tmp)
	}
	if os.Getenv("TRIN_KEEP_C") != "1" {
		_ = os.Remove(tmp)
	}
	return nil
}

func emitObj(args []string) {
	if len(args) < 1 {
		fatal("usage: trin emit-obj <file.trin> [-o out.o] [--llc path]")
	}
	in := args[0]
	out := strings.TrimSuffix(filepath.Base(in), filepath.Ext(in)) + ".o"
	llcPath := os.Getenv("TRIN_LLC")
	for i := 1; i < len(args); i++ {
		switch {
		case args[i] == "-o" && i+1 < len(args):
			out = args[i+1]
			i++
		case args[i] == "--llc" && i+1 < len(args):
			llcPath = args[i+1]
			i++
		case strings.HasPrefix(args[i], "--llc="):
			llcPath = strings.TrimPrefix(args[i], "--llc=")
		}
	}
	if err := emitObjectFile(in, out, llcPath); err != nil {
		fatal(err.Error())
	}
	fmt.Printf("wrote %s\n", out)
}

// emitObjectFile lowers Trin → LLVM IR text → llc object file when llc is available.
func emitObjectFile(in, out, llcPath string) error {
	src, err := os.ReadFile(in)
	if err != nil {
		return err
	}
	prog, err := parser.ParseSource(string(src))
	if err != nil {
		return err
	}
	ll, err := codegen.EmitLLVM(prog)
	if err != nil {
		return err
	}
	llcPath, err = resolveLLC(llcPath)
	if err != nil {
		return err
	}
	dir, err := os.MkdirTemp("", "trin-llc-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)
	llPath := filepath.Join(dir, "out.ll")
	if err := os.WriteFile(llPath, []byte(ll), 0o644); err != nil {
		return err
	}
	absOut, err := filepath.Abs(out)
	if err != nil {
		return err
	}
	// llc accepts .ll text and writes a relocatable object (PIC for PIE hosts).
	cmd := exec.Command(llcPath, "-filetype=obj", "-relocation-model=pic", "-o", absOut, llPath)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		cmd = exec.Command(llcPath, "-filetype=obj", "-o", absOut, llPath)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err2 := cmd.Run(); err2 != nil {
			return fmt.Errorf("llc failed: %w (IR left unlinked; try: trin emit-llvm %s)", err, in)
		}
	}
	return nil
}

func resolveLLC(llcPath string) (string, error) {
	if llcPath == "" {
		llcPath = "llc"
	}
	if p, err := exec.LookPath(llcPath); err == nil {
		return p, nil
	}
	// Common local installs
	cands := []string{
		filepath.Join(os.Getenv("HOME"), ".local", "bin", "llc"),
		filepath.Join(os.Getenv("HOME"), ".local", "llvm", "bin", "llc"),
		"/usr/bin/llc",
		"/usr/local/bin/llc",
	}
	for _, c := range cands {
		if st, err := os.Stat(c); err == nil && !st.IsDir() {
			return c, nil
		}
	}
	return "", fmt.Errorf("llc not found (%s): install LLVM tools or set TRIN_LLC / --llc; emit-llvm still works without llc", llcPath)
}

func resolveCC(ccPath string) string {
	if ccPath == "" {
		ccPath = os.Getenv("TRIN_CC")
	}
	if ccPath == "" {
		ccPath = "cc"
	}
	if p, err := exec.LookPath(ccPath); err == nil {
		return p
	}
	return ccPath
}

func linkLLVMCmd(args []string) {
	if len(args) < 1 {
		fatal("usage: trin link-llvm <file.trin> [-o out] [--llc path] [--cc path] [--entry name]")
	}
	in := args[0]
	out := strings.TrimSuffix(filepath.Base(in), filepath.Ext(in)) + "_llvm"
	llcPath := os.Getenv("TRIN_LLC")
	ccPath := os.Getenv("TRIN_CC")
	entry := os.Getenv("TRIN_LLVM_ENTRY")
	for i := 1; i < len(args); i++ {
		switch {
		case args[i] == "-o" && i+1 < len(args):
			out = args[i+1]
			i++
		case args[i] == "--llc" && i+1 < len(args):
			llcPath = args[i+1]
			i++
		case strings.HasPrefix(args[i], "--llc="):
			llcPath = strings.TrimPrefix(args[i], "--llc=")
		case args[i] == "--cc" && i+1 < len(args):
			ccPath = args[i+1]
			i++
		case strings.HasPrefix(args[i], "--cc="):
			ccPath = strings.TrimPrefix(args[i], "--cc=")
		case args[i] == "--entry" && i+1 < len(args):
			entry = args[i+1]
			i++
		case strings.HasPrefix(args[i], "--entry="):
			entry = strings.TrimPrefix(args[i], "--entry=")
		}
	}
	if err := linkLLVMBinary(in, out, llcPath, ccPath, entry); err != nil {
		fatal(err.Error())
	}
	fmt.Printf("wrote %s (backend=llvm)\n", out)
}

// linkLLVMBinary: Trin → LLVM IR → llc .o → cc host driver → executable.
// Kernel drivers print a fixed soft_mux/loss probe when entry is a pure float graph.
func linkLLVMBinary(in, out, llcPath, ccPath, entry string) error {
	src, err := os.ReadFile(in)
	if err != nil {
		return err
	}
	prog, err := parser.ParseSource(string(src))
	if err != nil {
		return err
	}
	ll, err := codegen.EmitLLVM(prog)
	if err != nil {
		return err
	}
	llcPath, err = resolveLLC(llcPath)
	if err != nil {
		return err
	}
	ccPath = resolveCC(ccPath)
	if _, err := exec.LookPath(ccPath); err != nil {
		return fmt.Errorf("cc not found (%s): set TRIN_CC / --cc to link LLVM objects into a binary", ccPath)
	}
	if entry == "" {
		entry = detectLLVMEntry(prog, ll)
	}
	dir, err := os.MkdirTemp("", "trin-llvm-link-*")
	if err != nil {
		return err
	}
	keep := os.Getenv("TRIN_KEEP_LLVM") == "1"
	if !keep {
		defer os.RemoveAll(dir)
	} else {
		fmt.Fprintf(os.Stderr, "trin: kept llvm workspace at %s\n", dir)
	}
	llPath := filepath.Join(dir, "out.ll")
	objPath := filepath.Join(dir, "out.o")
	hostPath := filepath.Join(dir, "host.c")
	// Prefer PIC objects so modern PIE toolchains can link without R_X86_64_32 errors.
	// When forcing a float-kernel host entry (e.g. --entry trin_vec_loop_probe) while the
	// module also defines @main, rename @main so the host driver is the only entrypoint.
	linkLL := ll
	forceHost := entry != "" && entry != "main" && strings.Contains(ll, "define float @"+entry+"(")
	if forceHost && strings.Contains(linkLL, "define i32 @main(") {
		linkLL = strings.ReplaceAll(linkLL, "define i32 @main(", "define i32 @trin_prog_main(")
	}
	if err := os.WriteFile(llPath, []byte(linkLL), 0o644); err != nil {
		return err
	}
	cmd := exec.Command(llcPath, "-filetype=obj", "-relocation-model=pic", "-o", objPath, llPath)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		// Fallback without pic for older llc.
		cmd = exec.Command(llcPath, "-filetype=obj", "-o", objPath, llPath)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err2 := cmd.Run(); err2 != nil {
			return fmt.Errorf("llc failed: %w", err)
		}
	}
	host := codegen.LLVMHostDriver(entry, linkLL)
	absOut, err := filepath.Abs(out)
	if err != nil {
		return err
	}
	var link *exec.Cmd
	if strings.TrimSpace(host) == "" {
		// Full-program module already defines @main — link object alone.
		link = exec.Command(ccPath, "-O2", "-fPIE", "-pie", "-o", absOut, objPath, "-lm")
	} else {
		if err := os.WriteFile(hostPath, []byte(host), 0o644); err != nil {
			return err
		}
		link = exec.Command(ccPath, "-O2", "-fPIE", "-pie", "-o", absOut, hostPath, objPath, "-lm")
	}
	link.Stdout = os.Stdout
	link.Stderr = os.Stderr
	if err := link.Run(); err != nil {
		return fmt.Errorf("link failed (%s): %w — try: trin emit-obj %s && link with your host", ccPath, err, in)
	}
	return nil
}

func detectLLVMEntry(prog *ast.Program, ll string) string {
	// Full-program modules own @main — no float host entry needed.
	if strings.Contains(ll, "define i32 @main(") {
		return "main"
	}
	// Prefer pure float kernels exported by emit-llvm demos.
	// Note: trin_vec_loop_probe is always present (v0.20) but only chosen when no loss/kernel.
	for _, name := range []string{"loss", "kernel", "soft_kernel", "trin_demo_kernel", "trin_vec_loop_probe", "trin_soft_mux"} {
		if strings.Contains(ll, "define float @"+name+"(") {
			return name
		}
	}
	if prog != nil {
		for _, it := range prog.Items {
			if fn, ok := it.(*ast.FnDef); ok && fn.Name != "main" {
				if strings.Contains(ll, "define float @"+fn.Name+"(") {
					return fn.Name
				}
			}
		}
	}
	return "trin_soft_mux"
}

func usage() {
	fmt.Fprintf(os.Stderr, `trin %s — ternary language compiler (Go)

Usage:
  trin run <file.trin>                      interpret
  trin build <file.trin> [-o x] [--backend go|c|llvm]
                                            compile to native binary
  trin build <file.trin> --link-llvm [-o x] same as --backend llvm
  trin emit-go <file.trin> [-o x]           emit Go only
  trin emit-c <file.trin> [-o x]            emit C only
  trin emit-ir <file.trin> [-o x]           emit TrinIR text
  trin emit-llvm <file.trin> [-o x]         emit LLVM IR text
  trin emit-mlir <file.trin> [-o x]         emit MLIR dialect / vector IR text
  trin emit-mlir-llvm <file.trin> [-o x]    emit-mlir then lower to LLVM IR
  trin lower-mlir <file.mlir> [-o x]        lower existing Trin MLIR text → LLVM IR
  trin rewrite-mlir <file.mlir> [-o x]      portable pattern-match rewrites on MLIR
  trin emit-obj <file.trin> [-o x.o]        LLVM IR → object via llc (optional)
  trin link-llvm <file.trin> [-o x]         LLVM IR → llc .o → cc host binary
  trin build <file.trin> --emit-obj [-o x.o]
                                            same as emit-obj
  trin <file.trin>                          same as run
  trin version

Default build backend: go (when go is on PATH), else c.
Override with --backend or TRIN_BACKEND.

Env:
  TRIN_BACKEND     go|c|llvm (default: auto go/c)
  TRIN_CC          C compiler (backend=c / llvm link; default: cc)
  TRIN_LLC         path to llc (emit-obj / link-llvm)
  TRIN_LLVM_ENTRY  float kernel symbol for host driver (default: auto)
  TRIN_KEEP_C      keep intermediate .c if 1
  TRIN_KEEP_GO     keep intermediate go workspace if 1
  TRIN_KEEP_LLVM   keep llc/host intermediates if 1
  TRIN_GO          go binary (default: go)
`, version)
}

func fatal(msg string) {
	fmt.Fprintln(os.Stderr, "trin:", msg)
	os.Exit(1)
}
