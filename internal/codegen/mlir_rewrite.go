package codegen

import (
	"fmt"
	"regexp"
	"strings"
)

// RewriteMLIR applies portable pattern-match rewrites on Trin MLIR text (v0.25+).
// No system mlir-opt is required. Rewrites are idempotent: already-expanded
// modules pass through with a rewrite summary note.
//
// Patterns:
//  1. High-level dialect ops "trin.*" → func.call @trin_* helpers (width by type)
//  2. soft_mux constant folds: w=1 → pure, w=0 → each
//  3. soft_mux strength / algebraic (v0.29 + v0.33 vector arms):
//       full const fold; pure==each → pure; each=0 → w*pure; pure=0 → (1-w)*each
//       also on soft_mux_vN when pure/each is a known zero vector
//  4. span/vector algebraic (v0.31):
//       span_add(x,0)/span_add(0,x) → x; span_scale(1,x) → x; span_scale(0,x) → 0;
//       sum(0) → 0; dot(0,x)/dot(x,0) → 0  (vector tiles v4/v8/v16)
//  5. bitplane / plane algebraic (v0.33):
//       plane_and(x,x)→x; plane_and(x,0)|and(0,x)→0; plane_or(x,x)→x; plane_or(x,0)|or(0,x)→x;
//       plane_xor(x,x)→0; popcount(0)→0; const popcount fold
//  6. nested soft_mux / bitplane product folds (v0.35):
//       soft_mux(w1, soft_mux(w2,p,e), e) → soft_mux(w1*w2, p, e)  (product weight)
//       soft_mux(w1, p, soft_mux(w2,p,e)) → soft_mux(w1+(1-w1)*w2, p, e)
//       plane_and(and(a,b), a|b) → and(a,b); plane_or(or(a,b), a|b) → or(a,b)
//       plane_and(a, or(a,b)) → a; plane_or(a, and(a,b)) → a  (absorption)
//  6b. multi-level soft_mux trees + multi-word plane GEMM (v0.38):
//       triple pure nest → w1*w2*w3; dual pure→each combo
//       soft_mux(w, soft_mux(w2,p,e), soft_mux(w3,p,e)) → soft_mux(w*w2+(1-w)*w3, p, e)
//       soft_mux(w, soft_mux(w2,p,e), soft_mux(w3,p2,e2)) partial left when outer const
//       trit_dot_planes_word zero-arm / same-vector folds
//       trit_dot_planes_x4 zero-word / all-zero / sum fold
//       multi-word plane_and/or/xor/popcount chain folds
//  6c. soft_mux forest + multi-word GEMM schedule (v0.40):
//       soft_mux(w, soft_mux(w2, soft_mux(w3,p,e), e), soft_mux(w4,p,e))
//         → soft_mux(w*w2*w3+(1-w)*w4, p, e)   (forest pure-chain both-arms)
//       scheduled x4: equal const words fold to 4*dot; mixed const schedule folds sum
//  6d. deeper soft_mux forest schedule (v0.41):
//       soft_mux(w, soft_mux(w2, soft_mux(w3, soft_mux(w4,p,e), e), e), soft_mux(w5,p,e))
//         → soft_mux(w*w2*w3*w4+(1-w)*w5, p, e)
//       soft_mux(w, soft_mux(w2,p,e), soft_mux(w3, soft_mux(w4,p,e), e))
//         → soft_mux(w*w2+(1-w)*w3*w4, p, e)   (each-arm product chain)
//  7. CSE of pure SSA bindings (arith / constants / soft_mux multi-use commoning)
//  8. Vector-width pick stamped when dialect op chooses v4/v8/v16
func RewriteMLIR(mlir string) (string, error) {
	if strings.TrimSpace(mlir) == "" {
		return "", fmt.Errorf("empty MLIR input")
	}
	if !strings.Contains(mlir, "module attributes {trin.version") &&
		!strings.Contains(mlir, "func.func @trin_soft_mux") &&
		!strings.Contains(mlir, `"trin.`) {
		return "", fmt.Errorf("not a Trin MLIR module (missing trin.version / soft_mux / dialect ops)")
	}

	stats := rewriteStats{}
	out := rewriteDialectOps(mlir, &stats)
	out = rewriteSoftMuxFolds(out, &stats)
	out = rewriteSpanVectorFolds(out, &stats)
	out = rewriteBitplaneFolds(out, &stats)
	out = rewriteCSE(out, &stats)
	out = stampRewriteHeader(out, stats)
	return out, nil
}

type rewriteStats struct {
	dialect   int
	softFolds int
	strength  int
	spanAlg   int
	bitplane  int
	nested    int
	cse       int
	vwidth    int
}

// softMuxForm records an unfolder soft_mux call for nested folds (v0.35).
type softMuxForm struct {
	w, pure, each string
}

// planeBinForm records an unfolded plane and/or for product folds (v0.35).
type planeBinForm struct {
	op   string // "and" | "or" | "xor"
	a, b string
}

// Match: %y = "trin.soft_mux"(%w, %pure, %each) : (f32, f32, f32) -> f32
var reDialectOp = regexp.MustCompile(
	`(?P<indent>[ \t]*)(?P<bind>%[A-Za-z0-9_.]+)\s*=\s*"(?P<op>trin\.[A-Za-z0-9_]+)"\s*\((?P<args>[^)]*)\)\s*:\s*(?P<sig>[^\n]+)`,
)

// Void form without result binding.
var reDialectOpVoid = regexp.MustCompile(
	`(?P<indent>[ \t]*)"(?P<op>trin\.[A-Za-z0-9_]+)"\s*\((?P<args>[^)]*)\)\s*:\s*(?P<sig>[^\n]+)`,
)

func rewriteDialectOps(mlir string, st *rewriteStats) string {
	lines := strings.Split(mlir, "\n")
	for i, line := range lines {
		if !strings.Contains(line, `"trin.`) {
			continue
		}
		if m := reDialectOp.FindStringSubmatch(line); m != nil {
			g := reGroups(reDialectOp, m)
			callee, widthPick, ok := dialectCallee(g["op"], g["sig"])
			if !ok {
				continue
			}
			lines[i] = fmt.Sprintf("%s%s = func.call @%s(%s) : %s",
				g["indent"], g["bind"], callee, strings.TrimSpace(g["args"]), strings.TrimSpace(g["sig"]))
			if widthPick > 0 {
				// stamp inline width-pick note on previous line insertion via comment prefix
				lines[i] = fmt.Sprintf("%s// rewrite: vector_width=%d → @%s\n%s",
					g["indent"], widthPick, callee, lines[i])
				st.vwidth++
			}
			st.dialect++
			continue
		}
		if strings.Contains(line, "=") {
			continue
		}
		if m := reDialectOpVoid.FindStringSubmatch(line); m != nil {
			g := reGroups(reDialectOpVoid, m)
			callee, widthPick, ok := dialectCallee(g["op"], g["sig"])
			if !ok {
				continue
			}
			call := fmt.Sprintf("%sfunc.call @%s(%s) : %s",
				g["indent"], callee, strings.TrimSpace(g["args"]), strings.TrimSpace(g["sig"]))
			if widthPick > 0 {
				call = fmt.Sprintf("%s// rewrite: vector_width=%d → @%s\n%s",
					g["indent"], widthPick, callee, call)
				st.vwidth++
			}
			lines[i] = call
			st.dialect++
		}
	}
	return strings.Join(lines, "\n")
}

// dialectCallee maps a high-level trin.* op + signature to a helper.
// widthPick is 4/8/16 when a vector tile width was selected; 0 for scalar helpers.
func dialectCallee(op, sig string) (callee string, widthPick int, ok bool) {
	sig = strings.ToLower(strings.TrimSpace(sig))
	width := 0
	switch {
	case strings.Contains(sig, "vector<16"):
		width = 16
	case strings.Contains(sig, "vector<8"):
		width = 8
	case strings.Contains(sig, "vector<4"):
		width = 4
	}
	switch op {
	case "trin.soft_mux":
		if width == 16 {
			return "trin_soft_mux_v16", width, true
		}
		if width == 8 {
			return "trin_soft_mux_v8", width, true
		}
		if width == 4 {
			return "trin_soft_mux_v4", width, true
		}
		return "trin_soft_mux", 0, true
	case "trin.span_add":
		return "trin_span_add_f32", 0, true
	case "trin.span_scale":
		return "trin_span_scale_f32", 0, true
	case "trin.sum":
		if width == 16 {
			return "trin_sum_v16", width, true
		}
		if width == 8 {
			return "trin_sum_v8", width, true
		}
		if width == 4 {
			return "trin_sum_v4", width, true
		}
		return "trin_sum_f32", 0, true
	case "trin.dot":
		if width == 16 {
			return "trin_dot_v16", width, true
		}
		if width == 8 {
			return "trin_dot_v8", width, true
		}
		if width == 4 {
			return "trin_dot_v4", width, true
		}
		return "trin_dot_f32", 0, true
	case "trin.vspan_add", "trin.span_add_v":
		if width == 16 {
			return "trin_span_add_v16", width, true
		}
		if width == 8 {
			return "trin_span_add_v8", width, true
		}
		return "trin_span_add_v4", 4, true
	case "trin.vspan_scale", "trin.span_scale_v":
		if width == 16 {
			return "trin_span_scale_v16", width, true
		}
		if width == 8 {
			return "trin_span_scale_v8", width, true
		}
		return "trin_span_scale_v4", 4, true
	case "trin.vsum":
		if width == 16 {
			return "trin_sum_v16", width, true
		}
		if width == 8 {
			return "trin_sum_v8", width, true
		}
		return "trin_sum_v4", 4, true
	case "trin.vdot":
		if width == 16 {
			return "trin_dot_v16", width, true
		}
		if width == 8 {
			return "trin_dot_v8", width, true
		}
		return "trin_dot_v4", 4, true
	default:
		if strings.HasPrefix(op, "trin.") {
			return "trin_" + strings.ReplaceAll(op[len("trin."):], ".", "_"), width, true
		}
		return "", 0, false
	}
}

// Soft-mux call after dialect expansion (scalar f32 form only for folding).
var reSoftMuxCall = regexp.MustCompile(
	`(?P<indent>[ \t]*)(?P<bind>%[A-Za-z0-9_.]+)\s*=\s*func\.call\s+@trin_soft_mux\(\s*(?P<w>%[A-Za-z0-9_.]+)\s*,\s*(?P<pure>%[A-Za-z0-9_.]+)\s*,\s*(?P<each>%[A-Za-z0-9_.]+)\s*\)\s*:\s*\(f32,\s*f32,\s*f32\)\s*->\s*f32`,
)

var reConstF32 = regexp.MustCompile(
	`(?P<name>%[A-Za-z0-9_.]+)\s*=\s*arith\.constant\s+(?P<val>[-+0-9.eE]+)\s*:\s*f32`,
)

// reSoftMuxVecCall matches vector soft_mux helpers for pure==each strength reduction.
var reSoftMuxVecCall = regexp.MustCompile(
	`(?P<indent>[ \t]*)(?P<bind>%[A-Za-z0-9_.]+)\s*=\s*func\.call\s+@(?P<fn>trin_soft_mux_v(?:4|8|16))\(\s*(?P<w>%[A-Za-z0-9_.]+)\s*,\s*(?P<pure>%[A-Za-z0-9_.]+)\s*,\s*(?P<each>%[A-Za-z0-9_.]+)\s*\)\s*:\s*(?P<sig>\([^)]*\)\s*->\s*vector<(?P<n>\d+)xf32>)`,
)

func rewriteSoftMuxFolds(mlir string, st *rewriteStats) string {
	consts := map[string]float64{}
	zeroVec := map[string]bool{}
	muxes := map[string]softMuxForm{} // bind → form for nested folds (v0.35)
	lines := strings.Split(mlir, "\n")
	var out []string
	for _, line := range lines {
		if m := reConstF32.FindStringSubmatch(line); m != nil {
			g := reGroups(reConstF32, m)
			var v float64
			if _, err := fmt.Sscanf(g["val"], "%f", &v); err == nil {
				consts[g["name"]] = v
			}
			out = append(out, line)
			continue
		}
		// Track zero vectors early for soft_mux_v strength (v0.33).
		if bm := reVecBroadcast.FindStringSubmatch(line); bm != nil {
			g := reGroups(reVecBroadcast, bm)
			if c, ok := consts[g["src"]]; ok && almostZero(c) {
				zeroVec[g["bind"]] = true
			}
		}

		m := reSoftMuxCall.FindStringSubmatch(line)
		if m == nil {
			// Vector soft_mux strength (v0.29 identity + v0.33 each=0 / pure=0).
			if vm := reSoftMuxVecCall.FindStringSubmatch(line); vm != nil {
				g := reGroups(reSoftMuxVecCall, vm)
				if g["pure"] == g["each"] {
					for _, l := range emitSoftMuxIdentity(g["indent"], g["bind"], g["pure"],
						fmt.Sprintf("soft_mux_v%s(w,x,x) → x", g["n"]), g["n"]) {
						out = append(out, l)
					}
					st.strength++
					continue
				}
				// each is zero vector → broadcast(w)*pure  (v0.33)
				if zeroVec[g["each"]] {
					wv := g["bind"] + "_wv"
					out = append(out, fmt.Sprintf("%s// rewrite: soft_mux_v%s(w,pure,0) → broadcast(w)*pure", g["indent"], g["n"]))
					out = append(out, fmt.Sprintf("%s%s = vector.broadcast %s : f32 to vector<%sxf32>",
						g["indent"], wv, g["w"], g["n"]))
					out = append(out, fmt.Sprintf("%s%s = arith.mulf %s, %s : vector<%sxf32>",
						g["indent"], g["bind"], wv, g["pure"], g["n"]))
					st.strength++
					continue
				}
				// pure is zero vector → broadcast(1-w)*each  (v0.33)
				if zeroVec[g["pure"]] {
					c1 := g["bind"] + "_one"
					omw := g["bind"] + "_omw"
					ov := g["bind"] + "_ov"
					out = append(out, fmt.Sprintf("%s// rewrite: soft_mux_v%s(w,0,each) → broadcast(1-w)*each", g["indent"], g["n"]))
					out = append(out, fmt.Sprintf("%s%s = arith.constant 1.000000e+00 : f32", g["indent"], c1))
					out = append(out, fmt.Sprintf("%s%s = arith.subf %s, %s : f32", g["indent"], omw, c1, g["w"]))
					out = append(out, fmt.Sprintf("%s%s = vector.broadcast %s : f32 to vector<%sxf32>",
						g["indent"], ov, omw, g["n"]))
					out = append(out, fmt.Sprintf("%s%s = arith.mulf %s, %s : vector<%sxf32>",
						g["indent"], g["bind"], ov, g["each"], g["n"]))
					st.strength++
					continue
				}
			}
			out = append(out, line)
			continue
		}
		g := reGroups(reSoftMuxCall, m)
		wName, pureName, eachName := g["w"], g["pure"], g["each"]
		w, wOK := consts[wName]
		pure, pureOK := consts[pureName]
		each, eachOK := consts[eachName]

		// 1) pure SSA equals each SSA → pure (identity blend; any w).
		if pureName == eachName {
			for _, l := range emitSoftMuxIdentity(g["indent"], g["bind"], pureName,
				"soft_mux(w,x,x) → x", "") {
				out = append(out, l)
			}
			st.strength++
			continue
		}
		// same when both sides are equal known constants (different SSA).
		if pureOK && eachOK && almostEqual(pure, each) {
			for _, l := range emitSoftMuxIdentity(g["indent"], g["bind"], pureName,
				"soft_mux(w,c,c) → c", "") {
				out = append(out, l)
			}
			st.strength++
			continue
		}

		// 2) classic w=0 / w=1 constant folds.
		if wOK {
			switch {
			case almostOne(w):
				for _, l := range emitSoftMuxIdentity(g["indent"], g["bind"], pureName,
					"soft_mux(1,pure,each) → pure", "") {
					out = append(out, l)
				}
				st.softFolds++
				continue
			case almostZero(w):
				for _, l := range emitSoftMuxIdentity(g["indent"], g["bind"], eachName,
					"soft_mux(0,pure,each) → each", "") {
					out = append(out, l)
				}
				st.softFolds++
				continue
			}
		}

		// 3) strength: each=0 → w*pure  (before full-const so mulf form is preferred)
		if eachOK && almostZero(each) {
			out = append(out, fmt.Sprintf("%s// rewrite: soft_mux(w,pure,0) → w*pure", g["indent"]))
			out = append(out, fmt.Sprintf("%s%s = arith.mulf %s, %s : f32",
				g["indent"], g["bind"], wName, pureName))
			st.strength++
			continue
		}

		// 4) strength: pure=0 → (1-w)*each
		if pureOK && almostZero(pure) {
			c1 := g["bind"] + "_one"
			one := g["bind"] + "_omw"
			out = append(out, fmt.Sprintf("%s// rewrite: soft_mux(w,0,each) → (1-w)*each", g["indent"]))
			out = append(out, fmt.Sprintf("%s%s = arith.constant 1.000000e+00 : f32", g["indent"], c1))
			out = append(out, fmt.Sprintf("%s%s = arith.subf %s, %s : f32", g["indent"], one, c1, wName))
			out = append(out, fmt.Sprintf("%s%s = arith.mulf %s, %s : f32",
				g["indent"], g["bind"], one, eachName))
			st.strength++
			continue
		}

		// 5) full constant fold: soft_mux(wc,pc,ec) → arith.constant (v0.29).
		if wOK && pureOK && eachOK {
			y := w*pure + (1-w)*each
			out = append(out, fmt.Sprintf("%s// rewrite: soft_mux(const) → %.6g", g["indent"], y))
			out = append(out, fmt.Sprintf("%s%s = arith.constant %s : f32",
				g["indent"], g["bind"], formatF32MLIR(y)))
			st.strength++
			// register newly produced constant for downstream folds
			consts[g["bind"]] = y
			continue
		}

		// 6) nested soft_mux folds (v0.35)
		if folded, form := tryNestedSoftMuxFold(g["indent"], g["bind"], wName, pureName, eachName,
			w, wOK, consts, muxes, st); folded {
			out = append(out, form...)
			continue
		}

		// Keep unfolder form for subsequent nested matches.
		muxes[g["bind"]] = softMuxForm{w: wName, pure: pureName, each: eachName}
		out = append(out, line)
	}
	return strings.Join(out, "\n")
}

// tryNestedSoftMuxFold attempts product-weight rewrites for nested soft_mux
// (v0.35 binary nests + v0.38 multi-level trees + v0.40 forest schedule).
// Algebra:
//   soft_mux(w1, soft_mux(w2,p,e), e) = soft_mux(w1*w2, p, e)
//   soft_mux(w1, p, soft_mux(w2,p,e)) = soft_mux(w1+(1-w1)*w2, p, e)
//   soft_mux(w, soft_mux(w2,p,e), soft_mux(w3,p,e)) = soft_mux(w*w2+(1-w)*w3, p, e)  (v0.38)
//   soft_mux(w, soft_mux(w2, soft_mux(w3,p,e), e), soft_mux(w4,p,e))
//     = soft_mux(w*w2*w3+(1-w)*w4, p, e)  (v0.40 forest)
//   soft_mux(w, soft_mux(w2, soft_mux(w3, soft_mux(w4,p,e), e), e), soft_mux(w5,p,e))
//     = soft_mux(w*w2*w3*w4+(1-w)*w5, p, e)  (v0.41 deep forest)
//   soft_mux(w, soft_mux(w2,p,e), soft_mux(w3, soft_mux(w4,p,e), e))
//     = soft_mux(w*w2+(1-w)*w3*w4, p, e)  (v0.41 each-arm product)
// Multi-level pure chains fold stepwise: w1*(w2*w3).
func tryNestedSoftMuxFold(
	indent, bind, wName, pureName, eachName string,
	w float64, wOK bool,
	consts map[string]float64,
	muxes map[string]softMuxForm,
	st *rewriteStats,
) (bool, []string) {
	// v0.40/v0.41 forest: pure arm is a pure-nest product chain with same each; each arm shares (p,e)
	if pureIn, okP := muxes[pureName]; okP {
		if eachIn, okE := muxes[eachName]; okE {
			// both arms share leaf (p,e)
			if pureIn.pure == eachIn.pure && pureIn.each == eachIn.each {
				return emitSoftMuxBothArmsSameTree(indent, bind, wName, pureIn.w, eachIn.w,
					pureIn.pure, pureIn.each, w, wOK, consts, muxes, st)
			}
			// v0.41 deep forest: pure arm is double product nest
			// pureIn = soft_mux(w2, mid, e); mid = soft_mux(w3, leaf, e); leaf = soft_mux(w4,p,e)
			// eachIn = soft_mux(w5,p,e)
			if mid, okM := muxes[pureIn.pure]; okM && pureIn.each == mid.each {
				if leaf, okL := muxes[mid.pure]; okL && mid.each == leaf.each &&
					leaf.pure == eachIn.pure && leaf.each == eachIn.each {
					return emitSoftMuxDeepForestProduct(indent, bind, wName, pureIn.w, mid.w, leaf.w, eachIn.w,
						leaf.pure, leaf.each, w, wOK, consts, muxes, st)
				}
				// v0.40 depth-2 forest
				if mid.pure == eachIn.pure && mid.each == eachIn.each {
					return emitSoftMuxForestProductBothArms(indent, bind, wName, pureIn.w, mid.w, eachIn.w,
						mid.pure, mid.each, w, wOK, consts, muxes, st)
				}
			}
			// v0.41 each-arm product chain:
			// pureIn = soft_mux(w2,p,e); eachIn = soft_mux(w3, soft_mux(w4,p,e), e)
			if eMid, okEM := muxes[eachIn.pure]; okEM &&
				eachIn.each == eMid.each &&
				eMid.pure == pureIn.pure && eMid.each == pureIn.each {
				return emitSoftMuxForestEachArmProduct(indent, bind, wName, pureIn.w, eachIn.w, eMid.w,
					pureIn.pure, pureIn.each, w, wOK, consts, muxes, st)
			}
		}
	}
	// soft_mux(w1, soft_mux(w2,p,e), e) → soft_mux(w1*w2, p, e)  (product weight)
	if inner, ok := muxes[pureName]; ok && inner.each == eachName {
		return emitSoftMuxProductWeight(indent, bind, wName, inner.w, inner.pure, eachName,
			w, wOK, consts, muxes, st, true)
	}
	// soft_mux(w1, p, soft_mux(w2,p,e)) → soft_mux(w1+(1-w1)*w2, p, e)
	if inner, ok := muxes[eachName]; ok && inner.pure == pureName {
		return emitSoftMuxEachArmNest(indent, bind, wName, pureName, inner.w, eachName,
			w, wOK, consts, muxes, st)
	}
	return false, nil
}

// emitSoftMuxDeepForestProduct rewrites
// soft_mux(w, soft_mux(w2, soft_mux(w3, soft_mux(w4,p,e), e), e), soft_mux(w5,p,e))
// → soft_mux(w*w2*w3*w4+(1-w)*w5, p, e).
func emitSoftMuxDeepForestProduct(
	indent, bind, w, w2, w3, w4, w5, pure, each string,
	wv float64, wOK bool,
	consts map[string]float64,
	muxes map[string]softMuxForm,
	st *rewriteStats,
) (bool, []string) {
	w2v, w2OK := consts[w2]
	w3v, w3OK := consts[w3]
	w4v, w4OK := consts[w4]
	w5v, w5OK := consts[w5]
	pureV, pureOK := consts[pure]
	eachV, eachOK := consts[each]
	if wOK && w2OK && w3OK && w4OK && w5OK && pureOK && eachOK {
		wn := wv*w2v*w3v*w4v + (1-wv)*w5v
		y := wn*pureV + (1-wn)*eachV
		lines := []string{
			fmt.Sprintf("%s// rewrite: soft_mux deep forest schedule → soft_mux(w*w2*w3*w4+(1-w)*w5,p,e) const → %.6g", indent, y),
			fmt.Sprintf("%s%s = arith.constant %s : f32", indent, bind, formatF32MLIR(y)),
		}
		consts[bind] = y
		st.nested++
		return true, lines
	}
	if wOK && w2OK && w3OK && w4OK && w5OK {
		wn := wv * w2v * w3v * w4v + (1-wv)*w5v
		cw := bind + "_wt"
		lines := []string{
			fmt.Sprintf("%s// rewrite: soft_mux deep forest schedule → soft_mux(w*w2*w3*w4+(1-w)*w5, p, e)", indent),
			fmt.Sprintf("%s%s = arith.constant %s : f32", indent, cw, formatF32MLIR(wn)),
			fmt.Sprintf("%s%s = func.call @trin_soft_mux(%s, %s, %s) : (f32, f32, f32) -> f32",
				indent, bind, cw, pure, each),
		}
		consts[cw] = wn
		muxes[bind] = softMuxForm{w: cw, pure: pure, each: each}
		st.nested++
		return true, lines
	}
	// Non-const: wn = w*w2*w3*w4 + (1-w)*w5
	c1 := bind + "_one"
	omw := bind + "_omw"
	t0 := bind + "_t0"
	t1 := bind + "_t1"
	t2 := bind + "_t2"
	t3 := bind + "_t3"
	cw := bind + "_wt"
	lines := []string{
		fmt.Sprintf("%s// rewrite: soft_mux deep forest schedule → soft_mux(w*w2*w3*w4+(1-w)*w5, p, e)", indent),
		fmt.Sprintf("%s%s = arith.constant 1.000000e+00 : f32", indent, c1),
		fmt.Sprintf("%s%s = arith.subf %s, %s : f32", indent, omw, c1, w),
		fmt.Sprintf("%s%s = arith.mulf %s, %s : f32", indent, t0, w, w2),
		fmt.Sprintf("%s%s = arith.mulf %s, %s : f32", indent, t1, t0, w3),
		fmt.Sprintf("%s%s = arith.mulf %s, %s : f32", indent, t2, t1, w4),
		fmt.Sprintf("%s%s = arith.mulf %s, %s : f32", indent, t3, omw, w5),
		fmt.Sprintf("%s%s = arith.addf %s, %s : f32", indent, cw, t2, t3),
		fmt.Sprintf("%s%s = func.call @trin_soft_mux(%s, %s, %s) : (f32, f32, f32) -> f32",
			indent, bind, cw, pure, each),
	}
	muxes[bind] = softMuxForm{w: cw, pure: pure, each: each}
	st.nested++
	return true, lines
}

// emitSoftMuxForestEachArmProduct rewrites
// soft_mux(w, soft_mux(w2,p,e), soft_mux(w3, soft_mux(w4,p,e), e))
// → soft_mux(w*w2+(1-w)*w3*w4, p, e).
func emitSoftMuxForestEachArmProduct(
	indent, bind, w, w2, w3, w4, pure, each string,
	wv float64, wOK bool,
	consts map[string]float64,
	muxes map[string]softMuxForm,
	st *rewriteStats,
) (bool, []string) {
	w2v, w2OK := consts[w2]
	w3v, w3OK := consts[w3]
	w4v, w4OK := consts[w4]
	pureV, pureOK := consts[pure]
	eachV, eachOK := consts[each]
	if wOK && w2OK && w3OK && w4OK && pureOK && eachOK {
		wn := wv*w2v + (1-wv)*w3v*w4v
		y := wn*pureV + (1-wn)*eachV
		lines := []string{
			fmt.Sprintf("%s// rewrite: soft_mux forest each-arm product → soft_mux(w*w2+(1-w)*w3*w4,p,e) const → %.6g", indent, y),
			fmt.Sprintf("%s%s = arith.constant %s : f32", indent, bind, formatF32MLIR(y)),
		}
		consts[bind] = y
		st.nested++
		return true, lines
	}
	if wOK && w2OK && w3OK && w4OK {
		wn := wv*w2v + (1-wv)*w3v*w4v
		cw := bind + "_wt"
		lines := []string{
			fmt.Sprintf("%s// rewrite: soft_mux forest each-arm product → soft_mux(w*w2+(1-w)*w3*w4, p, e)", indent),
			fmt.Sprintf("%s%s = arith.constant %s : f32", indent, cw, formatF32MLIR(wn)),
			fmt.Sprintf("%s%s = func.call @trin_soft_mux(%s, %s, %s) : (f32, f32, f32) -> f32",
				indent, bind, cw, pure, each),
		}
		consts[cw] = wn
		muxes[bind] = softMuxForm{w: cw, pure: pure, each: each}
		st.nested++
		return true, lines
	}
	c1 := bind + "_one"
	omw := bind + "_omw"
	t0 := bind + "_t0"
	t1 := bind + "_t1"
	t2 := bind + "_t2"
	cw := bind + "_wt"
	lines := []string{
		fmt.Sprintf("%s// rewrite: soft_mux forest each-arm product → soft_mux(w*w2+(1-w)*w3*w4, p, e)", indent),
		fmt.Sprintf("%s%s = arith.constant 1.000000e+00 : f32", indent, c1),
		fmt.Sprintf("%s%s = arith.subf %s, %s : f32", indent, omw, c1, w),
		fmt.Sprintf("%s%s = arith.mulf %s, %s : f32", indent, t0, w, w2),
		fmt.Sprintf("%s%s = arith.mulf %s, %s : f32", indent, t1, w3, w4),
		fmt.Sprintf("%s%s = arith.mulf %s, %s : f32", indent, t2, omw, t1),
		fmt.Sprintf("%s%s = arith.addf %s, %s : f32", indent, cw, t0, t2),
		fmt.Sprintf("%s%s = func.call @trin_soft_mux(%s, %s, %s) : (f32, f32, f32) -> f32",
			indent, bind, cw, pure, each),
	}
	muxes[bind] = softMuxForm{w: cw, pure: pure, each: each}
	st.nested++
	return true, lines
}

// emitSoftMuxForestProductBothArms rewrites
// soft_mux(w, soft_mux(w2, soft_mux(w3,p,e), e), soft_mux(w4,p,e))
// → soft_mux(w*w2*w3+(1-w)*w4, p, e).
func emitSoftMuxForestProductBothArms(
	indent, bind, w, w2, w3, w4, pure, each string,
	wv float64, wOK bool,
	consts map[string]float64,
	muxes map[string]softMuxForm,
	st *rewriteStats,
) (bool, []string) {
	w2v, w2OK := consts[w2]
	w3v, w3OK := consts[w3]
	w4v, w4OK := consts[w4]
	pureV, pureOK := consts[pure]
	eachV, eachOK := consts[each]
	if wOK && w2OK && w3OK && w4OK && pureOK && eachOK {
		wn := wv*w2v*w3v + (1-wv)*w4v
		y := wn*pureV + (1-wn)*eachV
		lines := []string{
			fmt.Sprintf("%s// rewrite: soft_mux forest product both-arms → soft_mux(w*w2*w3+(1-w)*w4,p,e) const → %.6g", indent, y),
			fmt.Sprintf("%s%s = arith.constant %s : f32", indent, bind, formatF32MLIR(y)),
		}
		consts[bind] = y
		st.nested++
		return true, lines
	}
	if wOK && w2OK && w3OK && w4OK {
		wn := wv*w2v*w3v + (1-wv)*w4v
		cw := bind + "_wt"
		lines := []string{
			fmt.Sprintf("%s// rewrite: soft_mux forest product both-arms → soft_mux(w*w2*w3+(1-w)*w4, p, e)", indent),
			fmt.Sprintf("%s%s = arith.constant %s : f32", indent, cw, formatF32MLIR(wn)),
			fmt.Sprintf("%s%s = func.call @trin_soft_mux(%s, %s, %s) : (f32, f32, f32) -> f32",
				indent, bind, cw, pure, each),
		}
		consts[cw] = wn
		muxes[bind] = softMuxForm{w: cw, pure: pure, each: each}
		if almostOne(wn) {
			return true, emitSoftMuxIdentity(indent, bind, pure,
				"soft_mux(forest, wn=1) → p", "")
		}
		if almostZero(wn) {
			return true, emitSoftMuxIdentity(indent, bind, each,
				"soft_mux(forest, wn=0) → e", "")
		}
		st.nested++
		return true, lines
	}
	// Non-const: wn = w*w2*w3 + (1-w)*w4
	c1 := bind + "_one"
	omw := bind + "_omw"
	t0 := bind + "_t0"
	t1 := bind + "_t1"
	t2 := bind + "_t2"
	cw := bind + "_wt"
	lines := []string{
		fmt.Sprintf("%s// rewrite: soft_mux forest product both-arms → soft_mux(w*w2*w3+(1-w)*w4, p, e)", indent),
		fmt.Sprintf("%s%s = arith.constant 1.000000e+00 : f32", indent, c1),
		fmt.Sprintf("%s%s = arith.subf %s, %s : f32", indent, omw, c1, w),
		fmt.Sprintf("%s%s = arith.mulf %s, %s : f32", indent, t0, w, w2),
		fmt.Sprintf("%s%s = arith.mulf %s, %s : f32", indent, t1, t0, w3),
		fmt.Sprintf("%s%s = arith.mulf %s, %s : f32", indent, t2, omw, w4),
		fmt.Sprintf("%s%s = arith.addf %s, %s : f32", indent, cw, t1, t2),
		fmt.Sprintf("%s%s = func.call @trin_soft_mux(%s, %s, %s) : (f32, f32, f32) -> f32",
			indent, bind, cw, pure, each),
	}
	muxes[bind] = softMuxForm{w: cw, pure: pure, each: each}
	st.nested++
	return true, lines
}

// emitSoftMuxBothArmsSameTree rewrites
// soft_mux(w, soft_mux(w2,p,e), soft_mux(w3,p,e)) → soft_mux(w*w2+(1-w)*w3, p, e).
func emitSoftMuxBothArmsSameTree(
	indent, bind, w, w2, w3, pure, each string,
	wv float64, wOK bool,
	consts map[string]float64,
	muxes map[string]softMuxForm,
	st *rewriteStats,
) (bool, []string) {
	w2v, w2OK := consts[w2]
	w3v, w3OK := consts[w3]
	pureV, pureOK := consts[pure]
	eachV, eachOK := consts[each]
	if wOK && w2OK && w3OK && pureOK && eachOK {
		wn := wv*w2v + (1-wv)*w3v
		y := wn*pureV + (1-wn)*eachV
		lines := []string{
			fmt.Sprintf("%s// rewrite: soft_mux(w, soft_mux(w2,p,e), soft_mux(w3,p,e)) → soft_mux(w*w2+(1-w)*w3,p,e) const → %.6g", indent, y),
			fmt.Sprintf("%s%s = arith.constant %s : f32", indent, bind, formatF32MLIR(y)),
		}
		consts[bind] = y
		st.nested++
		return true, lines
	}
	if wOK && w2OK && w3OK {
		wn := wv*w2v + (1-wv)*w3v
		cw := bind + "_wt"
		lines := []string{
			fmt.Sprintf("%s// rewrite: soft_mux(w, soft_mux(w2,p,e), soft_mux(w3,p,e)) → soft_mux(w*w2+(1-w)*w3, p, e)", indent),
			fmt.Sprintf("%s%s = arith.constant %s : f32", indent, cw, formatF32MLIR(wn)),
			fmt.Sprintf("%s%s = func.call @trin_soft_mux(%s, %s, %s) : (f32, f32, f32) -> f32",
				indent, bind, cw, pure, each),
		}
		consts[cw] = wn
		muxes[bind] = softMuxForm{w: cw, pure: pure, each: each}
		if almostOne(wn) {
			return true, emitSoftMuxIdentity(indent, bind, pure,
				"soft_mux(both-arms, wn=1) → p", "")
		}
		if almostZero(wn) {
			return true, emitSoftMuxIdentity(indent, bind, each,
				"soft_mux(both-arms, wn=0) → e", "")
		}
		st.nested++
		return true, lines
	}
	// Non-const: wn = w*w2 + (1-w)*w3
	c1 := bind + "_one"
	omw := bind + "_omw"
	t0 := bind + "_t0"
	t1 := bind + "_t1"
	cw := bind + "_wt"
	lines := []string{
		fmt.Sprintf("%s// rewrite: soft_mux(w, soft_mux(w2,p,e), soft_mux(w3,p,e)) → soft_mux(w*w2+(1-w)*w3, p, e)", indent),
		fmt.Sprintf("%s%s = arith.constant 1.000000e+00 : f32", indent, c1),
		fmt.Sprintf("%s%s = arith.subf %s, %s : f32", indent, omw, c1, w),
		fmt.Sprintf("%s%s = arith.mulf %s, %s : f32", indent, t0, w, w2),
		fmt.Sprintf("%s%s = arith.mulf %s, %s : f32", indent, t1, omw, w3),
		fmt.Sprintf("%s%s = arith.addf %s, %s : f32", indent, cw, t0, t1),
		fmt.Sprintf("%s%s = func.call @trin_soft_mux(%s, %s, %s) : (f32, f32, f32) -> f32",
			indent, bind, cw, pure, each),
	}
	muxes[bind] = softMuxForm{w: cw, pure: pure, each: each}
	st.nested++
	return true, lines
}

// emitSoftMuxProductWeight rewrites soft_mux(w1, soft_mux(w2,p,e), e) → soft_mux(w1*w2,p,e).
func emitSoftMuxProductWeight(
	indent, bind, w1, w2, pure, each string,
	w1v float64, w1OK bool,
	consts map[string]float64,
	muxes map[string]softMuxForm,
	st *rewriteStats,
	_ bool,
) (bool, []string) {
	w2v, w2OK := consts[w2]
	pureV, pureOK := consts[pure]
	eachV, eachOK := consts[each]
	// Full const product fold.
	if w1OK && w2OK && pureOK && eachOK {
		wn := w1v * w2v
		y := wn*pureV + (1-wn)*eachV
		lines := []string{
			fmt.Sprintf("%s// rewrite: soft_mux(w1, soft_mux(w2,p,e), e) → soft_mux(w1*w2,p,e) const → %.6g", indent, y),
			fmt.Sprintf("%s%s = arith.constant %s : f32", indent, bind, formatF32MLIR(y)),
		}
		consts[bind] = y
		st.nested++
		return true, lines
	}
	// Const weight product, non-const arms → soft_mux(const_prod, p, e)
	if w1OK && w2OK {
		wn := w1v * w2v
		cw := bind + "_wp"
		lines := []string{
			fmt.Sprintf("%s// rewrite: soft_mux(w1, soft_mux(w2,p,e), e) → soft_mux(w1*w2, p, e)", indent),
			fmt.Sprintf("%s%s = arith.constant %s : f32", indent, cw, formatF32MLIR(wn)),
			fmt.Sprintf("%s%s = func.call @trin_soft_mux(%s, %s, %s) : (f32, f32, f32) -> f32",
				indent, bind, cw, pure, each),
		}
		consts[cw] = wn
		muxes[bind] = softMuxForm{w: cw, pure: pure, each: each}
		// If weight is 0/1, leave as soft_mux (subsequent pass not available; try local fold)
		if almostOne(wn) {
			return true, emitSoftMuxIdentity(indent, bind, pure,
				"soft_mux(w1*w2=1, soft_mux(...), e) → p", "")
		}
		if almostZero(wn) {
			return true, emitSoftMuxIdentity(indent, bind, each,
				"soft_mux(w1*w2=0, soft_mux(...), e) → e", "")
		}
		st.nested++
		return true, lines
	}
	// Non-const weights: emit mulf then soft_mux.
	cw := bind + "_wp"
	lines := []string{
		fmt.Sprintf("%s// rewrite: soft_mux(w1, soft_mux(w2,p,e), e) → soft_mux(w1*w2, p, e)", indent),
		fmt.Sprintf("%s%s = arith.mulf %s, %s : f32", indent, cw, w1, w2),
		fmt.Sprintf("%s%s = func.call @trin_soft_mux(%s, %s, %s) : (f32, f32, f32) -> f32",
			indent, bind, cw, pure, each),
	}
	muxes[bind] = softMuxForm{w: cw, pure: pure, each: each}
	st.nested++
	return true, lines
}

// emitSoftMuxEachArmNest rewrites soft_mux(w1, p, soft_mux(w2,p,e)) → soft_mux(w1+(1-w1)*w2, p, e).
func emitSoftMuxEachArmNest(
	indent, bind, w1, pure, w2, each string,
	w1v float64, w1OK bool,
	consts map[string]float64,
	muxes map[string]softMuxForm,
	st *rewriteStats,
) (bool, []string) {
	w2v, w2OK := consts[w2]
	pureV, pureOK := consts[pure]
	eachV, eachOK := consts[each]
	if w1OK && w2OK && pureOK && eachOK {
		wn := w1v + (1-w1v)*w2v
		y := wn*pureV + (1-wn)*eachV
		lines := []string{
			fmt.Sprintf("%s// rewrite: soft_mux(w1, p, soft_mux(w2,p,e)) → soft_mux(w1+(1-w1)*w2,p,e) const → %.6g", indent, y),
			fmt.Sprintf("%s%s = arith.constant %s : f32", indent, bind, formatF32MLIR(y)),
		}
		consts[bind] = y
		st.nested++
		return true, lines
	}
	if w1OK && w2OK {
		wn := w1v + (1-w1v)*w2v
		cw := bind + "_we"
		lines := []string{
			fmt.Sprintf("%s// rewrite: soft_mux(w1, p, soft_mux(w2,p,e)) → soft_mux(w1+(1-w1)*w2, p, e)", indent),
			fmt.Sprintf("%s%s = arith.constant %s : f32", indent, cw, formatF32MLIR(wn)),
			fmt.Sprintf("%s%s = func.call @trin_soft_mux(%s, %s, %s) : (f32, f32, f32) -> f32",
				indent, bind, cw, pure, each),
		}
		consts[cw] = wn
		muxes[bind] = softMuxForm{w: cw, pure: pure, each: each}
		st.nested++
		return true, lines
	}
	// Non-const: wn = w1 + (1-w1)*w2
	c1 := bind + "_one"
	omw := bind + "_omw"
	tw := bind + "_tw"
	cw := bind + "_we"
	lines := []string{
		fmt.Sprintf("%s// rewrite: soft_mux(w1, p, soft_mux(w2,p,e)) → soft_mux(w1+(1-w1)*w2, p, e)", indent),
		fmt.Sprintf("%s%s = arith.constant 1.000000e+00 : f32", indent, c1),
		fmt.Sprintf("%s%s = arith.subf %s, %s : f32", indent, omw, c1, w1),
		fmt.Sprintf("%s%s = arith.mulf %s, %s : f32", indent, tw, omw, w2),
		fmt.Sprintf("%s%s = arith.addf %s, %s : f32", indent, cw, w1, tw),
		fmt.Sprintf("%s%s = func.call @trin_soft_mux(%s, %s, %s) : (f32, f32, f32) -> f32",
			indent, bind, cw, pure, each),
	}
	muxes[bind] = softMuxForm{w: cw, pure: pure, each: each}
	st.nested++
	return true, lines
}

// emitSoftMuxIdentity rewrites a soft_mux result to `val` via addf-with-0
// (keeps SSA shape used by older fold tests). For vector results, n is "4"/"8"/"16".
func emitSoftMuxIdentity(indent, bind, val, note, vecN string) []string {
	z := bind + "_foldz"
	out := []string{
		fmt.Sprintf("%s// rewrite: %s", indent, note),
	}
	if vecN == "" {
		out = append(out,
			fmt.Sprintf("%s%s = arith.constant 0.000000e+00 : f32", indent, z),
			fmt.Sprintf("%s%s = arith.addf %s, %s : f32", indent, bind, val, z),
		)
		return out
	}
	// vector: build zero splat then add
	zt := fmt.Sprintf("vector<%sxf32>", vecN)
	zs := bind + "_zscalar"
	out = append(out,
		fmt.Sprintf("%s%s = arith.constant 0.000000e+00 : f32", indent, zs),
		fmt.Sprintf("%s%s = vector.broadcast %s : f32 to %s", indent, z, zs, zt),
		fmt.Sprintf("%s%s = arith.addf %s, %s : %s", indent, bind, val, z, zt),
	)
	return out
}

func formatF32MLIR(v float64) string {
	// Match emit-mlir style scientific notation for common cases.
	if v == 0 {
		return "0.000000e+00"
	}
	return fmt.Sprintf("%e", v)
}

func almostOne(v float64) bool     { return v > 0.999 && v < 1.001 }
func almostZero(v float64) bool    { return v > -0.001 && v < 0.001 }
func almostEqual(a, b float64) bool { return almostZero(a - b) }

// Bitplane / plane algebraic patterns (v0.33).
var reConstI32 = regexp.MustCompile(
	`(?P<name>%[A-Za-z0-9_.]+)\s*=\s*arith\.constant\s+(?P<val>-?\d+)\s*:\s*i32`,
)

var rePlaneAnd = regexp.MustCompile(
	`(?P<indent>[ \t]*)(?P<bind>%[A-Za-z0-9_.]+)\s*=\s*func\.call\s+@trin_plane_and\(\s*(?P<a>%[A-Za-z0-9_.]+)\s*,\s*(?P<b>%[A-Za-z0-9_.]+)\s*\)\s*:\s*\(i32,\s*i32\)\s*->\s*i32`,
)

var rePlaneOr = regexp.MustCompile(
	`(?P<indent>[ \t]*)(?P<bind>%[A-Za-z0-9_.]+)\s*=\s*func\.call\s+@trin_plane_or\(\s*(?P<a>%[A-Za-z0-9_.]+)\s*,\s*(?P<b>%[A-Za-z0-9_.]+)\s*\)\s*:\s*\(i32,\s*i32\)\s*->\s*i32`,
)

var rePlaneXor = regexp.MustCompile(
	`(?P<indent>[ \t]*)(?P<bind>%[A-Za-z0-9_.]+)\s*=\s*func\.call\s+@trin_plane_xor\(\s*(?P<a>%[A-Za-z0-9_.]+)\s*,\s*(?P<b>%[A-Za-z0-9_.]+)\s*\)\s*:\s*\(i32,\s*i32\)\s*->\s*i32`,
)

var rePlanePop = regexp.MustCompile(
	`(?P<indent>[ \t]*)(?P<bind>%[A-Za-z0-9_.]+)\s*=\s*func\.call\s+@trin_plane_popcount\(\s*(?P<a>%[A-Za-z0-9_.]+)\s*\)\s*:\s*\(i32\)\s*->\s*i32`,
)

// rewriteBitplaneFolds applies trit plane algebraic identities (v0.33)
// plus product/absorption folds (v0.35) and multi-word GEMM folds (v0.38).
//   plane_and(x,x) → x; plane_and(x,0)|and(0,x) → 0
//   plane_or(x,x) → x; plane_or(x,0)|or(0,x) → x
//   plane_xor(x,x) → 0
//   plane_popcount(0) → 0; plane_popcount(const) → ctpop(const)
//   plane_and(and(a,b), a) → and(a,b); plane_and(a, and(a,b)) → and(a,b)
//   plane_or(or(a,b), a) → or(a,b); plane_or(a, or(a,b)) → or(a,b)
//   plane_and(a, or(a,b)) → a; plane_or(a, and(a,b)) → a
//   plane_and(and(a,b), or(a,b)) → and(a,b); plane_or(and(a,b), or(a,b)) → or(a,b)
//   trit_dot_planes_word zero-arm / same-vector / const folds (v0.38)
//   trit_dot_planes_x4 zero-word / all-zero / sum folds (v0.38)
func rewriteBitplaneFolds(mlir string, st *rewriteStats) string {
	consts := map[string]int64{}
	bins := map[string]planeBinForm{}
	lines := strings.Split(mlir, "\n")
	var out []string
	emitI32Const := func(indent, bind string, v int64, note string) {
		out = append(out, fmt.Sprintf("%s// rewrite: %s", indent, note))
		out = append(out, fmt.Sprintf("%s%s = arith.constant %d : i32", indent, bind, v))
		consts[bind] = v
		st.bitplane++
	}
	emitI32Alias := func(indent, bind, src, note string, nested bool) {
		// Keep a defined SSA: bind = src + 0
		z := bind + "_z"
		out = append(out, fmt.Sprintf("%s// rewrite: %s", indent, note))
		out = append(out, fmt.Sprintf("%s%s = arith.constant 0 : i32", indent, z))
		out = append(out, fmt.Sprintf("%s%s = arith.addi %s, %s : i32", indent, bind, src, z))
		if v, ok := consts[src]; ok {
			consts[bind] = v
		}
		if form, ok := bins[src]; ok {
			bins[bind] = form
		}
		if nested {
			st.nested++
		} else {
			st.bitplane++
		}
	}
	// planeOperandEquals reports whether ssa x equals operand o, either by SSA id or
	// as one of the children of a known bin form of o.
	sharesOperand := func(x, o string) bool {
		if x == o {
			return true
		}
		if f, ok := bins[o]; ok {
			return f.a == x || f.b == x
		}
		return false
	}
	for _, line := range lines {
		if m := reConstI32.FindStringSubmatch(line); m != nil {
			g := reGroups(reConstI32, m)
			var v int64
			if _, err := fmt.Sscanf(g["val"], "%d", &v); err == nil {
				consts[g["name"]] = v
			}
			out = append(out, line)
			continue
		}
		if m := rePlaneAnd.FindStringSubmatch(line); m != nil {
			g := reGroups(rePlaneAnd, m)
			a, b := g["a"], g["b"]
			av, aOK := consts[a]
			bv, bOK := consts[b]
			switch {
			case a == b:
				emitI32Alias(g["indent"], g["bind"], a, "plane_and(x,x) → x", false)
			case aOK && av == 0:
				emitI32Const(g["indent"], g["bind"], 0, "plane_and(0,x) → 0")
			case bOK && bv == 0:
				emitI32Const(g["indent"], g["bind"], 0, "plane_and(x,0) → 0")
			case aOK && bOK:
				emitI32Const(g["indent"], g["bind"], av&bv, "plane_and(const) fold")
			// absorption / nested product (v0.35)
			// plane_and(and(x,y), x|y) → and(x,y)
			case bins[a].op == "and" && (sharesOperand(bins[a].a, b) || sharesOperand(bins[a].b, b) ||
				(bins[b].op == "or" && samePair(bins[a], bins[b]))):
				emitI32Alias(g["indent"], g["bind"], a, "plane_and(and(a,b), …) absorb → and(a,b)", true)
			case bins[b].op == "and" && (sharesOperand(bins[b].a, a) || sharesOperand(bins[b].b, a) ||
				(bins[a].op == "or" && samePair(bins[a], bins[b]))):
				emitI32Alias(g["indent"], g["bind"], b, "plane_and(…, and(a,b)) absorb → and(a,b)", true)
			// plane_and(a, or(a,b)) → a
			case bins[b].op == "or" && (bins[b].a == a || bins[b].b == a):
				emitI32Alias(g["indent"], g["bind"], a, "plane_and(a, or(a,b)) → a", true)
			case bins[a].op == "or" && (bins[a].a == b || bins[a].b == b):
				emitI32Alias(g["indent"], g["bind"], b, "plane_and(or(a,b), a) → a", true)
			default:
				bins[g["bind"]] = planeBinForm{op: "and", a: a, b: b}
				out = append(out, line)
			}
			continue
		}
		if m := rePlaneOr.FindStringSubmatch(line); m != nil {
			g := reGroups(rePlaneOr, m)
			a, b := g["a"], g["b"]
			av, aOK := consts[a]
			bv, bOK := consts[b]
			switch {
			case a == b:
				emitI32Alias(g["indent"], g["bind"], a, "plane_or(x,x) → x", false)
			case aOK && av == 0:
				emitI32Alias(g["indent"], g["bind"], b, "plane_or(0,x) → x", false)
			case bOK && bv == 0:
				emitI32Alias(g["indent"], g["bind"], a, "plane_or(x,0) → x", false)
			case aOK && bOK:
				emitI32Const(g["indent"], g["bind"], av|bv, "plane_or(const) fold")
			// absorption / nested product (v0.35)
			case bins[a].op == "or" && (sharesOperand(bins[a].a, b) || sharesOperand(bins[a].b, b) ||
				(bins[b].op == "and" && samePair(bins[a], bins[b]))):
				emitI32Alias(g["indent"], g["bind"], a, "plane_or(or(a,b), …) absorb → or(a,b)", true)
			case bins[b].op == "or" && (sharesOperand(bins[b].a, a) || sharesOperand(bins[b].b, a) ||
				(bins[a].op == "and" && samePair(bins[a], bins[b]))):
				emitI32Alias(g["indent"], g["bind"], b, "plane_or(…, or(a,b)) absorb → or(a,b)", true)
			// plane_or(a, and(a,b)) → a
			case bins[b].op == "and" && (bins[b].a == a || bins[b].b == a):
				emitI32Alias(g["indent"], g["bind"], a, "plane_or(a, and(a,b)) → a", true)
			case bins[a].op == "and" && (bins[a].a == b || bins[a].b == b):
				emitI32Alias(g["indent"], g["bind"], b, "plane_or(and(a,b), a) → a", true)
			default:
				bins[g["bind"]] = planeBinForm{op: "or", a: a, b: b}
				out = append(out, line)
			}
			continue
		}
		if m := rePlaneXor.FindStringSubmatch(line); m != nil {
			g := reGroups(rePlaneXor, m)
			a, b := g["a"], g["b"]
			av, aOK := consts[a]
			bv, bOK := consts[b]
			switch {
			case a == b:
				emitI32Const(g["indent"], g["bind"], 0, "plane_xor(x,x) → 0")
			case aOK && av == 0:
				emitI32Alias(g["indent"], g["bind"], b, "plane_xor(0,x) → x", false)
			case bOK && bv == 0:
				emitI32Alias(g["indent"], g["bind"], a, "plane_xor(x,0) → x", false)
			case aOK && bOK:
				emitI32Const(g["indent"], g["bind"], av^bv, "plane_xor(const) fold")
			default:
				bins[g["bind"]] = planeBinForm{op: "xor", a: a, b: b}
				out = append(out, line)
			}
			continue
		}
		if m := rePlanePop.FindStringSubmatch(line); m != nil {
			g := reGroups(rePlanePop, m)
			a := g["a"]
			if av, ok := consts[a]; ok {
				// popcount of known i32 (use 32-bit bits.OnesCount)
				pc := int64(bitsOnesCount32(uint32(av)))
				note := "plane_popcount(const) fold"
				if av == 0 {
					note = "plane_popcount(0) → 0"
				}
				emitI32Const(g["indent"], g["bind"], pc, note)
			} else {
				out = append(out, line)
			}
			continue
		}
		// v0.38 multi-word GEMM / plane-dot folds
		if m := reDotPlanesWord.FindStringSubmatch(line); m != nil {
			g := reGroups(reDotPlanesWord, m)
			ap, an, bp, bn := g["ap"], g["an"], g["bp"], g["bn"]
			apv, apOK := consts[ap]
			anv, anOK := consts[an]
			bpv, bpOK := consts[bp]
			bnv, bnOK := consts[bn]
			switch {
			// all-zero planes or zero vector arms → 0
			case (apOK && anOK && apv == 0 && anv == 0) || (bpOK && bnOK && bpv == 0 && bnv == 0):
				emitI32Const(g["indent"], g["bind"], 0, "trit_dot_planes_word(zero) → 0")
			// zero pos/neg halves on either side
			case apOK && bpOK && anOK && bnOK:
				pos := (apv & bpv) | (anv & bnv)
				neg := (apv & bnv) | (anv & bpv)
				pc := int64(bitsOnesCount32(uint32(pos))) - int64(bitsOnesCount32(uint32(neg)))
				emitI32Const(g["indent"], g["bind"], pc, "trit_dot_planes_word(const) fold")
			default:
				out = append(out, line)
			}
			continue
		}
		if m := reDotPlanesX4.FindStringSubmatch(line); m != nil {
			g := reGroups(reDotPlanesX4, m)
			args := []string{}
			for _, part := range strings.Split(g["args"], ",") {
				args = append(args, strings.TrimSpace(part))
			}
			if len(args) == 16 {
				allConst := true
				var sum int64
				allZero := true
				wordPC := make([]int64, 4)
				for i := 0; i < 4; i++ {
					ap, an, bp, bn := args[i*4+0], args[i*4+1], args[i*4+2], args[i*4+3]
					apv, apOK := consts[ap]
					anv, anOK := consts[an]
					bpv, bpOK := consts[bp]
					bnv, bnOK := consts[bn]
					if !(apOK && anOK && bpOK && bnOK) {
						allConst = false
						break
					}
					if !(apv == 0 && anv == 0 && bpv == 0 && bnv == 0) {
						allZero = false
					}
					pos := (apv & bpv) | (anv & bnv)
					neg := (apv & bnv) | (anv & bpv)
					pc := int64(bitsOnesCount32(uint32(pos))) - int64(bitsOnesCount32(uint32(neg)))
					wordPC[i] = pc
					sum += pc
				}
				if allConst {
					// v0.40 multi-word schedule: 4 equal words → 4*word note
					eq4 := wordPC[0] == wordPC[1] && wordPC[1] == wordPC[2] && wordPC[2] == wordPC[3]
					// 2+2 scheduled pairs (w0==w1 && w2==w3)
					pair := wordPC[0] == wordPC[1] && wordPC[2] == wordPC[3]
					note := "trit_dot_planes_x4(const) fold"
					if allZero {
						note = "trit_dot_planes_x4(zero) → 0"
					} else if eq4 {
						note = fmt.Sprintf("trit_dot_planes_x4(schedule equal-words) → 4*%d", wordPC[0])
					} else if pair {
						note = fmt.Sprintf("trit_dot_planes_x4(schedule 2+2) → 2*(%d+%d)", wordPC[0], wordPC[2])
					}
					emitI32Const(g["indent"], g["bind"], sum, note)
					continue
				}
				// v0.40 schedule: identical SSA word repeated 4× → 4 * word
				sameWord := true
				for i := 1; i < 4 && sameWord; i++ {
					for j := 0; j < 4; j++ {
						if args[i*4+j] != args[j] {
							sameWord = false
							break
						}
					}
				}
				if sameWord {
					// fold to 4 * single-word helper result via temporary mul after call
					// const path already handled; non-const: still emit scheduled rewrite note + 4*call
					indent, bind := g["indent"], g["bind"]
					tmp := bind + "_w"
					out = append(out, fmt.Sprintf("%s// rewrite: trit_dot_planes_x4(schedule equal-words) → 4*word", indent))
					out = append(out, fmt.Sprintf("%s%s = func.call @trin_trit_dot_planes_word(%s, %s, %s, %s) : (i32, i32, i32, i32) -> i32",
						indent, tmp, args[0], args[1], args[2], args[3]))
					c4 := bind + "_c4"
					out = append(out, fmt.Sprintf("%s%s = arith.constant 4 : i32", indent, c4))
					out = append(out, fmt.Sprintf("%s%s = arith.muli %s, %s : i32", indent, bind, tmp, c4))
					st.bitplane++
					continue
				}
				// partial: if every word has a zero A or B vector, sum=0
				zeroWords := 0
				for i := 0; i < 4; i++ {
					ap, an, bp, bn := args[i*4+0], args[i*4+1], args[i*4+2], args[i*4+3]
					apv, apOK := consts[ap]
					anv, anOK := consts[an]
					bpv, bpOK := consts[bp]
					bnv, bnOK := consts[bn]
					if (apOK && anOK && apv == 0 && anv == 0) || (bpOK && bnOK && bpv == 0 && bnv == 0) {
						zeroWords++
					}
				}
				if zeroWords == 4 {
					emitI32Const(g["indent"], g["bind"], 0, "trit_dot_planes_x4(zero-words) → 0")
					continue
				}
			}
			out = append(out, line)
			continue
		}
		out = append(out, line)
	}
	return strings.Join(out, "\n")
}


var reDotPlanesWord = regexp.MustCompile(
	`(?P<indent>[ \t]*)(?P<bind>%[A-Za-z0-9_.]+)\s*=\s*func\.call\s+@trin_trit_dot_planes_word\(\s*(?P<ap>%[A-Za-z0-9_.]+)\s*,\s*(?P<an>%[A-Za-z0-9_.]+)\s*,\s*(?P<bp>%[A-Za-z0-9_.]+)\s*,\s*(?P<bn>%[A-Za-z0-9_.]+)\s*\)\s*:\s*\(i32,\s*i32,\s*i32,\s*i32\)\s*->\s*i32`,
)

var reDotPlanesX4 = regexp.MustCompile(
	`(?P<indent>[ \t]*)(?P<bind>%[A-Za-z0-9_.]+)\s*=\s*func\.call\s+@trin_trit_dot_planes_x4\((?P<args>[^)]*)\)\s*:\s*\([^)]*\)\s*->\s*i32`,
)

// samePair reports whether two binary plane ops share the same unordered operands.
func samePair(a, b planeBinForm) bool {
	return (a.a == b.a && a.b == b.b) || (a.a == b.b && a.b == b.a)
}

// bitsOnesCount32 is a tiny stdlib-free popcount for const folding.
func bitsOnesCount32(x uint32) int {
	// Hacker's Delight / SWAR
	x = x - ((x >> 1) & 0x55555555)
	x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
	return int((((x + (x >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24)
}

// Span/vector algebraic patterns (v0.31).
var reVecBroadcast = regexp.MustCompile(
	`(?P<indent>[ \t]*)(?P<bind>%[A-Za-z0-9_.]+)\s*=\s*vector\.broadcast\s+(?P<src>%[A-Za-z0-9_.]+)\s*:\s*f32\s+to\s+vector<(?P<n>\d+)xf32>`,
)

var reVecFromElems = regexp.MustCompile(
	`(?P<indent>[ \t]*)(?P<bind>%[A-Za-z0-9_.]+)\s*=\s*vector\.from_elements\s+(?P<args>[^:]+)\s*:\s*vector<(?P<n>\d+)xf32>`,
)

// Go RE2 has no backrefs; match lanes loosely and validate n later when needed.
var reSpanAddV = regexp.MustCompile(
	`(?P<indent>[ \t]*)(?P<bind>%[A-Za-z0-9_.]+)\s*=\s*func\.call\s+@(?P<fn>trin_span_add_v(?P<n>4|8|16))\(\s*(?P<a>%[A-Za-z0-9_.]+)\s*,\s*(?P<b>%[A-Za-z0-9_.]+)\s*\)\s*:\s*\(vector<\d+xf32>\s*,\s*vector<\d+xf32>\)\s*->\s*vector<\d+xf32>`,
)

var reSpanScaleV = regexp.MustCompile(
	`(?P<indent>[ \t]*)(?P<bind>%[A-Za-z0-9_.]+)\s*=\s*func\.call\s+@(?P<fn>trin_span_scale_v(?P<n>4|8|16))\(\s*(?P<s>%[A-Za-z0-9_.]+)\s*,\s*(?P<a>%[A-Za-z0-9_.]+)\s*\)\s*:\s*\(f32\s*,\s*vector<\d+xf32>\)\s*->\s*vector<\d+xf32>`,
)

var reSumV = regexp.MustCompile(
	`(?P<indent>[ \t]*)(?P<bind>%[A-Za-z0-9_.]+)\s*=\s*func\.call\s+@(?P<fn>trin_sum_v(?P<n>4|8|16))\(\s*(?P<a>%[A-Za-z0-9_.]+)\s*\)\s*:\s*\(vector<\d+xf32>\)\s*->\s*f32`,
)

var reDotV = regexp.MustCompile(
	`(?P<indent>[ \t]*)(?P<bind>%[A-Za-z0-9_.]+)\s*=\s*func\.call\s+@(?P<fn>trin_dot_v(?P<n>4|8|16))\(\s*(?P<a>%[A-Za-z0-9_.]+)\s*,\s*(?P<b>%[A-Za-z0-9_.]+)\s*\)\s*:\s*\(vector<\d+xf32>\s*,\s*vector<\d+xf32>\)\s*->\s*f32`,
)

// rewriteSpanVectorFolds applies span/vector algebraic identities after dialect
// expansion (v0.31). Tracks f32 constants and zero vectors (broadcast/from_elements).
func rewriteSpanVectorFolds(mlir string, st *rewriteStats) string {
	consts := map[string]float64{}
	zeroVec := map[string]bool{}
	lines := strings.Split(mlir, "\n")
	var out []string
	for _, line := range lines {
		// Track scalar constants.
		if m := reConstF32.FindStringSubmatch(line); m != nil {
			g := reGroups(reConstF32, m)
			var v float64
			if _, err := fmt.Sscanf(g["val"], "%f", &v); err == nil {
				consts[g["name"]] = v
			}
			out = append(out, line)
			continue
		}
		// Track zero vectors from broadcast.
		if m := reVecBroadcast.FindStringSubmatch(line); m != nil {
			g := reGroups(reVecBroadcast, m)
			if v, ok := consts[g["src"]]; ok && almostZero(v) {
				zeroVec[g["bind"]] = true
			}
			out = append(out, line)
			continue
		}
		// Track zero vectors from all-zero from_elements.
		if m := reVecFromElems.FindStringSubmatch(line); m != nil {
			g := reGroups(reVecFromElems, m)
			args := strings.Split(g["args"], ",")
			allZero := len(args) > 0
			for _, a := range args {
				name := strings.TrimSpace(a)
				v, ok := consts[name]
				if !ok || !almostZero(v) {
					allZero = false
					break
				}
			}
			if allZero {
				zeroVec[g["bind"]] = true
			}
			out = append(out, line)
			continue
		}

		// span_add_vN(x,0) / span_add_vN(0,x) → x
		if m := reSpanAddV.FindStringSubmatch(line); m != nil {
			g := reGroups(reSpanAddV, m)
			switch {
			case zeroVec[g["b"]]:
				for _, l := range emitSoftMuxIdentity(g["indent"], g["bind"], g["a"],
					"span_add(x,0) → x", g["n"]) {
					out = append(out, l)
				}
				st.spanAlg++
				// result equals a; propagate zero only if a is zero
				if zeroVec[g["a"]] {
					zeroVec[g["bind"]] = true
				}
				continue
			case zeroVec[g["a"]]:
				for _, l := range emitSoftMuxIdentity(g["indent"], g["bind"], g["b"],
					"span_add(0,x) → x", g["n"]) {
					out = append(out, l)
				}
				st.spanAlg++
				if zeroVec[g["b"]] {
					zeroVec[g["bind"]] = true
				}
				continue
			}
			out = append(out, line)
			continue
		}

		// span_scale_vN(1,x) → x; span_scale_vN(0,x) → 0
		if m := reSpanScaleV.FindStringSubmatch(line); m != nil {
			g := reGroups(reSpanScaleV, m)
			if s, ok := consts[g["s"]]; ok {
				if almostOne(s) {
					for _, l := range emitSoftMuxIdentity(g["indent"], g["bind"], g["a"],
						"span_scale(1,x) → x", g["n"]) {
						out = append(out, l)
					}
					st.spanAlg++
					if zeroVec[g["a"]] {
						zeroVec[g["bind"]] = true
					}
					continue
				}
				if almostZero(s) {
					// emit zero vector via broadcast
					zs := g["bind"] + "_zscalar"
					zb := g["bind"] + "_zvec"
					zt := fmt.Sprintf("vector<%sxf32>", g["n"])
					out = append(out, fmt.Sprintf("%s// rewrite: span_scale(0,x) → 0", g["indent"]))
					out = append(out, fmt.Sprintf("%s%s = arith.constant 0.000000e+00 : f32", g["indent"], zs))
					out = append(out, fmt.Sprintf("%s%s = vector.broadcast %s : f32 to %s", g["indent"], zb, zs, zt))
					// identity via add with zero so bind is preserved as the zero vector
					out = append(out, fmt.Sprintf("%s%s = arith.addf %s, %s : %s", g["indent"], g["bind"], zb, zb, zt))
					zeroVec[g["bind"]] = true
					st.spanAlg++
					continue
				}
			}
			// scale of a known zero vector is zero for any s
			if zeroVec[g["a"]] {
				zs := g["bind"] + "_zscalar"
				zb := g["bind"] + "_zvec"
				zt := fmt.Sprintf("vector<%sxf32>", g["n"])
				out = append(out, fmt.Sprintf("%s// rewrite: span_scale(s,0) → 0", g["indent"]))
				out = append(out, fmt.Sprintf("%s%s = arith.constant 0.000000e+00 : f32", g["indent"], zs))
				out = append(out, fmt.Sprintf("%s%s = vector.broadcast %s : f32 to %s", g["indent"], zb, zs, zt))
				out = append(out, fmt.Sprintf("%s%s = arith.addf %s, %s : %s", g["indent"], g["bind"], zb, zb, zt))
				zeroVec[g["bind"]] = true
				st.spanAlg++
				continue
			}
			out = append(out, line)
			continue
		}

		// sum_vN(0) → 0
		if m := reSumV.FindStringSubmatch(line); m != nil {
			g := reGroups(reSumV, m)
			if zeroVec[g["a"]] {
				out = append(out, fmt.Sprintf("%s// rewrite: sum(0) → 0", g["indent"]))
				out = append(out, fmt.Sprintf("%s%s = arith.constant 0.000000e+00 : f32",
					g["indent"], g["bind"]))
				consts[g["bind"]] = 0
				st.spanAlg++
				continue
			}
			out = append(out, line)
			continue
		}

		// dot_vN(0,x) / dot_vN(x,0) → 0
		if m := reDotV.FindStringSubmatch(line); m != nil {
			g := reGroups(reDotV, m)
			if zeroVec[g["a"]] || zeroVec[g["b"]] {
				out = append(out, fmt.Sprintf("%s// rewrite: dot(0,x)|dot(x,0) → 0", g["indent"]))
				out = append(out, fmt.Sprintf("%s%s = arith.constant 0.000000e+00 : f32",
					g["indent"], g["bind"]))
				consts[g["bind"]] = 0
				st.spanAlg++
				continue
			}
			out = append(out, line)
			continue
		}

		out = append(out, line)
	}
	return strings.Join(out, "\n")
}

// Pure SSA binding for CSE (v0.27).
var rePureBind = regexp.MustCompile(
	`(?P<indent>[ \t]*)(?P<bind>%[A-Za-z0-9_.]+)\s*=\s*(?P<rhs>\S.*)$`,
)

var reSSAName = regexp.MustCompile(`%[A-Za-z0-9_.]+`)

func isPureRHS(rhs string) bool {
	rhs = strings.TrimSpace(rhs)
	switch {
	case strings.HasPrefix(rhs, "arith.constant "):
		return true
	case strings.HasPrefix(rhs, "arith.addf "),
		strings.HasPrefix(rhs, "arith.subf "),
		strings.HasPrefix(rhs, "arith.mulf "):
		return true
	case strings.HasPrefix(rhs, "vector.from_elements "):
		return true
	case strings.Contains(rhs, "func.call @trin_soft_mux("),
		strings.Contains(rhs, "func.call @trin_soft_mux_v4("),
		strings.Contains(rhs, "func.call @trin_soft_mux_v8("),
		strings.Contains(rhs, "func.call @trin_soft_mux_v16("):
		return true
	default:
		return false
	}
}

func normalizeRHS(rhs string) string {
	return strings.Join(strings.Fields(strings.TrimSpace(rhs)), " ")
}

func resolveAlias(aliases map[string]string, name string) string {
	seen := map[string]bool{}
	for {
		if seen[name] {
			return name
		}
		seen[name] = true
		next, ok := aliases[name]
		if !ok {
			return name
		}
		name = next
	}
}

// rewriteNamesInLine rewrites %ssa uses with alias resolution (longest-safe: token regex).
func rewriteNamesInLine(line string, aliases map[string]string) string {
	if len(aliases) == 0 {
		return line
	}
	return reSSAName.ReplaceAllStringFunc(line, func(name string) string {
		return resolveAlias(aliases, name)
	})
}

// rewriteCSE commons pure SSA values within each func.func body (v0.27).
// Multi-use soft_mux is included: identical soft_mux calls reuse the first bind.
func rewriteCSE(mlir string, st *rewriteStats) string {
	lines := strings.Split(mlir, "\n")
	var out []string
	// per-function state
	seen := map[string]string{} // rhs key → first bind
	aliases := map[string]string{}
	inFunc := false

	flushScope := func() {
		seen = map[string]string{}
		aliases = map[string]string{}
		inFunc = false
	}

	for _, line := range lines {
		trim := strings.TrimSpace(line)
		if strings.HasPrefix(trim, "func.func ") {
			flushScope()
			inFunc = true
			out = append(out, line)
			continue
		}
		// End of a top-level-ish function body (indented closing brace of func).
		if inFunc && (trim == "}" || strings.HasPrefix(trim, "} ")) {
			// keep processing this line with aliases for return rewrites first
			line = rewriteNamesInLine(line, aliases)
			out = append(out, line)
			flushScope()
			continue
		}
		if !inFunc {
			out = append(out, line)
			continue
		}

		line = rewriteNamesInLine(line, aliases)

		// skip pure comments / notes
		if strings.HasPrefix(strings.TrimSpace(line), "//") {
			out = append(out, line)
			continue
		}

		m := rePureBind.FindStringSubmatch(line)
		if m == nil {
			out = append(out, line)
			continue
		}
		g := reGroups(rePureBind, m)
		rhs := g["rhs"]
		if !isPureRHS(rhs) {
			out = append(out, line)
			continue
		}
		key := normalizeRHS(rhs)
		bind := g["bind"]
		if prev, ok := seen[key]; ok && prev != bind {
			aliases[bind] = prev
			out = append(out, fmt.Sprintf("%s// rewrite: cse → %s", g["indent"], prev))
			st.cse++
			// Drop the duplicate definition; uses are rewritten to prev.
			continue
		}
		seen[key] = bind
		out = append(out, line)
	}
	return strings.Join(out, "\n")
}

func stampRewriteHeader(mlir string, st rewriteStats) string {
	out := mlir
	// Prefer keeping already-0.50 modules; otherwise bump legacy versions.
	if !strings.Contains(out, `trin.version = "0.81"`) {
		for _, ver := range []string{"0.48", "0.47", "0.46", "0.45", "0.44", "0.43", "0.42", "0.41", "0.40", "0.39", "0.38", "0.37", "0.36", "0.35", "0.34", "0.33", "0.32", "0.31", "0.30", "0.29", "0.28", "0.27", "0.26", "0.25", "0.24", "0.23", "0.22", "0.21"} {
			old := fmt.Sprintf(`trin.version = "%s"`, ver)
			if strings.Contains(out, old) {
				out = strings.Replace(out, old, `trin.version = "0.81"`, 1)
				break
			}
		}
	}
	// Bump emitter banners if present
	for _, old := range []string{
		"// Generated by Trin v0.24 emit-mlir (portable dialect text; no mlir-opt required)\n",
		"// Generated by Trin v0.25 emit-mlir (portable dialect text; no mlir-opt required)\n",
		"// Generated by Trin v0.26 emit-mlir (portable dialect text; no mlir-opt required)\n",
		"// Generated by Trin v0.27 emit-mlir (portable dialect text; no mlir-opt required)\n",
		"// Generated by Trin v0.28 emit-mlir (portable dialect text; no mlir-opt required)\n",
		"// Generated by Trin v0.29 emit-mlir (portable dialect text; no mlir-opt required)\n",
		"// Generated by Trin v0.30 emit-mlir (portable dialect text; no mlir-opt required)\n",
		"// Generated by Trin v0.31 emit-mlir (portable dialect text; no mlir-opt required)\n",
		"// Generated by Trin v0.32 emit-mlir (portable dialect text; no mlir-opt required)\n",
		"// Generated by Trin v0.33 emit-mlir (portable dialect text; no mlir-opt required)\n",
		"// Generated by Trin v0.34 emit-mlir (portable dialect text; no mlir-opt required)\n",
		"// Generated by Trin v0.35 emit-mlir (portable dialect text; no mlir-opt required)\n",
		"// Generated by Trin v0.36 emit-mlir (portable dialect text; no mlir-opt required)\n",
		"// Generated by Trin v0.37 emit-mlir (portable dialect text; no mlir-opt required)\n",
		"// Generated by Trin v0.38 emit-mlir (portable dialect text; no mlir-opt required)\n",
		"// Generated by Trin v0.39 emit-mlir (portable dialect text; no mlir-opt required)\n",
		"// Generated by Trin v0.40 emit-mlir (portable dialect text; no mlir-opt required)\n",
		"// Generated by Trin v0.41 emit-mlir (portable dialect text; no mlir-opt required)\n",
		"// Generated by Trin v0.42 emit-mlir (portable dialect text; no mlir-opt required)\n",
		"// Generated by Trin v0.43 emit-mlir (portable dialect text; no mlir-opt required)\n",
		"// Generated by Trin v0.44 emit-mlir (portable dialect text; no mlir-opt required)\n",
		"// Generated by Trin v0.45 emit-mlir (portable dialect text; no mlir-opt required)\n",
		"// Generated by Trin v0.46 emit-mlir (portable dialect text; no mlir-opt required)\n",
		"// Generated by Trin v0.47 emit-mlir (portable dialect text; no mlir-opt required)\n",
		"// Generated by Trin v0.48 emit-mlir (portable dialect text; no mlir-opt required)\n",
	} {
		out = strings.Replace(out, old,
			"// Generated by Trin v0.51 emit-mlir (portable dialect text; no mlir-opt required)\n", 1)
	}

	note := fmt.Sprintf("// rewrite: dialect_ops=%d soft_mux_folds=%d strength=%d span_alg=%d bitplane=%d nested=%d cse=%d vwidth=%d (v0.51 portable patterns; no mlir-opt)\n",
		st.dialect, st.softFolds, st.strength, st.spanAlg, st.bitplane, st.nested, st.cse, st.vwidth)
	if strings.Contains(out, "rewrite: dialect_ops=") {
		re := regexp.MustCompile(`// rewrite: dialect_ops=\d+ soft_mux_folds=\d+[^\n]*\n`)
		out = re.ReplaceAllString(out, note)
	} else if i := strings.Index(out, "\n"); i >= 0 {
		out = out[:i+1] + note + out[i+1:]
	} else {
		out = note + out
	}

	if !strings.Contains(out, "Patterns (v0.") {
		catalog := "" +
			"  // Patterns (v0.27–v0.51) — portable rewrites (no mlir-opt):\n" +
			"  //   \"trin.soft_mux\"(w,p,e) → func.call @trin_soft_mux / _v4 / _v8 / _v16\n" +
			"  //   \"trin.vspan_add\"(a,b) : vector<Nxf32> → @trin_span_add_vN (vector-width pick)\n" +
			"  //   \"trin.vsum\"(a) / \"trin.vdot\"(a,b) → @trin_sum_vN / @trin_dot_vN\n" +
			"  //   \"trin.sum\" / \"trin.dot\" / \"trin.span_*\" → scalar span helpers\n" +
			"  //   soft_mux(1,p,e) → p ; soft_mux(0,p,e) → e  (constant fold)\n" +
			"  //   soft_mux(w,x,x) → x ; soft_mux(const) fold ; each=0 → w*p ; pure=0 → (1-w)*e (v0.29 strength)\n" +
			"  //   soft_mux_vN(w,pure,0) → broadcast(w)*pure ; soft_mux_vN(w,0,each) → broadcast(1-w)*each (v0.33)\n" +
			"  //   nested soft_mux product weight: soft_mux(w1, soft_mux(w2,p,e), e) → soft_mux(w1*w2,p,e) (v0.35)\n" +
			"  //   span_add(x,0)|span_add(0,x) → x ; span_scale(1,x) → x ; span_scale(0,x)|scale(s,0) → 0 (v0.31)\n" +
			"  //   sum(0) → 0 ; dot(0,x)|dot(x,0) → 0  (vector tiles)\n" +
			"  //   plane_and(x,x)→x ; and(x,0)→0 ; or(x,x)→x ; or(x,0)→x ; xor(x,x)→0 ; popcount(0)→0 (v0.33)\n" +
			"  //   plane absorb/product: and(a,or(a,b))→a ; or(a,and(a,b))→a ; and(and,or)→and (v0.35)\n" +
			"  //   CSE pure SSA (arith/const/vector.from_elements/soft_mux multi-use)\n" +
			"  //   Hardware VLEN queries: @trin_hw_vlen / @trin_hw_vlen_for_n / @trin_hw_isa_code (runtime v0.30)\n" +
			"  //   Full-program @main dyn tile: sum/dot n∈{4,8,16} → @trin_dyn_sum* / @trin_dyn_dot* (v0.32)\n" +
			"  //   Residual peel: n∉{4,8,16} → max 16/8/4 tiles + scalar residual (v0.34)\n" +
			"  //   Auto residual peel: pointer/counted @trin_sum_f32/@trin_dot_f32 peel 16/8/4 + residual (v0.36)\n" +
			"  //   Store peels: @trin_span_add_f32 / @trin_span_scale_f32 vector 16/8/4 + residual (v0.37)\n" +
			"  //   Multi-level soft_mux trees + multi-word plane GEMM folds (v0.38)\n" +
			"  //   soft_mux(w, soft_mux(w2,p,e), soft_mux(w3,p,e)) → soft_mux(w*w2+(1-w)*w3,p,e) (v0.38)\n" +
			"  //   trit_dot_planes_word/x4 zero-arm / const multi-word folds (v0.38)\n" +
			"  //   Adaptive tile from runtime ISA: @trin_adaptive_tile* (v0.39)\n" +
			"  //   Map/store peels: @trin_span_mul_f32 / @trin_span_axpy_f32 (v0.40)\n" +
			"  //   soft_mux forest product both-arms + trit_dot_planes_x4 schedule equal/2+2 (v0.40)\n" +
			"  //   soft_mux deep forest schedule + each-arm product (v0.41)\n" +
			"  //   Scan/reduce peels: @trin_span_scan_add_f32 / @trin_reduce_add_f32 (v0.41)\n" +
			"  //   Adaptive ISA peel ladder: @trin_adaptive_peel_tile (v0.41)\n" +
"  //   Scan-mul/reduce-mul peels + AdaptivePeelChunks + wide 32-lane ladder (v0.42)\n" +
"  //   Fused scan-reduce + dyn_sum32 + VL>32 (64) ladder (v0.43)\n" +
"  //   Multi-output fused scan-reduce + RDVL host VL + fused-mul probe (v0.44)\n" +
"  //   First-class pair/tuple IR + live aarch64 RDVL + one-pass fused sum_scan (v0.45)\n" +
"  //   Language pair surface + streaming-SVE/SME VL + map+reduce fusion (v0.46)\n" +
"  //   pair destructure/return + SME ZA tile + broader map+reduce fusion (v0.51)\n"
		for _, marker := range []string{
			"  //   trin.vspan_add %a, %b : vector<16xf32>\n",
			"  //   trin.vspan_add %a, %b : vector<8xf32>\n",
		} {
			if strings.Contains(out, marker) {
				out = strings.Replace(out, marker, marker+"\n"+catalog, 1)
				return out
			}
		}
		for _, mod := range []string{
			`module attributes {trin.version = "0.81"`,
			`module attributes {trin.version = "0.44"`,
			`module attributes {trin.version = "0.43"`,
			`module attributes {trin.version = "0.42"`,
			`module attributes {trin.version = "0.40"`,
			`module attributes {trin.version = "0.39"`,
			`module attributes {trin.version = "0.37"`,
			`module attributes {trin.version = "0.35"`,
			`module attributes {trin.version = "0.34"`,
			`module attributes {trin.version = "0.33"`,
			`module attributes {trin.version = "0.32"`,
			`module attributes {trin.version = "0.31"`,
			`module attributes {trin.version = "0.30"`,
			`module attributes {trin.version = "0.29"`,
			`module attributes {trin.version = "0.28"`,
			`module attributes {trin.version = "0.27"`,
			`module attributes {trin.version = "0.26"`,
		} {
			if i := strings.Index(out, mod); i >= 0 {
				// insert catalog after the module open line
				if j := strings.Index(out[i:], "\n"); j >= 0 {
					at := i + j + 1
					out = out[:at] + "\n" + catalog + out[at:]
					break
				}
			}
		}
	}
	return out
}

func reGroups(re *regexp.Regexp, m []string) map[string]string {
	out := map[string]string{}
	for i, name := range re.SubexpNames() {
		if i == 0 || name == "" {
			continue
		}
		out[name] = m[i]
	}
	return out
}

// emitDialectRewriteSurface embeds high-level dialect probes into emit-mlir
// so rewrite + lower paths have pattern-matching fodder (v0.25 / v0.27 / v0.29).
func emitDialectRewriteSurface() string {
	var b strings.Builder
	// soft_mux dialect form → target 4 after rewrite+call.
	// pure/each via addf(+0) so v0.29 full-const strength does not fold the call;
	// this probe exists to show dialect → func.call @trin_soft_mux expansion.
	b.WriteString("  // High-level trin dialect ops (pre-rewrite form; v0.25+)\n")
	b.WriteString("  func.func @trin_mlir_dialect_probe() -> f32 {\n")
	b.WriteString("    %w = arith.constant 2.500000e-01 : f32\n")
	b.WriteString("    %p0 = arith.constant 1.000000e+01 : f32\n")
	b.WriteString("    %e0 = arith.constant 2.000000e+00 : f32\n")
	b.WriteString("    %z = arith.constant 0.000000e+00 : f32\n")
	b.WriteString("    %p = arith.addf %p0, %z : f32\n")
	b.WriteString("    %e = arith.addf %e0, %z : f32\n")
	b.WriteString("    %y = \"trin.soft_mux\"(%w, %p, %e) : (f32, f32, f32) -> f32\n")
	b.WriteString("    return %y : f32\n")
	b.WriteString("  }\n\n")

	// Fold probe: soft_mux(1, 7, 9) → 7 after rewrite
	b.WriteString("  func.func @trin_mlir_fold_probe() -> f32 {\n")
	b.WriteString("    %w1 = arith.constant 1.000000e+00 : f32\n")
	b.WriteString("    %pure = arith.constant 7.000000e+00 : f32\n")
	b.WriteString("    %each = arith.constant 9.000000e+00 : f32\n")
	b.WriteString("    %y = \"trin.soft_mux\"(%w1, %pure, %each) : (f32, f32, f32) -> f32\n")
	b.WriteString("    return %y : f32\n")
	b.WriteString("  }\n\n")

	// Vector dialect: vsum (1,2,3,4) → 10 after rewrite to @trin_sum_v4
	b.WriteString("  func.func @trin_mlir_vsum_probe() -> f32 {\n")
	b.WriteString("    %c1 = arith.constant 1.000000e+00 : f32\n")
	b.WriteString("    %c2 = arith.constant 2.000000e+00 : f32\n")
	b.WriteString("    %c3 = arith.constant 3.000000e+00 : f32\n")
	b.WriteString("    %c4 = arith.constant 4.000000e+00 : f32\n")
	b.WriteString("    %a = vector.from_elements %c1, %c2, %c3, %c4 : vector<4xf32>\n")
	b.WriteString("    %y = \"trin.vsum\"(%a) : (vector<4xf32>) -> f32\n")
	b.WriteString("    return %y : f32\n")
	b.WriteString("  }\n\n")

	// CSE probe (v0.27): (1+2) twice → commoned; sum = 6
	b.WriteString("  // CSE pure arith (v0.27): duplicate addf commons → 6\n")
	b.WriteString("  func.func @trin_mlir_cse_probe() -> f32 {\n")
	b.WriteString("    %c1 = arith.constant 1.000000e+00 : f32\n")
	b.WriteString("    %c2 = arith.constant 2.000000e+00 : f32\n")
	b.WriteString("    %a = arith.addf %c1, %c2 : f32\n")
	b.WriteString("    %b = arith.addf %c1, %c2 : f32\n")
	b.WriteString("    %y = arith.addf %a, %b : f32\n")
	b.WriteString("    return %y : f32\n")
	b.WriteString("  }\n\n")

	// Multi-use soft_mux commoning (v0.27): soft_mux(0.5,10,2)=6 twice → sum 12.
	// pure/each are produced via addf(+0) so v0.29 full-const strength does not
	// fold the soft_mux away before multi-use CSE can common the identical calls.
	b.WriteString("  // Multi-use soft_mux CSE (v0.27): duplicate soft_mux → 12\n")
	b.WriteString("  func.func @trin_mlir_soft_cse_probe() -> f32 {\n")
	b.WriteString("    %w = arith.constant 5.000000e-01 : f32\n")
	b.WriteString("    %p0 = arith.constant 1.000000e+01 : f32\n")
	b.WriteString("    %e0 = arith.constant 2.000000e+00 : f32\n")
	b.WriteString("    %z = arith.constant 0.000000e+00 : f32\n")
	b.WriteString("    %p = arith.addf %p0, %z : f32\n")
	b.WriteString("    %e = arith.addf %e0, %z : f32\n")
	b.WriteString("    %y1 = \"trin.soft_mux\"(%w, %p, %e) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %y2 = \"trin.soft_mux\"(%w, %p, %e) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %y = arith.addf %y1, %y2 : f32\n")
	b.WriteString("    return %y : f32\n")
	b.WriteString("  }\n\n")

	// Vector-width pick probe (v0.27): trin.vsum on vector<16xf32> → @trin_sum_v16 → 136
	b.WriteString("  // Vector-width pick (v0.27): vsum vector<16xf32> → sum_v16 = 136\n")
	b.WriteString("  func.func @trin_mlir_vwidth_probe() -> f32 {\n")
	b.WriteString("    %c1 = arith.constant 1.000000e+00 : f32\n")
	b.WriteString("    %c2 = arith.constant 2.000000e+00 : f32\n")
	b.WriteString("    %c3 = arith.constant 3.000000e+00 : f32\n")
	b.WriteString("    %c4 = arith.constant 4.000000e+00 : f32\n")
	b.WriteString("    %c5 = arith.constant 5.000000e+00 : f32\n")
	b.WriteString("    %c6 = arith.constant 6.000000e+00 : f32\n")
	b.WriteString("    %c7 = arith.constant 7.000000e+00 : f32\n")
	b.WriteString("    %c8 = arith.constant 8.000000e+00 : f32\n")
	b.WriteString("    %c9 = arith.constant 9.000000e+00 : f32\n")
	b.WriteString("    %c10 = arith.constant 1.000000e+01 : f32\n")
	b.WriteString("    %c11 = arith.constant 1.100000e+01 : f32\n")
	b.WriteString("    %c12 = arith.constant 1.200000e+01 : f32\n")
	b.WriteString("    %c13 = arith.constant 1.300000e+01 : f32\n")
	b.WriteString("    %c14 = arith.constant 1.400000e+01 : f32\n")
	b.WriteString("    %c15 = arith.constant 1.500000e+01 : f32\n")
	b.WriteString("    %c16 = arith.constant 1.600000e+01 : f32\n")
	b.WriteString("    %a = vector.from_elements %c1, %c2, %c3, %c4, %c5, %c6, %c7, %c8, %c9, %c10, %c11, %c12, %c13, %c14, %c15, %c16 : vector<16xf32>\n")
	b.WriteString("    %y = \"trin.vsum\"(%a) : (vector<16xf32>) -> f32\n")
	b.WriteString("    return %y : f32\n")
	b.WriteString("  }\n\n")

	// Strength-reduction probe (v0.29): pure==each → 5; each=0 → 3; pure=0 → 1.5; const fold → 4; sum = 13.5
	b.WriteString("  // soft_mux strength / algebraic folds (v0.29) → target 13.5\n")
	b.WriteString("  func.func @trin_mlir_strength_probe() -> f32 {\n")
	b.WriteString("    %w = arith.constant 5.000000e-01 : f32\n")
	b.WriteString("    %x = arith.constant 5.000000e+00 : f32\n")
	b.WriteString("    %p6 = arith.constant 6.000000e+00 : f32\n")
	b.WriteString("    %z = arith.constant 0.000000e+00 : f32\n")
	b.WriteString("    %e3 = arith.constant 3.000000e+00 : f32\n")
	b.WriteString("    %wc = arith.constant 2.500000e-01 : f32\n")
	b.WriteString("    %pc = arith.constant 1.000000e+01 : f32\n")
	b.WriteString("    %ec = arith.constant 2.000000e+00 : f32\n")
	// soft_mux(w,x,x) → x (=5); soft_mux(w,6,0) → 3; soft_mux(w,0,3) → 1.5; soft_mux(0.25,10,2) → 4
	b.WriteString("    %a = \"trin.soft_mux\"(%w, %x, %x) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %b = \"trin.soft_mux\"(%w, %p6, %z) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %c = \"trin.soft_mux\"(%w, %z, %e3) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %d = \"trin.soft_mux\"(%wc, %pc, %ec) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %s0 = arith.addf %a, %b : f32\n")
	b.WriteString("    %s1 = arith.addf %s0, %c : f32\n")
	b.WriteString("    %y = arith.addf %s1, %d : f32\n")
	b.WriteString("    return %y : f32\n")
	b.WriteString("  }\n\n")

	// Span/vector algebraic probe (v0.31):
	// a=(1,2,3,4) sum=10; z=0; add(a,z)→a sum=10; add(z,a)→a sum=10;
	// scale(1,a)→a sum=10; scale(0,a)→0 sum=0; sum(z)=0; dot(a,z)=0
	// total = 10+10+10+0+0+0 = 30
	b.WriteString("  // span/vector algebraic folds (v0.31) → target 30\n")
	b.WriteString("  func.func @trin_mlir_span_alg_probe() -> f32 {\n")
	b.WriteString("    %c1 = arith.constant 1.000000e+00 : f32\n")
	b.WriteString("    %c2 = arith.constant 2.000000e+00 : f32\n")
	b.WriteString("    %c3 = arith.constant 3.000000e+00 : f32\n")
	b.WriteString("    %c4 = arith.constant 4.000000e+00 : f32\n")
	b.WriteString("    %z0 = arith.constant 0.000000e+00 : f32\n")
	b.WriteString("    %one = arith.constant 1.000000e+00 : f32\n")
	b.WriteString("    %a = vector.from_elements %c1, %c2, %c3, %c4 : vector<4xf32>\n")
	b.WriteString("    %z = vector.broadcast %z0 : f32 to vector<4xf32>\n")
	b.WriteString("    %add0 = \"trin.vspan_add\"(%a, %z) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>\n")
	b.WriteString("    %add1 = \"trin.vspan_add\"(%z, %a) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>\n")
	b.WriteString("    %sc1 = \"trin.vspan_scale\"(%one, %a) : (f32, vector<4xf32>) -> vector<4xf32>\n")
	b.WriteString("    %sc0 = \"trin.vspan_scale\"(%z0, %a) : (f32, vector<4xf32>) -> vector<4xf32>\n")
	b.WriteString("    %s0 = \"trin.vsum\"(%add0) : (vector<4xf32>) -> f32\n")
	b.WriteString("    %s1 = \"trin.vsum\"(%add1) : (vector<4xf32>) -> f32\n")
	b.WriteString("    %s2 = \"trin.vsum\"(%sc1) : (vector<4xf32>) -> f32\n")
	b.WriteString("    %s3 = \"trin.vsum\"(%sc0) : (vector<4xf32>) -> f32\n")
	b.WriteString("    %s4 = \"trin.vsum\"(%z) : (vector<4xf32>) -> f32\n")
	b.WriteString("    %s5 = \"trin.vdot\"(%a, %z) : (vector<4xf32>, vector<4xf32>) -> f32\n")
	b.WriteString("    %t0 = arith.addf %s0, %s1 : f32\n")
	b.WriteString("    %t1 = arith.addf %t0, %s2 : f32\n")
	b.WriteString("    %t2 = arith.addf %t1, %s3 : f32\n")
	b.WriteString("    %t3 = arith.addf %t2, %s4 : f32\n")
	b.WriteString("    %y = arith.addf %t3, %s5 : f32\n")
	b.WriteString("    return %y : f32\n")
	b.WriteString("  }\n\n")

	// Bitplane + soft_mux_v strength probe (v0.33) → float target 42
	// Planes: x=0b1111=15, z=0, y=0b0011=3
	// and(x,x)→15; and(x,0)→0; or(x,0)→15; xor(x,x)→0; popcount(15)→4; popcount(0)→0
	// plane sum = 15+0+15+0+4+0 = 34
	// soft_mux_v4(w=0.5, a=(2,2,2,2), 0) → broadcast(0.5)*a sum = 4
	// soft_mux_v4(w=0.5, 0, a) → broadcast(0.5)*a sum = 4
	// total = 34+4+4 = 42
	b.WriteString("  // soft_mux_v + bitplane algebraic folds (v0.33) → target 42\n")
	b.WriteString("  func.func @trin_plane_and(%a: i32, %b: i32) -> i32 {\n")
	b.WriteString("    %y = arith.andi %a, %b : i32\n")
	b.WriteString("    return %y : i32\n")
	b.WriteString("  }\n\n")
	b.WriteString("  func.func @trin_plane_or(%a: i32, %b: i32) -> i32 {\n")
	b.WriteString("    %y = arith.ori %a, %b : i32\n")
	b.WriteString("    return %y : i32\n")
	b.WriteString("  }\n\n")
	b.WriteString("  func.func @trin_plane_xor(%a: i32, %b: i32) -> i32 {\n")
	b.WriteString("    %y = arith.xori %a, %b : i32\n")
	b.WriteString("    return %y : i32\n")
	b.WriteString("  }\n\n")
	b.WriteString("  func.func @trin_plane_popcount(%w: i32) -> i32 {\n")
	b.WriteString("    // lowered to llvm.ctpop on LLVM path; portable rewrite folds constants\n")
	b.WriteString("    return %w : i32\n")
	b.WriteString("  }\n\n")
	b.WriteString("  func.func @trin_mlir_bitplane_probe() -> f32 {\n")
	b.WriteString("    %x = arith.constant 15 : i32\n")
	b.WriteString("    %z = arith.constant 0 : i32\n")
	b.WriteString("    %y = arith.constant 3 : i32\n")
	b.WriteString("    %a1 = func.call @trin_plane_and(%x, %x) : (i32, i32) -> i32\n")
	b.WriteString("    %a0 = func.call @trin_plane_and(%x, %z) : (i32, i32) -> i32\n")
	b.WriteString("    %o0 = func.call @trin_plane_or(%x, %z) : (i32, i32) -> i32\n")
	b.WriteString("    %xr = func.call @trin_plane_xor(%x, %x) : (i32, i32) -> i32\n")
	b.WriteString("    %pc = func.call @trin_plane_popcount(%x) : (i32) -> i32\n")
	b.WriteString("    %pz = func.call @trin_plane_popcount(%z) : (i32) -> i32\n")
	// cast i32 plane results to f32 and sum: 15+0+15+0+4+0 = 34
	b.WriteString("    %fa1 = arith.sitofp %a1 : i32 to f32\n")
	b.WriteString("    %fa0 = arith.sitofp %a0 : i32 to f32\n")
	b.WriteString("    %fo0 = arith.sitofp %o0 : i32 to f32\n")
	b.WriteString("    %fxr = arith.sitofp %xr : i32 to f32\n")
	b.WriteString("    %fpc = arith.sitofp %pc : i32 to f32\n")
	b.WriteString("    %fpz = arith.sitofp %pz : i32 to f32\n")
	b.WriteString("    %s0 = arith.addf %fa1, %fa0 : f32\n")
	b.WriteString("    %s1 = arith.addf %s0, %fo0 : f32\n")
	b.WriteString("    %s2 = arith.addf %s1, %fxr : f32\n")
	b.WriteString("    %s3 = arith.addf %s2, %fpc : f32\n")
	b.WriteString("    %plane = arith.addf %s3, %fpz : f32\n")
	// soft_mux_v4 strength: each=0 and pure=0 arms
	b.WriteString("    %w = arith.constant 5.000000e-01 : f32\n")
	b.WriteString("    %c2 = arith.constant 2.000000e+00 : f32\n")
	b.WriteString("    %z0 = arith.constant 0.000000e+00 : f32\n")
	b.WriteString("    %pure = vector.from_elements %c2, %c2, %c2, %c2 : vector<4xf32>\n")
	b.WriteString("    %zv = vector.broadcast %z0 : f32 to vector<4xf32>\n")
	b.WriteString("    %m0 = func.call @trin_soft_mux_v4(%w, %pure, %zv) : (f32, vector<4xf32>, vector<4xf32>) -> vector<4xf32>\n")
	b.WriteString("    %m1 = func.call @trin_soft_mux_v4(%w, %zv, %pure) : (f32, vector<4xf32>, vector<4xf32>) -> vector<4xf32>\n")
	b.WriteString("    %sm0 = func.call @trin_sum_v4(%m0) : (vector<4xf32>) -> f32\n")
	b.WriteString("    %sm1 = func.call @trin_sum_v4(%m1) : (vector<4xf32>) -> f32\n")
	b.WriteString("    %t0 = arith.addf %plane, %sm0 : f32\n")
	b.WriteString("    %y2 = arith.addf %t0, %sm1 : f32\n")
	// touch %y so CSE doesn't drop the unused const in notes
	b.WriteString("    %_ = arith.addi %y, %z : i32\n")
	b.WriteString("    return %y2 : f32\n")
	b.WriteString("  }\n\n")

	// Nested soft_mux / bitplane product probe (v0.35) → target 52
	// soft product pure-arm: soft_mux(0.5, soft_mux(0.5,10,2), 2) → soft_mux(0.25,10,2) = 4
	// soft product each-arm: soft_mux(0.5, 10, soft_mux(0.5,10,2)) → soft_mux(0.75,10,2) = 8
	// soft product pure again (distinct SSA weights) = 4; soft sum = 16
	// plane: and(a,x)→3, or(o,x)→15, and(x,o)→15, and(a,o)→3 = 36
	// total 52
	b.WriteString("  // nested soft_mux + bitplane product folds (v0.35) → target 52\n")
	b.WriteString("  func.func @trin_mlir_nested_probe() -> f32 {\n")
	b.WriteString("    %w1 = arith.constant 5.000000e-01 : f32\n")
	b.WriteString("    %w2 = arith.constant 5.000000e-01 : f32\n")
	b.WriteString("    %w3 = arith.constant 5.000000e-01 : f32\n")
	b.WriteString("    %w4 = arith.constant 5.000000e-01 : f32\n")
	b.WriteString("    %p = arith.constant 1.000000e+01 : f32\n")
	b.WriteString("    %e = arith.constant 2.000000e+00 : f32\n")
	b.WriteString("    %z0 = arith.constant 0.000000e+00 : f32\n")
	b.WriteString("    %pp = arith.addf %p, %z0 : f32\n")
	b.WriteString("    %ee = arith.addf %e, %z0 : f32\n")
	// pure-arm product
	b.WriteString("    %inner_p = func.call @trin_soft_mux(%w2, %pp, %ee) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %prod_p = func.call @trin_soft_mux(%w1, %inner_p, %ee) : (f32, f32, f32) -> f32\n")
	// each-arm product
	b.WriteString("    %inner_e = func.call @trin_soft_mux(%w4, %pp, %ee) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %prod_e = func.call @trin_soft_mux(%w3, %pp, %inner_e) : (f32, f32, f32) -> f32\n")
	// second pure-arm product for soft sum 4+8+4
	b.WriteString("    %w5 = arith.constant 5.000000e-01 : f32\n")
	b.WriteString("    %w6 = arith.constant 5.000000e-01 : f32\n")
	b.WriteString("    %inner_p2 = func.call @trin_soft_mux(%w6, %pp, %ee) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %prod_p2 = func.call @trin_soft_mux(%w5, %inner_p2, %ee) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %s0 = arith.addf %prod_p, %prod_e : f32\n")
	b.WriteString("    %soft = arith.addf %s0, %prod_p2 : f32\n")
	// bitplane product / absorb
	b.WriteString("    %x = arith.constant 15 : i32\n")
	b.WriteString("    %y = arith.constant 3 : i32\n")
	b.WriteString("    %zz = arith.constant 0 : i32\n")
	b.WriteString("    %xa = arith.addi %x, %zz : i32\n")
	b.WriteString("    %ya = arith.addi %y, %zz : i32\n")
	b.WriteString("    %a = func.call @trin_plane_and(%xa, %ya) : (i32, i32) -> i32\n")
	b.WriteString("    %o = func.call @trin_plane_or(%xa, %ya) : (i32, i32) -> i32\n")
	b.WriteString("    %and_abs = func.call @trin_plane_and(%a, %xa) : (i32, i32) -> i32\n")
	b.WriteString("    %or_abs = func.call @trin_plane_or(%o, %xa) : (i32, i32) -> i32\n")
	b.WriteString("    %and_abs2 = func.call @trin_plane_and(%xa, %o) : (i32, i32) -> i32\n")
	b.WriteString("    %and_prod = func.call @trin_plane_and(%a, %o) : (i32, i32) -> i32\n")
	b.WriteString("    %f0 = arith.sitofp %and_abs : i32 to f32\n")
	b.WriteString("    %f1 = arith.sitofp %or_abs : i32 to f32\n")
	b.WriteString("    %f2 = arith.sitofp %and_abs2 : i32 to f32\n")
	b.WriteString("    %f3 = arith.sitofp %and_prod : i32 to f32\n")
	b.WriteString("    %p0 = arith.addf %f0, %f1 : f32\n")
	b.WriteString("    %p1 = arith.addf %p0, %f2 : f32\n")
	b.WriteString("    %plane = arith.addf %p1, %f3 : f32\n")
	b.WriteString("    %y2 = arith.addf %soft, %plane : f32\n")
	b.WriteString("    return %y2 : f32\n")
	b.WriteString("  }\n\n")

	// Multi-level soft_mux trees + multi-word plane GEMM (v0.38) → target 64
	// Redesign:
	// soft_tree1 = 5 (both-arms)
	// soft_triple = 3
	// soft_tree2 pure nest → 8
	// soft total = 5+3+8 = 16
	// ap=0x00FFFFFF bp=0x00FFFFFF → popcount 24 twice = 48
	// soft 16 + plane 48 = 64
	b.WriteString("  // multi-level soft_mux trees + multi-word plane GEMM folds (v0.38) → target 64\n")
	// declare multi-word GEMM helpers as opaque dialect callees
	b.WriteString("  func.func @trin_trit_dot_planes_word(%ap: i32, %an: i32, %bp: i32, %bn: i32) -> i32 {\n")
	b.WriteString("    %t0 = arith.andi %ap, %bp : i32\n")
	b.WriteString("    %t1 = arith.andi %an, %bn : i32\n")
	b.WriteString("    %pos = arith.ori %t0, %t1 : i32\n")
	b.WriteString("    %t2 = arith.andi %ap, %bn : i32\n")
	b.WriteString("    %t3 = arith.andi %an, %bp : i32\n")
	b.WriteString("    %neg = arith.ori %t2, %t3 : i32\n")
	b.WriteString("    %pcp = func.call @trin_plane_popcount(%pos) : (i32) -> i32\n")
	b.WriteString("    %pcn = func.call @trin_plane_popcount(%neg) : (i32) -> i32\n")
	b.WriteString("    %y = arith.subi %pcp, %pcn : i32\n")
	b.WriteString("    return %y : i32\n")
	b.WriteString("  }\n\n")
	b.WriteString("  func.func @trin_trit_dot_planes_x4(\n")
	b.WriteString("      %ap0: i32, %an0: i32, %bp0: i32, %bn0: i32,\n")
	b.WriteString("      %ap1: i32, %an1: i32, %bp1: i32, %bn1: i32,\n")
	b.WriteString("      %ap2: i32, %an2: i32, %bp2: i32, %bn2: i32,\n")
	b.WriteString("      %ap3: i32, %an3: i32, %bp3: i32, %bn3: i32) -> i32 {\n")
	b.WriteString("    %d0 = func.call @trin_trit_dot_planes_word(%ap0, %an0, %bp0, %bn0) : (i32, i32, i32, i32) -> i32\n")
	b.WriteString("    %d1 = func.call @trin_trit_dot_planes_word(%ap1, %an1, %bp1, %bn1) : (i32, i32, i32, i32) -> i32\n")
	b.WriteString("    %d2 = func.call @trin_trit_dot_planes_word(%ap2, %an2, %bp2, %bn2) : (i32, i32, i32, i32) -> i32\n")
	b.WriteString("    %d3 = func.call @trin_trit_dot_planes_word(%ap3, %an3, %bp3, %bn3) : (i32, i32, i32, i32) -> i32\n")
	b.WriteString("    %s0 = arith.addi %d0, %d1 : i32\n")
	b.WriteString("    %s1 = arith.addi %s0, %d2 : i32\n")
	b.WriteString("    %y = arith.addi %s1, %d3 : i32\n")
	b.WriteString("    return %y : i32\n")
	b.WriteString("  }\n\n")
	b.WriteString("  func.func @trin_mlir_tree_probe() -> f32 {\n")
	// both-arms same tree: soft_mux(0.5, soft_mux(0.5,10,2), soft_mux(0.25,10,2)) = 5
	b.WriteString("    %w = arith.constant 5.000000e-01 : f32\n")
	b.WriteString("    %w2 = arith.constant 5.000000e-01 : f32\n")
	b.WriteString("    %w3 = arith.constant 2.500000e-01 : f32\n")
	b.WriteString("    %p0 = arith.constant 1.000000e+01 : f32\n")
	b.WriteString("    %e0 = arith.constant 2.000000e+00 : f32\n")
	b.WriteString("    %z0 = arith.constant 0.000000e+00 : f32\n")
	b.WriteString("    %pp = arith.addf %p0, %z0 : f32\n")
	b.WriteString("    %ee = arith.addf %e0, %z0 : f32\n")
	b.WriteString("    %in_p = func.call @trin_soft_mux(%w2, %pp, %ee) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %in_e = func.call @trin_soft_mux(%w3, %pp, %ee) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %tree = func.call @trin_soft_mux(%w, %in_p, %in_e) : (f32, f32, f32) -> f32\n")
	// triple pure nest: 0.5*0.5*0.5 → soft_mux(0.125,10,2)=3
	b.WriteString("    %t1 = arith.constant 5.000000e-01 : f32\n")
	b.WriteString("    %t2 = arith.constant 5.000000e-01 : f32\n")
	b.WriteString("    %t3 = arith.constant 5.000000e-01 : f32\n")
	b.WriteString("    %innermost = func.call @trin_soft_mux(%t3, %pp, %ee) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %mid = func.call @trin_soft_mux(%t2, %innermost, %ee) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %triple = func.call @trin_soft_mux(%t1, %mid, %ee) : (f32, f32, f32) -> f32\n")
	// second triple pure nest on (36,4): w^3=0.125 → soft_mux(0.125,36,4)=8
	b.WriteString("    %u1 = arith.constant 5.000000e-01 : f32\n")
	b.WriteString("    %u2 = arith.constant 5.000000e-01 : f32\n")
	b.WriteString("    %u3 = arith.constant 5.000000e-01 : f32\n")
	b.WriteString("    %p36 = arith.constant 3.600000e+01 : f32\n")
	b.WriteString("    %e4 = arith.constant 4.000000e+00 : f32\n")
	b.WriteString("    %p36a = arith.addf %p36, %z0 : f32\n")
	b.WriteString("    %e4a = arith.addf %e4, %z0 : f32\n")
	b.WriteString("    %innermost2 = func.call @trin_soft_mux(%u3, %p36a, %e4a) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %mid2 = func.call @trin_soft_mux(%u2, %innermost2, %e4a) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %tree2 = func.call @trin_soft_mux(%u1, %mid2, %e4a) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %s0 = arith.addf %tree, %triple : f32\n")
	b.WriteString("    %soft = arith.addf %s0, %tree2 : f32\n")
	// multi-word GEMM folds: word(const)=24, word(zero)=0, x4(zero)=0, word2=24 → 48
	b.WriteString("    %ap = arith.constant 16777215 : i32\n") // 0x00FFFFFF → 24 bits
	b.WriteString("    %an = arith.constant 0 : i32\n")
	b.WriteString("    %bp = arith.constant 16777215 : i32\n")
	b.WriteString("    %bn = arith.constant 0 : i32\n")
	b.WriteString("    %z = arith.constant 0 : i32\n")
	b.WriteString("    %d0 = func.call @trin_trit_dot_planes_word(%ap, %an, %bp, %bn) : (i32, i32, i32, i32) -> i32\n")
	b.WriteString("    %d1 = func.call @trin_trit_dot_planes_word(%z, %z, %bp, %bn) : (i32, i32, i32, i32) -> i32\n")
	b.WriteString("    %x4z = func.call @trin_trit_dot_planes_x4(%z, %z, %z, %z, %z, %z, %z, %z, %z, %z, %z, %z, %z, %z, %z, %z) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32\n")
	b.WriteString("    %d2 = func.call @trin_trit_dot_planes_word(%ap, %an, %bp, %bn) : (i32, i32, i32, i32) -> i32\n")
	b.WriteString("    %g0 = arith.addi %d0, %d1 : i32\n")
	b.WriteString("    %g1 = arith.addi %g0, %x4z : i32\n")
	b.WriteString("    %g2 = arith.addi %g1, %d2 : i32\n")
	b.WriteString("    %plane = arith.sitofp %g2 : i32 to f32\n")
	b.WriteString("    %y = arith.addf %soft, %plane : f32\n")
	b.WriteString("    return %y : f32\n")
	b.WriteString("  }\n\n")

	// soft_mux forest + multi-word GEMM schedule (v0.40) → target 80
	// Forest: soft_mux(0.5, soft_mux(0.5, soft_mux(0.5,10,2), 2), soft_mux(0.25,10,2))
	//   wn = 0.5*0.5*0.5 + 0.5*0.25 = 0.125+0.125 = 0.25 → soft_mux(0.25,10,2)=4
	// forest2: soft_mux(0.5, soft_mux(1, soft_mux(0.5,20,4), 4), soft_mux(0,20,4))
	//   wn = 0.5*1*0.5 + 0.5*0 = 0.25 → soft_mux(0.25,20,4)=8
	// soft = 4+8 = 12
	// schedule equal x4 of word 24: ap=bp=0x00FFFFFF → 4*24=96? too big
	// Use 0x0000000F (pop 4) equal x4 = 16; plus one zero x4 = 0; plus word 24; plus word 0
	// plane = 16+0+24+0 = 40; soft 12 + 40? = 52 → bump soft
	// forest3 pure each: soft_mux(1, soft_mux(0.5, soft_mux(1,36,4), 4), soft_mux(0.5,36,4))
	//   wn = 1*0.5*1 + 0*0.5 = 0.5 → soft_mux(0.5,36,4)=20
	// soft = 4+8+20 = 32; plane with equal-4 of pop4=16 + word24 + zeros = 40 → 72
	// target 80: equal schedule of 0x0000FFFF (pop 16)*4 = 64 + word(16) + zeros = 80? skip soft
	// Combine: soft 16 (reuse forest1+2 and another 4) + plane equal 4*12 + zeros?
	// ap=0x00000FFF (12 bits)*4 equal = 48 + soft forest1+2+3 = 32 → 80
	// forest3: soft_mux(0.5, soft_mux(0.5, soft_mux(1, 28, 4), 4), soft_mux(0.5,28,4))
	//   wn = 0.5*0.5*1 + 0.5*0.5 = 0.25+0.25=0.5 → soft_mux(0.5,28,4)=16
	// soft 4+8+16=28; plane: x4 equal pop 12 → 48 + word 4 → total 52… soft+plane = 80 when plane=52
	// plane: equal x4 of 0x00000FFF (12)*4=48 + word(ap=0x0000000F)=4 + zeros = 52. Yes.
	b.WriteString("  // soft_mux forest + multi-word GEMM schedule folds (v0.40) → target 80\n")
	b.WriteString("  func.func @trin_mlir_forest_probe() -> f32 {\n")
	// forest1 → 4
	b.WriteString("    %w = arith.constant 5.000000e-01 : f32\n")
	b.WriteString("    %w2 = arith.constant 5.000000e-01 : f32\n")
	b.WriteString("    %w3 = arith.constant 5.000000e-01 : f32\n")
	b.WriteString("    %w4 = arith.constant 2.500000e-01 : f32\n")
	b.WriteString("    %p0 = arith.constant 1.000000e+01 : f32\n")
	b.WriteString("    %e0 = arith.constant 2.000000e+00 : f32\n")
	b.WriteString("    %z0 = arith.constant 0.000000e+00 : f32\n")
	b.WriteString("    %pp = arith.addf %p0, %z0 : f32\n")
	b.WriteString("    %ee = arith.addf %e0, %z0 : f32\n")
	b.WriteString("    %leaf = func.call @trin_soft_mux(%w3, %pp, %ee) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %mid = func.call @trin_soft_mux(%w2, %leaf, %ee) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %arm_e = func.call @trin_soft_mux(%w4, %pp, %ee) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %forest1 = func.call @trin_soft_mux(%w, %mid, %arm_e) : (f32, f32, f32) -> f32\n")
	// forest2 → 8
	b.WriteString("    %v = arith.constant 5.000000e-01 : f32\n")
	b.WriteString("    %v2 = arith.constant 1.000000e+00 : f32\n")
	b.WriteString("    %v3 = arith.constant 5.000000e-01 : f32\n")
	b.WriteString("    %v4 = arith.constant 0.000000e+00 : f32\n")
	b.WriteString("    %p20 = arith.constant 2.000000e+01 : f32\n")
	b.WriteString("    %e4 = arith.constant 4.000000e+00 : f32\n")
	b.WriteString("    %p20a = arith.addf %p20, %z0 : f32\n")
	b.WriteString("    %e4a = arith.addf %e4, %z0 : f32\n")
	b.WriteString("    %leaf2 = func.call @trin_soft_mux(%v3, %p20a, %e4a) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %mid2 = func.call @trin_soft_mux(%v2, %leaf2, %e4a) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %arm_e2 = func.call @trin_soft_mux(%v4, %p20a, %e4a) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %forest2 = func.call @trin_soft_mux(%v, %mid2, %arm_e2) : (f32, f32, f32) -> f32\n")
	// forest3 → 16
	b.WriteString("    %u = arith.constant 5.000000e-01 : f32\n")
	b.WriteString("    %u2 = arith.constant 5.000000e-01 : f32\n")
	b.WriteString("    %u3 = arith.constant 1.000000e+00 : f32\n")
	b.WriteString("    %u4 = arith.constant 5.000000e-01 : f32\n")
	b.WriteString("    %p28 = arith.constant 2.800000e+01 : f32\n")
	b.WriteString("    %e4b = arith.constant 4.000000e+00 : f32\n")
	b.WriteString("    %p28a = arith.addf %p28, %z0 : f32\n")
	b.WriteString("    %e4c = arith.addf %e4b, %z0 : f32\n")
	b.WriteString("    %leaf3 = func.call @trin_soft_mux(%u3, %p28a, %e4c) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %mid3 = func.call @trin_soft_mux(%u2, %leaf3, %e4c) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %arm_e3 = func.call @trin_soft_mux(%u4, %p28a, %e4c) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %forest3 = func.call @trin_soft_mux(%u, %mid3, %arm_e3) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %s0 = arith.addf %forest1, %forest2 : f32\n")
	b.WriteString("    %soft = arith.addf %s0, %forest3 : f32\n")
	// multi-word schedule: equal x4 of 0x00000FFF (12) → 48; word 0xF → 4; zeros → 0
	b.WriteString("    %ap = arith.constant 4095 : i32\n") // 0x00000FFF → 12 bits
	b.WriteString("    %an = arith.constant 0 : i32\n")
	b.WriteString("    %bp = arith.constant 4095 : i32\n")
	b.WriteString("    %bn = arith.constant 0 : i32\n")
	b.WriteString("    %z = arith.constant 0 : i32\n")
	b.WriteString("    %ap4 = arith.constant 15 : i32\n") // 0xF → 4 bits
	b.WriteString("    %bp4 = arith.constant 15 : i32\n")
	b.WriteString("    %x4 = func.call @trin_trit_dot_planes_x4(%ap, %an, %bp, %bn, %ap, %an, %bp, %bn, %ap, %an, %bp, %bn, %ap, %an, %bp, %bn) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32\n")
	b.WriteString("    %x4z = func.call @trin_trit_dot_planes_x4(%z, %z, %z, %z, %z, %z, %z, %z, %z, %z, %z, %z, %z, %z, %z, %z) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32\n")
	b.WriteString("    %d0 = func.call @trin_trit_dot_planes_word(%ap4, %an, %bp4, %bn) : (i32, i32, i32, i32) -> i32\n")
	b.WriteString("    %d1 = func.call @trin_trit_dot_planes_word(%z, %z, %bp, %bn) : (i32, i32, i32, i32) -> i32\n")
	b.WriteString("    %g0 = arith.addi %x4, %x4z : i32\n")
	b.WriteString("    %g1 = arith.addi %g0, %d0 : i32\n")
	b.WriteString("    %g2 = arith.addi %g1, %d1 : i32\n")
	b.WriteString("    %plane = arith.sitofp %g2 : i32 to f32\n")
	b.WriteString("    %y = arith.addf %soft, %plane : f32\n")
	b.WriteString("    return %y : f32\n")
	b.WriteString("  }\n\n")


	// deeper soft_mux forest schedule (v0.41) → target 90
	// deep forest: soft_mux(0.5, soft_mux(0.5, soft_mux(0.5, soft_mux(0.5,10,2), 2), 2), soft_mux(0.25,10,2))
	//   wn = 0.5^4 + 0.5*0.25 = 0.0625+0.125 = 0.1875 → soft_mux(0.1875,10,2)=3.5
	// each-arm product: soft_mux(0.5, soft_mux(0.5,10,2), soft_mux(0.5, soft_mux(0.5,10,2), 2))
	//   wn = 0.5*0.5 + 0.5*0.5*0.5 = 0.25+0.125=0.375 → soft_mux(0.375,10,2)=5
	// soft = 3.5+5 = 8.5
	// Also reuse tree both-arms =5 and pure nest=4 → wait redesign target 90:
	// Use large soft values via (p,e)=(36,4):
	// deep: soft_mux(0.5, nest4 of (36,4), soft_mux(0,36,4))
	//   pure product 0.5^4=0.0625 → 0.0625*36+0.9375*4 = 2.25+3.75=6 ; each arm wn=0 → 4; 
	// better const: deep forest with wn const folded to y.
	// Design: soft_deep=6.5 + each_prod=8 + deep2=10.5 + plane equal? 
	// Simple: three deep forests that const-fold:
	// D1: w=1,w2=1,w3=1,w4=0.5,w5=0,p=20,e=0 → wn=0.5 → y=10
	// D2: w=0.5,w2=1,w3=1,w4=1,w5=0.5,p=20,e=4 → wn=0.5*1 + 0.5*0.5=0.75 → y=16
	// E1 each-arm: w=0.5,w2=1,w3=1,w4=0.5,p=20,e=4 → wn=0.5 + 0.5*0.5=0.75 → y=16
	// soft=10+16+16=42
	// plane: x4 equal 0xFFF=12 *4 =48 → grand=90
	b.WriteString("  // deeper soft_mux forest schedule (v0.41) → target 90\n")
	b.WriteString("  func.func @trin_mlir_deep_forest_probe() -> f32 {\n")
	b.WriteString("    %z0 = arith.constant 0.000000e+00 : f32\n")
	// D1 → 10
	b.WriteString("    %w = arith.constant 1.000000e+00 : f32\n")
	b.WriteString("    %w2 = arith.constant 1.000000e+00 : f32\n")
	b.WriteString("    %w3 = arith.constant 1.000000e+00 : f32\n")
	b.WriteString("    %w4 = arith.constant 5.000000e-01 : f32\n")
	b.WriteString("    %w5 = arith.constant 0.000000e+00 : f32\n")
	b.WriteString("    %p20 = arith.constant 2.000000e+01 : f32\n")
	b.WriteString("    %e0 = arith.constant 0.000000e+00 : f32\n")
	b.WriteString("    %pp = arith.addf %p20, %z0 : f32\n")
	b.WriteString("    %ee = arith.addf %e0, %z0 : f32\n")
	b.WriteString("    %leaf = func.call @trin_soft_mux(%w4, %pp, %ee) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %m1 = func.call @trin_soft_mux(%w3, %leaf, %ee) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %m2 = func.call @trin_soft_mux(%w2, %m1, %ee) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %arm = func.call @trin_soft_mux(%w5, %pp, %ee) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %d1 = func.call @trin_soft_mux(%w, %m2, %arm) : (f32, f32, f32) -> f32\n")
	// D2 → 16
	b.WriteString("    %u = arith.constant 5.000000e-01 : f32\n")
	b.WriteString("    %u2 = arith.constant 1.000000e+00 : f32\n")
	b.WriteString("    %u3 = arith.constant 1.000000e+00 : f32\n")
	b.WriteString("    %u4 = arith.constant 1.000000e+00 : f32\n")
	b.WriteString("    %u5 = arith.constant 5.000000e-01 : f32\n")
	b.WriteString("    %p20b = arith.constant 2.000000e+01 : f32\n")
	b.WriteString("    %e4 = arith.constant 4.000000e+00 : f32\n")
	b.WriteString("    %pp2 = arith.addf %p20b, %z0 : f32\n")
	b.WriteString("    %ee2 = arith.addf %e4, %z0 : f32\n")
	b.WriteString("    %leaf2 = func.call @trin_soft_mux(%u4, %pp2, %ee2) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %m12 = func.call @trin_soft_mux(%u3, %leaf2, %ee2) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %m22 = func.call @trin_soft_mux(%u2, %m12, %ee2) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %arm2 = func.call @trin_soft_mux(%u5, %pp2, %ee2) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %d2 = func.call @trin_soft_mux(%u, %m22, %arm2) : (f32, f32, f32) -> f32\n")
	// E1 each-arm product → 16
	b.WriteString("    %v = arith.constant 5.000000e-01 : f32\n")
	b.WriteString("    %v2 = arith.constant 1.000000e+00 : f32\n")
	b.WriteString("    %v3 = arith.constant 1.000000e+00 : f32\n")
	b.WriteString("    %v4 = arith.constant 5.000000e-01 : f32\n")
	b.WriteString("    %p20c = arith.constant 2.000000e+01 : f32\n")
	b.WriteString("    %e4b = arith.constant 4.000000e+00 : f32\n")
	b.WriteString("    %pp3 = arith.addf %p20c, %z0 : f32\n")
	b.WriteString("    %ee3 = arith.addf %e4b, %z0 : f32\n")
	b.WriteString("    %pure_arm = func.call @trin_soft_mux(%v2, %pp3, %ee3) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %e_leaf = func.call @trin_soft_mux(%v4, %pp3, %ee3) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %e_mid = func.call @trin_soft_mux(%v3, %e_leaf, %ee3) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %e1 = func.call @trin_soft_mux(%v, %pure_arm, %e_mid) : (f32, f32, f32) -> f32\n")
	b.WriteString("    %s0 = arith.addf %d1, %d2 : f32\n")
	b.WriteString("    %soft = arith.addf %s0, %e1 : f32\n")
	// plane 48
	b.WriteString("    %ap = arith.constant 4095 : i32\n")
	b.WriteString("    %an = arith.constant 0 : i32\n")
	b.WriteString("    %bp = arith.constant 4095 : i32\n")
	b.WriteString("    %bn = arith.constant 0 : i32\n")
	b.WriteString("    %x4 = func.call @trin_trit_dot_planes_x4(%ap, %an, %bp, %bn, %ap, %an, %bp, %bn, %ap, %an, %bp, %bn, %ap, %an, %bp, %bn) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32\n")
	b.WriteString("    %plane = arith.sitofp %x4 : i32 to f32\n")
	b.WriteString("    %y = arith.addf %soft, %plane : f32\n")
	b.WriteString("    return %y : f32\n")
	b.WriteString("  }\n\n")

	return b.String()
}
