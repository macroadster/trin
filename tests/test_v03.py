"""Trin v0.3: bitplane trit packing + kernel emission."""

from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trin_lang.bitplane import (  # noqa: E402
    assert_gemm_equiv,
    pack_trits,
    pack_words,
    trit_dot_bitplane,
    trit_dot_words,
    trit_gemm_bitplane,
    trit_gemm_scalar,
    unpack_trits,
    unpack_words,
)
from trin_lang.interpreter import run_file, run_source  # noqa: E402
from trin_lang.typecheck import typecheck_source  # noqa: E402


class TestBitplaneCore(unittest.TestCase):
    def test_pack_unpack_element(self) -> None:
        ts = [1, 0, -1, 1, -1, 0, 1]
        pos, neg = pack_trits(ts)
        self.assertEqual(unpack_trits(pos, neg), ts)

    def test_pack_unpack_words(self) -> None:
        ts = [1, -1, 0, 1, 1, -1] * 20  # 120 trits, multi-word
        pw, nw, n = pack_words(ts)
        self.assertEqual(n, len(ts))
        self.assertEqual(unpack_words(pw, nw, n), ts)

    def test_dot_matches(self) -> None:
        a = [1, 0, -1, 1, -1, 0, 1, 1]
        b = [1, 1, 1, -1, 0, 1, -1, 1]
        scalar = sum(x * y for x, y in zip(a, b))
        self.assertEqual(trit_dot_bitplane(a, b), scalar)
        bp = __import__("trin_lang.bitplane", fromlist=["Bitplanes"]).Bitplanes
        ab = bp.from_trits(a)
        bb = bp.from_trits(b)
        self.assertEqual(
            trit_dot_words(ab.pos_words, ab.neg_words, bb.pos_words, bb.neg_words, len(a)),
            scalar,
        )

    def test_gemm_equiv_randomish(self) -> None:
        A = [1, 0, -1, 0, 1, 1, -1, -1, 0]
        B = [1, -1, 0, 1, 0, 1]
        # 3x3 @ 3x2
        out = assert_gemm_equiv(A, B, 3, 3, 2)
        self.assertEqual(out, trit_gemm_scalar(A, B, 3, 3, 2))
        self.assertEqual(out, trit_gemm_bitplane(A, B, 3, 3, 2))

    def test_gemm_matches_example_dims(self) -> None:
        A = [1, 0, -1, 0, 1, 1]
        B = [1, -1, 1, 0, 0, 1]
        self.assertEqual(assert_gemm_equiv(A, B, 2, 3, 2), [1, -2, 1, 1])


class TestBitplaneLang(unittest.TestCase):
    def test_example_runs(self) -> None:
        out = io.StringIO()
        run_file(ROOT / "examples" / "bitplane_gemm.trin", stdout=out)
        text = out.getvalue()
        self.assertIn("scalar   C = [1, -2, 1, 1]", text)
        self.assertIn("bitplane C = [1, -2, 1, 1]", text)
        self.assertIn("roundtrip A = [1, 0, -1, 0, 1, 1]", text)

    def test_example_typechecks(self) -> None:
        src = (ROOT / "examples" / "bitplane_gemm.trin").read_text(encoding="utf-8")
        errs = typecheck_source(src)
        self.assertEqual(errs, [], msg="; ".join(str(e) for e in errs))

    def test_builtin_parity(self) -> None:
        src = r"""
        fn main() {
            let I = [trit(1), trit(0), trit(0), trit(1)];
            let X = [trit(1), trit(-1), trit(0), trit(1)];
            let a = trit_gemm(I, X, 2, 2, 2);
            let b = trit_gemm_bitplane(I, X, 2, 2, 2);
            println(to_str(a));
            println(to_str(b));
        }
        """
        out = io.StringIO()
        run_source(src, stdout=out)
        lines = out.getvalue().strip().splitlines()
        self.assertEqual(lines[0], lines[1])
        self.assertEqual(lines[0], "[1, -1, 0, 1]")


class TestKernelEmission(unittest.TestCase):
    def test_emit_trit_gemm(self) -> None:
        from backend.emit_kernels import detect_kernel_calls, emit_c_from_kernels, emit_file
        from trin_lang.parser import parse_source

        src = (ROOT / "examples" / "trit_gemm.trin").read_text(encoding="utf-8")
        prog = parse_source(src)
        found = detect_kernel_calls(prog)
        self.assertTrue(any(k == "trit_gemm" for k, _ in found), found)
        c = emit_c_from_kernels(found, title="trit_gemm.trin")
        self.assertIn("trin_trit_gemm", c)
        self.assertIn("trin_trit_gemm_bitplane", c)
        self.assertIn("int8_t A", c)

        out = ROOT / "backend" / "_emit_test_out.c"
        try:
            emit_file(ROOT / "examples" / "trit_gemm.trin", out)
            text = out.read_text(encoding="utf-8")
            self.assertIn("trin_trit_gemm_bitplane", text)
            self.assertTrue((ROOT / "backend" / "kernels.h").exists())
        finally:
            if out.exists():
                out.unlink()

    def test_kernels_h_bitplane_symbols(self) -> None:
        text = (ROOT / "backend" / "kernels.h").read_text(encoding="utf-8")
        for sym in (
            "trin_pack_bitplanes",
            "trin_trit_dot_bitplane_u64",
            "trin_trit_gemm_bitplane",
            "trin_popcount64",
        ):
            self.assertIn(sym, text)


if __name__ == "__main__":
    unittest.main()
