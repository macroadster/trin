"""Trin v0.2 tests: typechecker, span kernels, trit GEMM."""

from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trin_lang.interpreter import run_file, run_source  # noqa: E402
from trin_lang.typecheck import typecheck_source  # noqa: E402


class TestTypecheck(unittest.TestCase):
    def test_hello_typechecks(self) -> None:
        src = (ROOT / "examples" / "hello.trin").read_text(encoding="utf-8")
        errs = typecheck_source(src)
        self.assertEqual(errs, [], msg="; ".join(str(e) for e in errs))

    def test_bad_gate_type(self) -> None:
        src = r"""
        fn main() {
            let x = "nope" | 1 | 0;
            println(x);
        }
        """
        errs = typecheck_source(src)
        self.assertTrue(any("gate" in e.message for e in errs), errs)

    def test_ternary_arm_mismatch(self) -> None:
        src = r"""
        fn main() {
            let x = true | "hi" | 3;
            println(x);
        }
        """
        errs = typecheck_source(src)
        self.assertTrue(any("disagree" in e.message for e in errs), errs)

    def test_undefined_name(self) -> None:
        src = r"""
        fn main() {
            println(missing);
        }
        """
        errs = typecheck_source(src)
        self.assertTrue(any("undefined" in e.message for e in errs), errs)


class TestSpanOps(unittest.TestCase):
    def test_span_example(self) -> None:
        out = io.StringIO()
        run_file(ROOT / "examples" / "span_ops.trin", stdout=out)
        text = out.getvalue()
        self.assertIn("doubled: [2.0, 4.0, 6.0, 8.0]", text)
        self.assertIn("sum: 10.0", text)
        self.assertIn("span_add: [2.0, 3.0, 4.0, 5.0]", text)
        self.assertIn("dot: 10.0", text)
        self.assertIn("non-empty", text)

    def test_zip_with(self) -> None:
        src = r"""
        fn main() {
            let a = [1, 2, 3];
            let b = [10, 20, 30];
            let c = zip_with((x, y) => x + y, a, b);
            println(to_str(c));
        }
        """
        out = io.StringIO()
        run_source(src, stdout=out)
        self.assertEqual(out.getvalue().strip(), "[11, 22, 33]")


class TestTritGemm(unittest.TestCase):
    def test_trit_gemm_example(self) -> None:
        out = io.StringIO()
        run_file(ROOT / "examples" / "trit_gemm.trin", stdout=out)
        text = out.getvalue()
        self.assertIn("C = [1, -2, 1, 1]", text)
        self.assertIn("C00 positive", text)

    def test_trit_identity(self) -> None:
        src = r"""
        fn main() {
            let I = [trit(1), trit(0), trit(0), trit(1)];
            let X = [trit(1), trit(-1), trit(0), trit(1)];
            let Y = trit_gemm(I, X, 2, 2, 2);
            println(to_str(Y));
        }
        """
        out = io.StringIO()
        run_source(src, stdout=out)
        self.assertEqual(out.getvalue().strip(), "[1, -1, 0, 1]")


class TestKernelsHeader(unittest.TestCase):
    def test_kernels_h_present(self) -> None:
        text = (ROOT / "backend" / "kernels.h").read_text(encoding="utf-8")
        self.assertIn("trin_span_add_f32", text)
        self.assertIn("trin_trit_gemm", text)
        self.assertIn("trin_span_soft_mux_f32", text)


if __name__ == "__main__":
    unittest.main()
