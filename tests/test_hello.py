"""End-to-end tests for Trin hello + ternary."""

from __future__ import annotations

import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trin_lang.interpreter import run_file, run_source  # noqa: E402
from trin_lang.parser import parse_source  # noqa: E402
from backend.c_lower import lower_file  # noqa: E402


HELLO = ROOT / "examples" / "hello.trin"


class TestHello(unittest.TestCase):
    def test_parse_hello(self) -> None:
        src = HELLO.read_text(encoding="utf-8")
        prog = parse_source(src)
        self.assertTrue(prog.items)

    def test_hello_with_name(self) -> None:
        out = io.StringIO()
        inp = io.StringIO("Maya\n")
        run_file(HELLO, stdin=inp, stdout=out)
        self.assertEqual(out.getvalue(), "Enter your name: Hello world, Maya\n")

    def test_hello_empty_name(self) -> None:
        out = io.StringIO()
        inp = io.StringIO("\n")
        run_file(HELLO, stdin=inp, stdout=out)
        self.assertEqual(out.getvalue(), "Enter your name: Hello world, stranger\n")

    def test_soft_blend(self) -> None:
        src = r"""
        fn main() {
            let g = unit_of(0.25);
            let n = g | 100 | 0;
            println(n);
        }
        """
        out = io.StringIO()
        run_source(src, stdin=io.StringIO(""), stdout=out)
        val = float(out.getvalue().strip())
        self.assertAlmostEqual(val, 25.0, places=5)

    def test_c_lower_emits_ternary_mux(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            c_path = lower_file(HELLO, td_path / "hello_gen.c")
            text = c_path.read_text(encoding="utf-8")
            self.assertIn("trin_hard_mux_str", text)
            self.assertIn("Hello world, ", text)
            self.assertIn("Hello world, stranger", text)
            self.assertIn("Enter your name:", text)
            self.assertIn("trin_soft_mux_f32", text)
            # Optional compile when a working C toolchain is present
            probe = subprocess.run(
                ["cc", "-O2", "-o", str(td_path / "hello_bin"), str(c_path)],
                capture_output=True,
                text=True,
            )
            if probe.returncode != 0:
                self.skipTest(f"C toolchain unavailable: {probe.stderr.strip()}")
            r2 = subprocess.run(
                [str(td_path / "hello_bin")],
                input="Eric\n",
                capture_output=True,
                text=True,
            )
            self.assertEqual(r2.returncode, 0)
            self.assertIn("Enter your name:", r2.stdout)
            self.assertIn("Hello world, Eric", r2.stdout)


if __name__ == "__main__":
    unittest.main()
