from __future__ import annotations

import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from slurm_mcp import transfers


class TransferHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_exclude_patterns_match_like_tar(self) -> None:
        patterns = ["__pycache__", "*.ckpt", "data/raw"]
        self.assertTrue(transfers.is_excluded("src/__pycache__/x.pyc", patterns))
        self.assertTrue(transfers.is_excluded("runs/a/model.ckpt", patterns))
        self.assertTrue(transfers.is_excluded("data/raw/file.bin", patterns))
        self.assertFalse(transfers.is_excluded("data/processed/file.bin", patterns))
        self.assertFalse(transfers.is_excluded("src/main.py", patterns))

    def test_local_tar_round_trip_with_excludes(self) -> None:
        source = self.tmp / "src"
        (source / "pkg" / "__pycache__").mkdir(parents=True)
        (source / "pkg" / "a.py").write_text("print(1)\n", encoding="utf-8", newline="\n")
        (source / "pkg" / "__pycache__" / "a.pyc").write_bytes(b"x")
        (source / "weights.ckpt").write_bytes(b"0" * 10)
        (source / "README.md").write_text("hi\n", encoding="utf-8", newline="\n")
        archive = self.tmp / "a.tar"
        entries, size = transfers.make_local_tar(source, archive, ["__pycache__", "*.ckpt"])
        self.assertEqual(entries, 3)  # pkg/, pkg/a.py, README.md
        self.assertEqual(size, len("print(1)\n") + len("hi\n"))
        dest = self.tmp / "dest"
        self.assertEqual(transfers.safe_extract(archive, dest), 3)
        self.assertEqual((dest / "pkg" / "a.py").read_text(encoding="utf-8"), "print(1)\n")
        self.assertFalse((dest / "weights.ckpt").exists())
        self.assertFalse((dest / "pkg" / "__pycache__").exists())

    def _archive_with(self, name: str, *, link_target: str | None = None) -> Path:
        archive = self.tmp / "crafted.tar"
        with tarfile.open(archive, "w") as tar:
            info = tarfile.TarInfo(name)
            if link_target is not None:
                info.type = tarfile.SYMTYPE
                info.linkname = link_target
                tar.addfile(info)
            else:
                data = b"evil"
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        return archive

    def test_safe_extract_refuses_entries_outside_destination(self) -> None:
        out = self.tmp / "out"
        for name, link in (("../evil.txt", None), ("escape", "../../outside"), ("abs-link", "/etc/passwd")):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "unsafe archive entry"):
                    transfers.safe_extract(self._archive_with(name, link_target=link), out)
        self.assertFalse((self.tmp / "evil.txt").exists())

    def test_safe_extract_keeps_absolute_names_inside_destination(self) -> None:
        # The data filter strips the leading '/', so the entry lands inside the destination.
        out = self.tmp / "out"
        transfers.safe_extract(self._archive_with("/abs/evil.txt"), out)
        self.assertTrue((out / "abs" / "evil.txt").exists())

    def test_validate_excludes(self) -> None:
        self.assertEqual(transfers.validate_excludes(None), [])
        self.assertEqual(transfers.validate_excludes([" .git "]), [".git"])
        for bad in (["a\nb"], [""], "notalist", ["x"] * 51):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                transfers.validate_excludes(bad)  # type: ignore[arg-type]

    def test_remote_argv_pass_paths_as_arguments(self) -> None:
        argv = transfers.remote_tar_create_argv("/a b/src", "/t/x.tar", ["*.ckpt"])
        self.assertEqual(argv[:2], ["bash", "-c"])
        self.assertEqual(argv[3:], ["_", "/a b/src", "/t/x.tar", "--exclude=*.ckpt"])
        self.assertEqual(transfers.parse_size("1234\t/path\n"), 1234)
        self.assertIsNone(transfers.parse_size(""))


if __name__ == "__main__":
    unittest.main()
