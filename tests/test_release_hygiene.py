from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReleaseHygieneTests(unittest.TestCase):
    def _release_text_files(self) -> list[Path]:
        suffixes = {
            "",
            ".cff",
            ".in",
            ".json",
            ".lock",
            ".md",
            ".py",
            ".toml",
            ".txt",
            ".yaml",
            ".yml",
        }
        return [
            path
            for path in ROOT.rglob("*")
            if path.is_file()
            and (path.suffix.lower() in suffixes or path.name.startswith(".git"))
            and not any(
                part in {".git", ".venv", "build", "dist", "tmp"}
                for part in path.parts
            )
        ]

    def _release_markdown_files(self) -> list[Path]:
        return [
            path
            for path in ROOT.rglob("*.md")
            if not any(
                part in {".git", ".venv", "build", "dist", "tmp"}
                for part in path.parts
            )
        ]

    def test_release_versions_are_aligned(self) -> None:
        init_text = (ROOT / "src" / "thetascan" / "__init__.py").read_text(
            encoding="utf-8"
        )
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")

        init_version = re.search(r'__version__\s*=\s*"([^"]+)"', init_text)
        project_version = re.search(
            r'^version\s*=\s*"([^"]+)"', pyproject, flags=re.MULTILINE
        )
        citation_version = re.search(
            r'^version:\s*"([^"]+)"', citation, flags=re.MULTILINE
        )
        self.assertIsNotNone(init_version)
        self.assertIsNotNone(project_version)
        self.assertIsNotNone(citation_version)
        self.assertEqual(
            init_version.group(1),
            project_version.group(1),
        )
        self.assertEqual(
            init_version.group(1),
            citation_version.group(1),
        )

    def test_readme_relative_links_exist(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        targets = re.findall(r"\[[^\]]+\]\(([^)]+)\)", readme)
        missing: list[str] = []
        for target in targets:
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path_text = target.split("#", 1)[0]
            if path_text and not (ROOT / path_text).exists():
                missing.append(target)
        self.assertEqual(missing, [])

    def test_all_release_markdown_relative_links_exist(self) -> None:
        missing: list[str] = []
        for document in self._release_markdown_files():
            text = document.read_text(encoding="utf-8")
            targets = re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)
            for target in targets:
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                path_text = target.split("#", 1)[0]
                if path_text and not (document.parent / path_text).resolve().exists():
                    missing.append(f"{document.relative_to(ROOT)}: {target}")
        self.assertEqual(missing, [])

    def test_no_private_desktop_or_agent_paths_in_release_text(self) -> None:
        suffixes = {".cff", ".json", ".md", ".py", ".toml", ".yaml", ".yml"}
        forbidden = (
            "C:\\Users\\user\\",
            "/Users/user/",
            ".codex/attachments",
            ".claude/",
            "agent-memory",
        )
        hits: list[str] = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            # The test necessarily contains the forbidden literals it enforces.
            if path.resolve() == Path(__file__).resolve():
                continue
            if any(part in {".git", ".venv", "build", "dist", "tmp"} for part in path.parts):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for token in forbidden:
                if token in text:
                    hits.append(f"{path.relative_to(ROOT)}: {token}")
        self.assertEqual(hits, [])

    def test_public_release_text_is_english_only(self) -> None:
        self.assertFalse((ROOT / "docs" / "USER_GUIDE_RU.md").exists())
        hits: list[str] = []
        for document in self._release_text_files():
            text = document.read_text(encoding="utf-8", errors="replace")
            if re.search(r"[\u0400-\u04ff]", text):
                hits.append(str(document.relative_to(ROOT)))
        self.assertEqual(hits, [])

    def test_paper_and_legal_boundaries_exist(self) -> None:
        required = (
            "LICENSE",
            "LICENSING.md",
            "PATENTS.md",
            "THIRD_PARTY_NOTICES.md",
            "paper/ThetaScan-Scan-Parallel-Nonlinear-Memory.md",
            "paper/ThetaScan-Scan-Parallel-Nonlinear-Memory.pdf",
        )
        self.assertEqual([name for name in required if not (ROOT / name).exists()], [])

    def test_paper_pdf_contains_no_local_file_references(self) -> None:
        pdf_bytes = (
            ROOT / "paper" / "ThetaScan-Scan-Parallel-Nonlinear-Memory.pdf"
        ).read_bytes().lower()
        forbidden = (b"file:", b"c:/users/", b"c:\\users\\")
        self.assertEqual(
            [token.decode("ascii") for token in forbidden if token in pdf_bytes],
            [],
        )


if __name__ == "__main__":
    unittest.main()
