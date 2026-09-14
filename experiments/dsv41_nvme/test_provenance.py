"""Contract test: every implementation's bytes must match recorded provenance.

`source-provenance.json` records the sha256 of each implementation. The packed
K2-K8 records, cancellation, alignment, checksum and lease behaviour under review
are exactly those bytes, so silent drift here invalidates the review.

Drift is never waived. An implementation that was deliberately repaired during
review must declare `extracted_without_code_changes: false` AND carry a
`review_repair` record whose patch artifact is itself hash-pinned and whose
contents are checked against the live source, so the deviation is enumerated,
reproducible and reviewable instead of being excused. There is no skip, xfail or
tolerance path in this file.
"""
import hashlib
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parent
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def digest_of(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patch_line_sets(text):
    """Removed and added lines of a unified diff, headers and hunk markers dropped."""
    removed, added = [], []
    for line in text.splitlines():
        if line.startswith(("--- ", "+++ ", "@@", "diff ", "index ", "\\")):
            continue
        if line.startswith("-"):
            removed.append(line[1:])
        elif line.startswith("+"):
            added.append(line[1:])
    return [l for l in removed if l.strip()], [l for l in added if l.strip()]


class ProvenanceTests(unittest.TestCase):
    def test_implementation_bytes_match_recorded_provenance(self):
        recorded = json.loads((ROOT / "source-provenance.json").read_text())["sources"]
        self.assertTrue(recorded, "provenance file must pin the extracted implementations")
        for name, entry in recorded.items():
            source = ROOT / name
            self.assertTrue(source.is_file(), f"{name} is missing")
            digest = digest_of(source)
            self.assertEqual(digest, entry["sha256"], f"{name} drifted from its recorded provenance")
            self.assertIn("extracted_without_code_changes", entry, f"{name} must declare its extraction status")
            if not entry["extracted_without_code_changes"]:
                self.assert_repaired_entry_is_fully_pinned(name, entry)

    def assert_repaired_entry_is_fully_pinned(self, name, entry):
        """A repaired implementation is only acceptable as a fully pinned deviation."""
        repair = entry.get("review_repair")
        self.assertIsInstance(repair, dict, f"{name} changed but declares no review_repair record")
        for field in ("id", "subject", "patch", "patch_sha256", "target_sha256", "what_changed"):
            self.assertTrue(repair.get(field), f"{name} review_repair is missing {field}")
        original = entry.get("original_extraction_sha256")
        self.assertRegex(original or "", HEX64, f"{name} must preserve the original extraction hash")
        self.assertRegex(repair["patch_sha256"], HEX64)
        self.assertRegex(repair["target_sha256"], HEX64)
        self.assertNotEqual(
            entry["sha256"], original,
            f"{name} claims unchanged bytes while declaring a repair",
        )
        self.assertEqual(
            repair["target_sha256"], entry["sha256"],
            f"{name} review_repair.target_sha256 must equal the pinned live hash",
        )
        patch = ROOT / repair["patch"]
        self.assertTrue(patch.is_file(), f"{name} review_repair patch artifact is missing")
        self.assertEqual(
            digest_of(patch), repair["patch_sha256"],
            f"{name} review_repair patch artifact drifted from its recorded hash",
        )
        text = patch.read_text()
        self.assertIn(name, text, f"{name} review_repair patch must touch {name}")
        live = (ROOT / name).read_text().splitlines()
        removed, added = patch_line_sets(text)
        self.assertTrue(added, f"{name} review_repair patch adds nothing")
        for line in added:
            self.assertIn(
                line, live,
                f"{name} review_repair patch claims an added line that is not in the live source: {line!r}",
            )
        for line in removed:
            self.assertNotIn(
                line, live,
                f"{name} review_repair patch claims a removed line that is still in the live source: {line!r}",
            )
        return True

    def test_pinned_implementations_are_not_imported_as_an_installed_backend(self):
        # The components stay an isolated experiment: no production module may
        # import them, and no plugin entry point may register them.
        text = ROOT.parent.parent.joinpath("pyproject.toml").read_text()
        for name in ("expert_store", "expert_cache", "async_store", "engram_rows"):
            self.assertNotIn(name, text)


if __name__ == "__main__":
    unittest.main()
