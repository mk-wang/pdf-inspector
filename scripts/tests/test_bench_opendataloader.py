import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench_opendataloader import (
    EVALUATOR_MODULES,
    BenchmarkFailure,
    _arguments,
    _print_report,
    _run_binary,
    _run_engine,
    compare_evaluations,
    evaluate_gates,
    main,
)


def evaluation(overall, documents, *, missing=0):
    return {
        "metrics": {
            "score": {
                "overall_mean": overall,
                "nid_mean": overall + 0.01,
            },
            "missing_predictions": missing,
        },
        "documents": [
            {
                "document_id": document_id,
                "scores": {"overall": score},
            }
            for document_id, score in documents.items()
        ],
    }


def write_executable(path, text):
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


BINARY_TEMPLATE = '''#!/usr/bin/env python3
import json
import pathlib
import sys

pdf = pathlib.Path(sys.argv[1])
if "--json" not in sys.argv[2:]:
    print("missing --json", file=sys.stderr)
    raise SystemExit(9)
__MARKER__if pdf.stem == "empty":
    print(json.dumps({
        "pdf_type": "image_based",
        "page_count": 1,
        "has_text": False,
        "markdown_length": 0,
        "pages_needing_ocr": [1],
        "is_complex": False,
        "pages_with_tables": [],
        "pages_with_columns": [],
        "has_encoding_issues": False,
        "markdown": "",
    }))
    raise SystemExit(0)
if pdf.stem == "broken":
    print("{not json")
    raise SystemExit(0)
if pdf.stem == "nonstring":
    print(json.dumps({
        "pdf_type": "text_based",
        "has_text": True,
        "pages_needing_ocr": [],
        "markdown": 7,
    }))
    raise SystemExit(0)
__FAILURE__markdown = "# %s body from %s" % (pdf.stem, __LABEL__) + chr(10)
print(json.dumps({
    "pdf_type": "text_based",
    "page_count": 1,
    "has_text": True,
    "markdown_length": len(markdown),
    "pages_needing_ocr": [],
    "is_complex": False,
    "pages_with_tables": [],
    "pages_with_columns": [],
    "has_encoding_issues": False,
    "markdown": markdown,
}))
'''


def fake_binary(path, label, *, marker=None, fails=True):
    marker_line = f"pathlib.Path({str(marker)!r}).write_text('ran')\n" if marker else ""
    failure_block = """if pdf.stem == "fail":
    print("simulated extraction failure", file=sys.stderr)
    raise SystemExit(3)
""" if fails else ""
    script = (
        BINARY_TEMPLATE.replace("__MARKER__", marker_line)
        .replace("__FAILURE__", failure_block)
        .replace("__LABEL__", repr(label))
    )
    write_executable(path, script)


FAKE_EVALUATOR = """#!/usr/bin/env python3
import argparse
import json
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("--prediction-root", required=True)
parser.add_argument("--ground-truth-dir", required=True)
parser.add_argument("--engine", required=True)
parser.add_argument("--output-filename", required=True)
parser.add_argument("--log-level")
args = parser.parse_args()

root = pathlib.Path(args.prediction_root)
markdown = root / args.engine / "markdown"
documents = [
    {
        "document_id": path.stem,
        "scores": {"overall": 0.5},
        "prediction_available": True,
    }
    for path in sorted(markdown.glob("*.md"))
]
score = 0.82 if args.engine == "candidate" else 0.80
payload = {
    "summary": {},
    "metrics": {
        "score": {
            "overall_mean": score,
            "nid_mean": score,
            "nid_s_mean": score,
            "teds_mean": score,
            "teds_s_mean": score,
            "mhs_mean": score,
            "mhs_s_mean": score,
        },
        "missing_predictions": 0,
    },
    "documents": documents,
}
(root / args.engine / args.output_filename).write_text(json.dumps(payload))
"""


def init_git(path):
    git = shutil.which("git")
    if git is None:
        raise unittest.SkipTest("git is unavailable")
    environment = [
        git,
        "-C",
        str(path),
        "-c",
        "user.name=Benchmark",
        "-c",
        "user.email=benchmark@example.invalid",
        "-c",
        "commit.gpgsign=false",
    ]
    subprocess.run([git, "init", "-q", str(path)], check=True)
    subprocess.run(environment + ["add", "."], check=True)
    subprocess.run(environment + ["commit", "-q", "-m", "bench"], check=True)


def make_bench(root, names=("alpha",), evaluator=FAKE_EVALUATOR, external=False):
    bench = root / "bench"
    (bench / "src").mkdir(parents=True)
    (bench / "pdfs").mkdir()
    (bench / "ground-truth" / "markdown").mkdir(parents=True)
    for name in names:
        (bench / "pdfs" / f"{name}.pdf").write_bytes(b"%PDF-1.4 " + name.encode())
        (bench / "ground-truth" / "markdown" / f"{name}.md").write_text(
            f"# {name}\n", encoding="utf-8"
        )
    (bench / "ground-truth" / "reference.json").write_text("{}\n", encoding="utf-8")
    (bench / "src" / "evaluator.py").write_text(evaluator, encoding="utf-8")
    for module in EVALUATOR_MODULES:
        path = bench / "src" / module
        if not path.exists():
            path.write_text("# placeholder\n", encoding="utf-8")
    if external:
        predictions = bench / "prediction" / "pdf-inspector" / "markdown"
        predictions.mkdir(parents=True)
        (predictions / "keep.md").write_text("external prediction\n", encoding="utf-8")
    init_git(bench)
    return bench


def run_main(bench, baseline, candidate, *extra):
    output = io.StringIO()
    errors = io.StringIO()
    with redirect_stdout(output), redirect_stderr(errors):
        code = main(
            [
                "--bench-dir",
                str(bench),
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
                "--python",
                sys.executable,
                *extra,
            ]
        )
    return code, output.getvalue(), errors.getvalue()


class ComparisonTests(unittest.TestCase):
    def test_reports_metric_and_document_deltas(self):
        baseline = evaluation(0.80, {"a": 0.8, "b": 0.6, "c": 0.7})
        candidate = evaluation(0.82, {"a": 0.9, "b": 0.5, "c": 0.7})

        result = compare_evaluations(baseline, candidate, top=1)

        self.assertAlmostEqual(result["deltas"]["overall_mean"], 0.02)
        self.assertEqual(result["documents"]["improved"], 1)
        self.assertEqual(result["documents"]["regressed"], 1)
        self.assertEqual(result["documents"]["unchanged"], 1)
        self.assertEqual(
            result["documents"]["largest_improvements"][0]["document_id"], "a"
        )
        self.assertEqual(
            result["documents"]["largest_regressions"][0]["document_id"], "b"
        )

    def test_reference_delta_is_reported(self):
        baseline = evaluation(0.80, {})
        candidate = evaluation(0.82, {})
        reference = evaluation(0.81, {})

        result = compare_evaluations(baseline, candidate, reference)

        self.assertAlmostEqual(
            result["candidate_vs_reference"]["overall_mean"], 0.01
        )

    def test_gates_cover_aggregate_document_missing_and_reference(self):
        comparison = compare_evaluations(
            evaluation(0.80, {"a": 0.8}),
            evaluation(0.79, {"a": 0.7}, missing=1),
            evaluation(0.81, {}),
        )

        failures = evaluate_gates(
            comparison,
            min_overall_delta=0.0,
            max_document_regression=0.05,
            max_missing=0,
            require_reference_lead=True,
        )

        self.assertEqual(len(failures), 4)

    def test_regression_gate_is_independent_of_report_limit(self):
        comparison = compare_evaluations(
            evaluation(0.80, {"a": 0.8}),
            evaluation(0.80, {"a": 0.7}),
            top=0,
        )

        failures = evaluate_gates(
            comparison,
            min_overall_delta=0.0,
            max_document_regression=0.05,
            max_missing=0,
            require_reference_lead=False,
        )

        self.assertEqual(len(failures), 1)
        self.assertIn("largest document regression", failures[0])

    def test_report_handles_reference_without_overall_score(self):
        result = compare_evaluations(
            evaluation(0.80, {}),
            evaluation(0.82, {}),
            {"metrics": {"score": {"nid_mean": 0.81}}},
        )

        output = io.StringIO()
        with redirect_stdout(output):
            _print_report(result)

        self.assertIn("Reference overall: n/a; candidate delta: n/a", output.getvalue())

    def test_reference_gate_reports_missing_score_as_unavailable(self):
        comparison = compare_evaluations(
            evaluation(0.80, {}),
            evaluation(0.82, {}),
        )

        failures = evaluate_gates(
            comparison,
            min_overall_delta=0.0,
            max_document_regression=None,
            max_missing=0,
            require_reference_lead=True,
        )

        self.assertEqual(failures, ["reference overall score is unavailable"])

    def test_arguments_reject_negative_counts_and_allow_zero_top(self):
        required = [
            "--bench-dir",
            ".",
            "--baseline",
            "baseline",
            "--candidate",
            "candidate",
        ]
        self.assertEqual(_arguments(required + ["--top", "0"]).top, 0)
        self.assertEqual(
            _arguments(required + ["--artifacts-dir", "artifacts"]).artifacts_dir,
            Path("artifacts"),
        )
        for option in ("--top", "--max-document-regression", "--max-missing"):
            with self.subTest(option=option), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    _arguments(required + [option, "-1"])

    def test_arguments_reject_nonfinite_float_thresholds(self):
        required = [
            "--bench-dir",
            ".",
            "--baseline",
            "baseline",
            "--candidate",
            "candidate",
        ]
        for option in ("--min-overall-delta", "--max-document-regression"):
            for value in ("nan", "inf", "-inf"):
                with self.subTest(option=option, value=value), redirect_stderr(
                    io.StringIO()
                ):
                    with self.assertRaises(SystemExit):
                        _arguments(required + [option, value])


class BinaryTests(unittest.TestCase):
    def test_nonzero_exit_identifies_binary_and_pdf_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "pdf2md"
            fake_binary(binary, "baseline")
            output = root / "fail.md"

            with self.assertRaises(BenchmarkFailure) as raised:
                _run_binary(binary, root / "fail.pdf", output)

            message = str(raised.exception)
            self.assertIn(str(binary), message)
            self.assertIn("fail.pdf", message)
            self.assertIn("simulated extraction failure", message)
            self.assertFalse(output.exists())

    def test_malformed_or_nonstring_json_fails_with_context(self):
        for stem in ("broken", "nonstring"):
            with self.subTest(stem=stem), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                binary = root / "pdf2md"
                fake_binary(binary, "baseline")
                output = root / f"{stem}.md"

                with self.assertRaises(BenchmarkFailure) as raised:
                    _run_binary(binary, root / f"{stem}.pdf", output)

                message = str(raised.exception)
                self.assertIn(str(binary), message)
                self.assertIn(f"{stem}.pdf", message)
                self.assertFalse(output.exists())


class EngineTests(unittest.TestCase):
    def test_run_engine_preserves_external_predictions_and_isolates_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bench = make_bench(root, names=("alpha",), external=True)
            baseline = root / "baseline-pdf2md"
            candidate = root / "candidate-pdf2md"
            fake_binary(baseline, "baseline")
            fake_binary(candidate, "candidate")
            external = bench / "prediction" / "pdf-inspector"
            before = {
                path.relative_to(external): path.read_bytes()
                for path in external.rglob("*")
                if path.is_file()
            }
            scratch = root / "scratch"
            scratch.mkdir()

            baseline_run = _run_engine(
                bench_dir=bench,
                python=Path(sys.executable),
                binary=baseline,
                label="baseline",
                scratch_root=scratch,
            )
            candidate_run = _run_engine(
                bench_dir=bench,
                python=Path(sys.executable),
                binary=candidate,
                label="candidate",
                scratch_root=scratch,
            )

            after = {
                path.relative_to(external): path.read_bytes()
                for path in external.rglob("*")
                if path.is_file()
            }
            self.assertEqual(before, after)
            self.assertEqual(
                (scratch / "baseline" / "markdown" / "alpha.md").read_text(
                    encoding="utf-8"
                ),
                "# alpha body from baseline\n",
            )
            self.assertEqual(
                (scratch / "candidate" / "markdown" / "alpha.md").read_text(
                    encoding="utf-8"
                ),
                "# alpha body from candidate\n",
            )
            self.assertAlmostEqual(
                baseline_run.evaluation["metrics"]["score"]["overall_mean"], 0.80
            )
            self.assertAlmostEqual(
                candidate_run.evaluation["metrics"]["score"]["overall_mean"], 0.82
            )

    def test_nonzero_binary_aborts_engine_without_that_prediction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bench = make_bench(root, names=("alpha", "fail"))
            binary = root / "pdf2md"
            fake_binary(binary, "candidate")
            scratch = root / "scratch"
            scratch.mkdir()

            with self.assertRaises(BenchmarkFailure) as raised:
                _run_engine(
                    bench_dir=bench,
                    python=Path(sys.executable),
                    binary=binary,
                    label="candidate",
                    scratch_root=scratch,
                )

            message = str(raised.exception)
            self.assertIn(str(binary), message)
            self.assertIn("fail.pdf", message)
            self.assertFalse((scratch / "candidate" / "markdown" / "fail.md").exists())
            self.assertFalse((scratch / "candidate" / "evaluation.json").exists())


class CommandTests(unittest.TestCase):
    def test_end_to_end_writes_artifacts_manifest_and_json_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bench = make_bench(root, names=("alpha",))
            baseline = root / "baseline-pdf2md"
            candidate = root / "candidate-pdf2md"
            fake_binary(baseline, "baseline")
            fake_binary(candidate, "candidate")
            artifacts = root / "artifacts"
            json_output = root / "comparison.json"

            code, output, errors = run_main(
                bench,
                baseline,
                candidate,
                "--artifacts-dir",
                str(artifacts),
                "--json-output",
                str(json_output),
            )

            self.assertEqual(code, 0, errors)
            self.assertIn("Benchmark gate passed", output)
            self.assertTrue((artifacts / "baseline" / "markdown" / "alpha.md").is_file())
            self.assertTrue(
                (artifacts / "candidate" / "markdown" / "alpha.md").is_file()
            )
            self.assertTrue((artifacts / "baseline" / "evaluation.json").is_file())
            self.assertTrue((artifacts / "candidate" / "evaluation.json").is_file())
            self.assertTrue((artifacts / "baseline" / "extraction.json").is_file())
            self.assertTrue((artifacts / "candidate" / "extraction.json").is_file())

            manifest = json.loads(
                (artifacts / "manifest.json").read_text(encoding="utf-8")
            )
            revision = subprocess.run(
                ["git", "-C", str(bench), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            baseline_hash = hashlib.sha256(baseline.read_bytes()).hexdigest()
            self.assertEqual(manifest["bench"]["revision"], revision)
            self.assertEqual(manifest["corpus"]["files"]["pdfs/alpha.pdf"], hashlib.sha256((bench / "pdfs" / "alpha.pdf").read_bytes()).hexdigest())
            self.assertIn(
                "ground-truth/markdown/alpha.md", manifest["ground_truth"]["files"]
            )
            self.assertIn("ground-truth/reference.json", manifest["ground_truth"]["files"])
            self.assertEqual(
                manifest["engines"]["baseline"]["binary"]["sha256"], baseline_hash
            )
            self.assertEqual(manifest["engines"]["baseline"]["documents"], 1)
            self.assertEqual(manifest["engines"]["baseline"]["empty_predictions"], 0)
            self.assertEqual(manifest["engines"]["baseline"]["ocr_required_count"], 0)
            self.assertEqual(
                manifest["engines"]["baseline"]["ocr_required_documents"], []
            )
            self.assertEqual(
                set(manifest["evaluator"]["modules"]), set(EVALUATOR_MODULES)
            )
            self.assertEqual(
                manifest["evaluator"]["python"]["path"],
                str(Path(sys.executable).absolute()),
            )

            comparison = json.loads(json_output.read_text(encoding="utf-8"))
            self.assertAlmostEqual(comparison["deltas"]["overall_mean"], 0.02)

    def test_ocr_required_empty_markdown_is_recorded_without_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bench = make_bench(root, names=("empty",))
            baseline = root / "baseline-pdf2md"
            candidate = root / "candidate-pdf2md"
            fake_binary(baseline, "baseline")
            fake_binary(candidate, "candidate")
            artifacts = root / "artifacts"

            code, output, errors = run_main(
                bench,
                baseline,
                candidate,
                "--artifacts-dir",
                str(artifacts),
            )

            self.assertEqual(code, 0, errors)
            self.assertEqual(
                output.count("1 predictions (1 empty, 1 OCR-required)"), 2
            )
            self.assertEqual(output.count("OCR-required: empty"), 2)
            markdown = artifacts / "candidate" / "markdown" / "empty.md"
            self.assertTrue(markdown.is_file())
            self.assertEqual(markdown.read_text(encoding="utf-8"), "")
            extraction = json.loads(
                (artifacts / "candidate" / "extraction.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(extraction["ocr_required_documents"], ["empty"])
            document = extraction["documents"][0]
            self.assertEqual(document["pdf_type"], "image_based")
            self.assertIs(document["has_text"], False)
            self.assertEqual(document["pages_needing_ocr"], [1])
            self.assertTrue(document["ocr_required"])
            manifest = json.loads(
                (artifacts / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["engines"]["candidate"]["empty_predictions"], 1)
            self.assertEqual(
                manifest["engines"]["candidate"]["ocr_required_count"], 1
            )
            self.assertEqual(
                manifest["engines"]["candidate"]["ocr_required_documents"],
                ["empty"],
            )

    def test_nonzero_binary_fails_without_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bench = make_bench(root, names=("alpha", "fail"))
            baseline = root / "baseline-pdf2md"
            candidate = root / "candidate-pdf2md"
            fake_binary(baseline, "baseline", fails=False)
            fake_binary(candidate, "candidate")
            artifacts = root / "artifacts"
            json_output = root / "comparison.json"

            code, _output, errors = run_main(
                bench,
                baseline,
                candidate,
                "--artifacts-dir",
                str(artifacts),
                "--json-output",
                str(json_output),
            )

            self.assertEqual(code, 1)
            self.assertIn(str(candidate), errors)
            self.assertIn("fail.pdf", errors)
            self.assertFalse(artifacts.exists())
            self.assertFalse(json_output.exists())

    def test_evaluator_failure_is_reported_without_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluator = "#!/usr/bin/env python3\nimport sys\nprint('evaluator exploded', file=sys.stderr)\nraise SystemExit(4)\n"
            bench = make_bench(root, names=("alpha",), evaluator=evaluator)
            baseline = root / "baseline-pdf2md"
            candidate = root / "candidate-pdf2md"
            fake_binary(baseline, "baseline")
            fake_binary(candidate, "candidate")
            artifacts = root / "artifacts"
            json_output = root / "comparison.json"

            code, _output, errors = run_main(
                bench,
                baseline,
                candidate,
                "--artifacts-dir",
                str(artifacts),
                "--json-output",
                str(json_output),
            )

            self.assertEqual(code, 1)
            self.assertIn("OpenDataLoader evaluator", errors)
            self.assertIn("evaluator exploded", errors)
            self.assertFalse(artifacts.exists())
            self.assertFalse(json_output.exists())

    def test_occupied_artifacts_destination_is_refused_before_running(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bench = make_bench(root, names=("alpha",))
            marker = root / "binary-ran"
            baseline = root / "baseline-pdf2md"
            candidate = root / "candidate-pdf2md"
            fake_binary(baseline, "baseline", marker=marker)
            fake_binary(candidate, "candidate", marker=marker)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            occupied = artifacts / "occupied.txt"
            occupied.write_text("keep\n", encoding="utf-8")

            code, _output, errors = run_main(
                bench,
                baseline,
                candidate,
                "--artifacts-dir",
                str(artifacts),
            )

            self.assertEqual(code, 1)
            self.assertIn("artifacts destination already exists", errors)
            self.assertFalse(marker.exists())
            self.assertEqual(occupied.read_text(encoding="utf-8"), "keep\n")
            self.assertEqual(sorted(path.name for path in artifacts.iterdir()), ["occupied.txt"])


if __name__ == "__main__":
    unittest.main()
