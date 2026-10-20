#!/usr/bin/env python3
"""Run a paired pdf-inspector OpenDataLoader benchmark and report deltas.

Both binaries are invoked directly over the corpus PDFs. The OpenDataLoader
parser adapter and the Python ``pdf_inspector`` package are not used; only the
existing Markdown evaluator is invoked for scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCORE_KEYS = (
    "overall_mean",
    "nid_mean",
    "nid_s_mean",
    "teds_mean",
    "teds_s_mean",
    "mhs_mean",
    "mhs_s_mean",
)

EVALUATOR_MODULES = (
    "converter_markdown_table.py",
    "evaluator.py",
    "evaluator_heading_level.py",
    "evaluator_reading_order.py",
    "evaluator_table.py",
)

OCR_REQUIRED_TYPES = frozenset({"scanned", "image_based"})


class BenchmarkFailure(RuntimeError):
    """A benchmark prerequisite or subprocess contract was violated."""


@dataclass(frozen=True)
class EngineRun:
    evaluation: dict[str, Any]
    prediction_count: int
    empty_predictions: int
    ocr_required_documents: tuple[str, ...]


@dataclass(frozen=True)
class DocumentResult:
    markdown: str
    status: dict[str, Any]
    ocr_required: bool


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite")
    return parsed


def _scores(evaluation: dict[str, Any]) -> dict[str, float]:
    score = evaluation.get("metrics", {}).get("score", {})
    return {key: float(score[key]) for key in SCORE_KEYS if score.get(key) is not None}


def _documents(evaluation: dict[str, Any]) -> dict[str, float]:
    documents: dict[str, float] = {}
    for document in evaluation.get("documents", []):
        overall = document.get("scores", {}).get("overall")
        if overall is not None:
            documents[str(document["document_id"])] = float(overall)
    return documents


def compare_evaluations(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    reference: dict[str, Any] | None = None,
    *,
    top: int = 10,
) -> dict[str, Any]:
    """Build aggregate and per-document deltas from evaluator JSON payloads."""
    baseline_scores = _scores(baseline)
    candidate_scores = _scores(candidate)
    metric_deltas = {
        key: candidate_scores[key] - baseline_scores[key]
        for key in SCORE_KEYS
        if key in baseline_scores and key in candidate_scores
    }

    baseline_documents = _documents(baseline)
    candidate_documents = _documents(candidate)
    shared = sorted(baseline_documents.keys() & candidate_documents.keys())
    document_deltas = [
        {
            "document_id": document_id,
            "baseline": baseline_documents[document_id],
            "candidate": candidate_documents[document_id],
            "delta": candidate_documents[document_id] - baseline_documents[document_id],
        }
        for document_id in shared
    ]
    epsilon = 1e-12
    improvements = sorted(document_deltas, key=lambda item: item["delta"], reverse=True)
    regressions = sorted(document_deltas, key=lambda item: item["delta"])

    result: dict[str, Any] = {
        "baseline": baseline_scores,
        "candidate": candidate_scores,
        "deltas": metric_deltas,
        "missing_predictions": {
            "baseline": int(baseline.get("metrics", {}).get("missing_predictions", 0)),
            "candidate": int(candidate.get("metrics", {}).get("missing_predictions", 0)),
        },
        "documents": {
            "shared": len(shared),
            "improved": sum(item["delta"] > epsilon for item in document_deltas),
            "regressed": sum(item["delta"] < -epsilon for item in document_deltas),
            "unchanged": sum(abs(item["delta"]) <= epsilon for item in document_deltas),
            "largest_improvements": [
                item for item in improvements if item["delta"] > epsilon
            ][:top],
            "largest_regressions": [
                item for item in regressions if item["delta"] < -epsilon
            ][:top],
            "worst_regression": next(
                (item for item in regressions if item["delta"] < -epsilon), None
            ),
        },
    }
    if reference is not None:
        reference_scores = _scores(reference)
        result["reference"] = reference_scores
        result["candidate_vs_reference"] = {
            key: candidate_scores[key] - reference_scores[key]
            for key in SCORE_KEYS
            if key in candidate_scores and key in reference_scores
        }
    return result


def evaluate_gates(
    comparison: dict[str, Any],
    *,
    min_overall_delta: float,
    max_document_regression: float | None,
    max_missing: int,
    require_reference_lead: bool,
) -> list[str]:
    """Return human-readable gate failures; an empty list means pass."""
    failures: list[str] = []
    overall_delta = comparison["deltas"].get("overall_mean")
    if overall_delta is None or overall_delta < min_overall_delta:
        failures.append(
            f"overall delta {overall_delta!r} is below {min_overall_delta:+.6f}"
        )
    candidate_missing = comparison["missing_predictions"]["candidate"]
    if candidate_missing > max_missing:
        failures.append(
            f"candidate has {candidate_missing} missing predictions (maximum {max_missing})"
        )
    if max_document_regression is not None:
        regression = comparison["documents"].get("worst_regression")
        if regression is not None and regression["delta"] < -max_document_regression:
            failures.append(
                "largest document regression "
                f"{regression['document_id']}={regression['delta']:+.6f} "
                f"exceeds {-max_document_regression:+.6f}"
            )
    if require_reference_lead:
        reference_delta = comparison.get("candidate_vs_reference", {}).get("overall_mean")
        if reference_delta is None:
            failures.append("reference overall score is unavailable")
        elif reference_delta < 0.0:
            failures.append(
                f"candidate trails reference overall by {reference_delta!r}"
            )
    return failures


def _tail(text: str, limit: int = 500) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    tail = lines[-1]
    return tail if len(tail) <= limit else tail[: limit - 3] + "..."


def _run_checked(
    command: list[str], *, cwd: Path, description: str
) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command), flush=True)
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        raise BenchmarkFailure(f"{description} could not start: {exc}") from exc
    if completed.returncode != 0:
        detail = _tail(completed.stderr or completed.stdout)
        suffix = f": {detail}" if detail else ""
        raise BenchmarkFailure(
            f"{description} failed (exit {completed.returncode}){suffix}"
        )
    return completed


def _run_binary(binary: Path, pdf_path: Path, output_path: Path) -> DocumentResult:
    """Convert one PDF through the JSON CLI contract."""
    command = [str(binary), str(pdf_path), "--json"]
    print("+", " ".join(command), flush=True)
    try:
        completed = subprocess.run(command, capture_output=True, check=False)
    except OSError as exc:
        raise BenchmarkFailure(
            f"cannot run {binary} for {pdf_path}: {exc}"
        ) from exc
    if completed.returncode != 0:
        detail = _tail(completed.stderr.decode("utf-8", errors="replace"))
        suffix = f": {detail}" if detail else ""
        raise BenchmarkFailure(
            f"{binary} exited {completed.returncode} for {pdf_path}{suffix}"
        )
    try:
        raw = completed.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BenchmarkFailure(
            f"{binary} produced non-UTF-8 output for {pdf_path}: {exc}"
        ) from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BenchmarkFailure(
            f"{binary} returned malformed JSON for {pdf_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise BenchmarkFailure(
            f"{binary} returned a non-object JSON payload for {pdf_path}"
        )
    markdown = payload.get("markdown")
    if not isinstance(markdown, str):
        raise BenchmarkFailure(
            f"{binary} returned non-string Markdown for {pdf_path}: "
            f"{type(markdown).__name__}"
        )

    pdf_type = payload.get("pdf_type")
    ocr_required = (
        isinstance(pdf_type, str) and pdf_type.lower() in OCR_REQUIRED_TYPES
    )
    status = {
        "document_id": pdf_path.stem,
        "pdf_type": pdf_type,
        "has_text": payload.get("has_text"),
        "pages_needing_ocr": payload.get("pages_needing_ocr"),
        "markdown_length": len(markdown),
        "empty_markdown": not markdown,
        "ocr_required": ocr_required,
    }
    try:
        output_path.write_bytes(markdown.encode("utf-8"))
    except OSError as exc:
        raise BenchmarkFailure(
            f"cannot write prediction {output_path} for {pdf_path}: {exc}"
        ) from exc
    return DocumentResult(
        markdown=markdown,
        status=status,
        ocr_required=ocr_required,
    )


def _evaluate(
    *, bench_dir: Path, python: Path, scratch_root: Path, label: str
) -> dict[str, Any]:
    evaluator = bench_dir / "src" / "evaluator.py"
    _run_checked(
        [
            str(python),
            str(evaluator),
            "--prediction-root",
            str(scratch_root),
            "--ground-truth-dir",
            str(bench_dir / "ground-truth" / "markdown"),
            "--engine",
            label,
            "--output-filename",
            "evaluation.json",
            "--log-level",
            "WARNING",
        ],
        cwd=bench_dir,
        description=f"OpenDataLoader evaluator for {label}",
    )
    evaluation_path = scratch_root / label / "evaluation.json"
    if not evaluation_path.is_file():
        raise BenchmarkFailure(
            f"OpenDataLoader evaluator for {label} did not write {evaluation_path}"
        )
    try:
        payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkFailure(
            f"cannot read evaluation for {label} at {evaluation_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise BenchmarkFailure(
            f"evaluation for {label} is not a JSON object: {evaluation_path}"
        )
    return payload


def _run_engine(
    *,
    bench_dir: Path,
    python: Path,
    binary: Path,
    label: str,
    scratch_root: Path,
) -> EngineRun:
    pdfs = sorted((bench_dir / "pdfs").glob("*.pdf"))
    if not pdfs:
        raise BenchmarkFailure(f"no corpus PDFs found in {bench_dir / 'pdfs'}")

    destination = scratch_root / label
    if destination.exists():
        raise BenchmarkFailure(f"engine scratch directory already exists: {destination}")
    markdown_dir = destination / "markdown"
    try:
        markdown_dir.mkdir(parents=True)
    except OSError as exc:
        raise BenchmarkFailure(
            f"cannot create prediction directory {markdown_dir}: {exc}"
        ) from exc

    empty_predictions = 0
    ocr_required_documents: list[str] = []
    statuses: list[dict[str, Any]] = []
    for pdf_path in pdfs:
        result = _run_binary(binary, pdf_path, markdown_dir / f"{pdf_path.stem}.md")
        statuses.append(result.status)
        if not result.markdown:
            empty_predictions += 1
        if result.ocr_required:
            ocr_required_documents.append(pdf_path.stem)

    extraction = {
        "engine": label,
        "empty_predictions": empty_predictions,
        "ocr_required_count": len(ocr_required_documents),
        "ocr_required_documents": ocr_required_documents,
        "documents": statuses,
    }
    try:
        (destination / "extraction.json").write_text(
            json.dumps(extraction, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        raise BenchmarkFailure(
            f"cannot write extraction status {destination / 'extraction.json'}: {exc}"
        ) from exc

    evaluation = _evaluate(
        bench_dir=bench_dir,
        python=python,
        scratch_root=scratch_root,
        label=label,
    )
    print(
        f"{label}: {len(pdfs)} predictions "
        f"({empty_predictions} empty, {len(ocr_required_documents)} OCR-required)",
        flush=True,
    )
    if ocr_required_documents:
        print(f"  OCR-required: {', '.join(ocr_required_documents)}", flush=True)
    return EngineRun(
        evaluation=evaluation,
        prediction_count=len(pdfs),
        empty_predictions=empty_predictions,
        ocr_required_documents=tuple(ocr_required_documents),
    )


def _print_report(comparison: dict[str, Any]) -> None:
    print("\nMetric                 baseline   candidate       delta")
    print("--------------------  ----------  ----------  ----------")
    for key in SCORE_KEYS:
        if key not in comparison["deltas"]:
            continue
        print(
            f"{key:<20}  {comparison['baseline'][key]:>10.6f}  "
            f"{comparison['candidate'][key]:>10.6f}  "
            f"{comparison['deltas'][key]:>+10.6f}"
        )
    if "reference" in comparison:
        delta = comparison["candidate_vs_reference"].get("overall_mean")
        reference = comparison["reference"].get("overall_mean")
        reference_display = f"{reference:.6f}" if reference is not None else "n/a"
        delta_display = f"{delta:+.6f}" if delta is not None else "n/a"
        print(f"\nReference overall: {reference_display}; candidate delta: {delta_display}")

    documents = comparison["documents"]
    print(
        "\nDocuments: "
        f"{documents['improved']} improved, {documents['regressed']} regressed, "
        f"{documents['unchanged']} unchanged ({documents['shared']} shared)"
    )
    for heading, key in (
        ("Largest improvements", "largest_improvements"),
        ("Largest regressions", "largest_regressions"),
    ):
        print(f"\n{heading}:")
        rows = documents[key]
        if not rows:
            print("  none")
        for row in rows:
            print(
                f"  {row['document_id']}: {row['delta']:+.6f} "
                f"({row['baseline']:.6f} -> {row['candidate']:.6f})"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_hashes(bench_dir: Path, relative: str) -> dict[str, Any]:
    directory = bench_dir / relative
    files: dict[str, str] = {}
    try:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                files[path.relative_to(bench_dir).as_posix()] = _sha256(path)
    except OSError as exc:
        raise BenchmarkFailure(f"cannot hash identity files under {directory}: {exc}") from exc
    if not files:
        raise BenchmarkFailure(f"no identity files found under {directory}")
    aggregate = hashlib.sha256()
    for name, file_hash in files.items():
        aggregate.update(f"{name}\0{file_hash}\n".encode("utf-8"))
    return {
        "directory": str(directory),
        "sha256": aggregate.hexdigest(),
        "files": files,
    }


def _bench_revision(bench_dir: Path) -> str:
    completed = _run_checked(
        ["git", "-C", str(bench_dir), "rev-parse", "HEAD"],
        cwd=bench_dir,
        description=f"git revision for {bench_dir}",
    )
    revision = completed.stdout.strip()
    if not revision:
        raise BenchmarkFailure(f"git revision for {bench_dir} is empty")
    return revision


def _python_version(python: Path) -> str:
    completed = _run_checked(
        [str(python), "-c", "import sys; print(sys.version.split()[0])"],
        cwd=Path.cwd(),
        description=f"Python version for {python}",
    )
    return completed.stdout.strip()


def _evaluator_identity(bench_dir: Path, python: Path) -> dict[str, Any]:
    module_hashes: dict[str, str] = {}
    for module in EVALUATOR_MODULES:
        path = bench_dir / "src" / module
        try:
            module_hashes[module] = _sha256(path)
        except OSError as exc:
            raise BenchmarkFailure(f"cannot hash evaluator module {path}: {exc}") from exc
    return {
        "python": {"path": str(python), "version": _python_version(python)},
        "modules": module_hashes,
    }


def _binary_identity(binary: Path) -> dict[str, str]:
    try:
        return {"path": str(binary), "sha256": _sha256(binary)}
    except OSError as exc:
        raise BenchmarkFailure(f"cannot hash binary {binary}: {exc}") from exc


def _identity(
    *, bench_dir: Path, binaries: dict[str, Path], evaluator: dict[str, Any]
) -> dict[str, Any]:
    return {
        "bench": {"path": str(bench_dir), "revision": _bench_revision(bench_dir)},
        "corpus": _directory_hashes(bench_dir, "pdfs"),
        "ground_truth": _directory_hashes(bench_dir, "ground-truth"),
        "evaluator": evaluator,
        "binaries": {
            label: _binary_identity(binary)
            for label, binary in binaries.items()
        },
    }


def _manifest(
    *, identity: dict[str, Any], runs: dict[str, EngineRun]
) -> dict[str, Any]:
    return {
        "bench": identity["bench"],
        "corpus": identity["corpus"],
        "ground_truth": identity["ground_truth"],
        "evaluator": identity["evaluator"],
        "engines": {
            label: {
                "binary": identity["binaries"][label],
                "documents": runs[label].prediction_count,
                "empty_predictions": runs[label].empty_predictions,
                "ocr_required_count": len(runs[label].ocr_required_documents),
                "ocr_required_documents": list(
                    runs[label].ocr_required_documents
                ),
                "evaluation": f"{label}/evaluation.json",
                "extraction": f"{label}/extraction.json",
            }
            for label in identity["binaries"]
        },
    }


def _require_absent(path: Path) -> None:
    if path.exists():
        raise BenchmarkFailure(f"artifacts destination already exists: {path}")


def _write_artifacts(
    *,
    destination: Path,
    scratch_root: Path,
    labels: tuple[str, ...],
    manifest: dict[str, Any],
) -> None:
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise BenchmarkFailure(
            f"artifacts destination already exists: {destination}"
        ) from exc
    except OSError as exc:
        raise BenchmarkFailure(
            f"cannot create artifacts destination {destination}: {exc}"
        ) from exc
    try:
        for label in labels:
            shutil.copytree(scratch_root / label, destination / label)
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, shutil.Error) as exc:
        raise BenchmarkFailure(
            f"cannot write artifacts to {destination}: {exc}"
        ) from exc
    print(f"\nArtifacts written to {destination}", flush=True)


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--reference-evaluation", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--artifacts-dir", type=Path)
    parser.add_argument("--top", type=_non_negative_int, default=10)
    parser.add_argument("--min-overall-delta", type=_finite_float, default=0.0)
    parser.add_argument("--max-document-regression", type=_non_negative_float)
    parser.add_argument("--max-missing", type=_non_negative_int, default=0)
    parser.add_argument("--require-reference-lead", action="store_true")
    return parser.parse_args(argv)


def _execute(args: argparse.Namespace) -> int:
    bench_dir = args.bench_dir.resolve()
    baseline = args.baseline.resolve()
    candidate = args.candidate.resolve()
    # Keep the virtualenv launcher path intact. Resolving its symlink would
    # invoke the underlying system interpreter without the benchmark's site
    # packages.
    python = (args.python or bench_dir / ".venv" / "bin" / "python").absolute()
    artifacts_dir = (
        args.artifacts_dir.absolute() if args.artifacts_dir is not None else None
    )
    if artifacts_dir is not None:
        _require_absent(artifacts_dir)

    for path, description in (
        (bench_dir / "src" / "evaluator.py", "OpenDataLoader evaluator"),
        (bench_dir / "ground-truth" / "markdown", "OpenDataLoader ground truth"),
        (bench_dir / "pdfs", "OpenDataLoader PDF corpus"),
        (baseline, "baseline binary"),
        (candidate, "candidate binary"),
        (python, "Python interpreter"),
    ):
        if not path.exists():
            raise BenchmarkFailure(f"{description} not found: {path}")

    reference = None
    if args.reference_evaluation is not None:
        reference_path = args.reference_evaluation.resolve()
        try:
            reference = json.loads(reference_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BenchmarkFailure(
                f"cannot read reference evaluation {reference_path}: {exc}"
            ) from exc

    binaries = {"baseline": baseline, "candidate": candidate}
    labels = ("baseline", "candidate")
    identity = None
    if artifacts_dir is not None:
        identity = _identity(
            bench_dir=bench_dir,
            binaries=binaries,
            evaluator=_evaluator_identity(bench_dir, python),
        )

    with tempfile.TemporaryDirectory(prefix="pdf-inspector-opendataloader-") as temporary:
        scratch_root = Path(temporary)
        runs = {
            label: _run_engine(
                bench_dir=bench_dir,
                python=python,
                binary=binaries[label],
                label=label,
                scratch_root=scratch_root,
            )
            for label in labels
        }
        comparison = compare_evaluations(
            runs["baseline"].evaluation,
            runs["candidate"].evaluation,
            reference,
            top=args.top,
        )

        if identity is not None:
            _write_artifacts(
                destination=artifacts_dir,
                scratch_root=scratch_root,
                labels=labels,
                manifest=_manifest(identity=identity, runs=runs),
            )

    _print_report(comparison)
    if args.json_output is not None:
        json_output = args.json_output.resolve()
        try:
            json_output.write_text(
                json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
            )
        except OSError as exc:
            raise BenchmarkFailure(
                f"cannot write JSON output {json_output}: {exc}"
            ) from exc

    failures = evaluate_gates(
        comparison,
        min_overall_delta=args.min_overall_delta,
        max_document_regression=args.max_document_regression,
        max_missing=args.max_missing,
        require_reference_lead=args.require_reference_lead,
    )
    if failures:
        print("\nBenchmark gate failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("\nBenchmark gate passed.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    try:
        return _execute(args)
    except BenchmarkFailure as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
