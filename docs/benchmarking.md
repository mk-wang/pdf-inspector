# Benchmarking against OpenDataLoader

The paired harness runs two `pdf2md` binaries directly over the same local
OpenDataLoader corpus, evaluates both outputs with the corpus evaluator, and
reports aggregate and per-document deltas. This avoids comparing results
produced from different corpus revisions or evaluator versions.

The harness does not invoke `src/pdf_parser.py`, does not import the Python
`pdf_inspector` package, and never reads or writes
`prediction/pdf-inspector`. Only `pdf2md` and the evaluator are used.

## Setup

Build the candidate, and prepare a released or worktree build as the baseline:

```bash
cargo build --release
```

The external OpenDataLoader checkout is public. Use the revision recorded by
the comparison, and clone it explicitly rather than relying on a mutable
default branch:

```bash
git clone https://github.com/firecrawl/opendataloader-bench.git \
  ../opendataloader-bench
git -C ../opendataloader-bench checkout \
  87e4cb9247b1fffc1b11d6dc2da885d5197defd4
```

That revision contains `pdfs/*.pdf`, `ground-truth/markdown/*.md`, and the
evaluator under `src/`. The evaluator reads the ground-truth Markdown, not
`ground-truth/reference.json`; the runner nevertheless hashes every file under
`ground-truth/` for its identity manifest, including that file when present.

Create a Python 3.13 environment with only the evaluator dependencies:

```bash
python3.13 -m venv ../opendataloader-bench/.venv
../opendataloader-bench/.venv/bin/pip install \
  apted==1.0.3 beautifulsoup4==4.15.0 lxml==6.1.3 rapidfuzz==3.14.6
```

A different benchmark or evaluator revision has a different identity. It needs
its own manifest and its own baseline/candidate run; do not compare manifests,
corpus hashes, or evaluator module hashes across revisions as if they described
the same benchmark.

The harness uses `BENCH/.venv/bin/python` by default; pass `--python` to use a
different interpreter. The interpreter path is kept unresolved so a virtualenv
launcher still sees its installed packages. `--artifacts-dir` additionally
requires the benchmark checkout to be a Git checkout, because the identity
manifest records its `HEAD` revision.

## Invocation

```bash
python3 scripts/bench_opendataloader.py \
  --bench-dir ../opendataloader-bench \
  --baseline ../pdf-inspector-main/target/release/pdf2md \
  --candidate target/release/pdf2md \
  --artifacts-dir /tmp/pdf-inspector-bench-$(git rev-parse --short HEAD) \
  --json-output /tmp/pdf-inspector-benchmark.json \
  --max-document-regression 0.02
```

Pass `--reference-evaluation path/to/evaluation.json` to report the candidate
delta against another evaluation, and add `--require-reference-lead` to make a
negative reference delta fail the run. By default, the candidate must not
regress the baseline overall score or introduce missing predictions. Use
`--min-overall-delta` to require a specific aggregate gain.

For every sorted `pdfs/*.pdf`, the harness runs:

```text
<binary> <pdf> --json
```

A successful process must emit one UTF-8 JSON object whose `markdown` value is
a string; that exact Markdown is written to the prediction file. The harness
preserves `pdf_type`, `has_text`, and `pages_needing_ocr` per document in
`<engine>/extraction.json`.

`ImageBased` and `Scanned` classifications are legitimate OCR-required domain
results: the CLI returns status 0 with `has_text: false` and an empty
`markdown`, the empty `.md` file is scored as an empty prediction, and the
document is counted and listed as OCR-required in the console output and
manifest. This is not a process failure and is not treated as a missing
prediction.

A nonzero exit still aborts the run with the binary path, PDF path, exit
status, and the last stderr line. Malformed JSON, a non-object payload, a
non-string `markdown` value, or non-UTF-8 output also aborts with the same
document/binary context. The harness does not interpret stderr to override the
exit status. No Markdown file is written for a failed document. Each engine is
converted into its own temporary directory (`baseline/` or `candidate/`), so
one cannot overwrite the other and the external `prediction/` tree is left
untouched.

After conversion, the harness invokes the existing evaluator once per engine:

```text
<python> <bench>/src/evaluator.py \
  --prediction-root <temporary-root> \
  --ground-truth-dir <bench>/ground-truth/markdown \
  --engine <label> \
  --output-filename evaluation.json --log-level WARNING
```

An evaluator failure or a missing `evaluation.json` aborts the run. `--json-output`
keeps the same comparison payload; no output file is written if conversion or
evaluation fails.

## Artifacts and identity

`--artifacts-dir` must not already exist; an occupied path is refused before
any binary runs. On success the directory contains:

```text
<artifacts>/
  manifest.json
  baseline/
    markdown/*.md
    extraction.json
    evaluation.json
    evaluation.csv
  candidate/
    markdown/*.md
    extraction.json
    evaluation.json
    evaluation.csv
```

`extraction.json` is the per-engine status record: one entry per processed PDF
with `document_id`, `pdf_type`, `has_text`, `pages_needing_ocr`,
`markdown_length`, `empty_markdown`, and `ocr_required`, plus the engine-level
empty and OCR-required document lists.

`manifest.json` records the benchmark checkout revision, SHA-256 for both
binaries, the aggregate and per-file hashes of `pdfs/` and `ground-truth/`,
SHA-256 for the evaluator modules plus the evaluator interpreter path and
version, and per-engine document, empty-prediction, and OCR-required
counts/document IDs. It is written once, after both engines have been
evaluated; a failed conversion or evaluation leaves no artifacts.

## Limitations

- The harness measures quality only. It runs the corpus sequentially in a
  temporary directory and does not report timings.
- `pdf2md` is invoked without OCR. A scanned or image-based PDF is classified
  as OCR-required and scored as an empty prediction; a document with no native
  text is not a process failure.
- The evaluator scores every ground-truth Markdown file. A corpus PDF without
  ground truth is still converted but is not scored, and a ground-truth file
  without a prediction counts as a missing prediction.
- Binary stdout must be UTF-8 JSON with a string `markdown`; malformed or
  non-string output is a conversion failure rather than being written lossily.

## Controlled regression check: 2026-10-20

The four variants used source `fe5a37322fdbaa760e295401f5adcb78c1a63ffd`,
`lopdf` revision `10afc64a2792cb6194d5507c36991bccb43c7bed`, and the
benchmark revision and dependency versions above. Each variant processed the
same 200 PDFs through the JSON CLI contract.

| Variant | Artifact filtering | Empty interior column guard | Overall | NID | TEDS | MHS |
|---|---|---|---:|---:|---:|---:|
| A | All artifacts | No | 0.871074 | 0.907629 | 0.798904 | 0.787551 |
| B | All artifacts | Yes | 0.872945 | 0.907805 | 0.808812 | 0.793826 |
| C | Only `Tr=3` artifacts | No | 0.874265 | 0.915466 | 0.800149 | 0.785562 |
| D | Only `Tr=3` artifacts | Yes | 0.874265 | 0.915466 | 0.800149 | 0.785562 |

Against A, variant B changed only document `01030000000150`. It restored the
standalone heading and a two-column table without restoring Artifact text.
No document metric decreased; the other 199 Markdown outputs were byte-identical.

Variants C and D were temporary policy probes, not production behavior.
Against B, retaining visible artifacts improved 71 documents and regressed
20 documents. Running headers such as `322 Yarrow` became false headings.
The worst overall document decrease was `0.099416`.

All variants classified `01030000000141` as OCR-required. Its empty native
prediction was recorded explicitly, not inferred from a failed process.

Keep Artifact suppression in production. Treat any visible-Artifact recovery
as a separate policy change, with its own semantic checks and document gates.
Earlier `--raw` adapter results used a different failure contract; do not mix
those numbers with this controlled comparison.

### Upstream v1.20.0 comparison

A separate run compared the rebuilt upstream tag `v1.20.0` (`f568230`) with
the production guard build. It used the same JSON runner, corpus, evaluator,
and Python evaluator environment as the controlled variants.

| Metric | Upstream v1.20.0 | Production guard build | Delta |
|---|---:|---:|---:|
| Overall | 0.874093 | 0.872945 | -0.001148 |
| NID | 0.915479 | 0.907805 | -0.007675 |
| TEDS | 0.800102 | 0.808812 | +0.008710 |
| MHS | 0.784925 | 0.793826 | +0.008902 |

The production build improved 32 documents, regressed 79, and left 89 unchanged.
Eleven document decreases exceeded `0.02`; both the aggregate and document
gates failed. There were no missing predictions.

| Document | Upstream overall | Production overall | Delta |
|---|---:|---:|---:|
| `01030000000085` | 0.698508 | 0.497950 | -0.200557 |
| `01030000000149` | 0.971927 | 0.875378 | -0.096549 |
| `01030000000148` | 0.424832 | 0.356332 | -0.068500 |
| `01030000000150` | 0.837380 | 0.777267 | -0.060113 |
| `01030000000158` | 0.988055 | 0.938782 | -0.049273 |

For documents `00085`, `00149`, and `00150`, probe D matched the upstream
metrics exactly. This isolates their remaining differences to the Artifact
policy, but does not justify the probe's regressions elsewhere.

The guard is a verified improvement over A, not proof that the branch is
regression-free against upstream. The upstream tag has no tracked `Cargo.lock`;
the run retained its resolved lockfile with the evaluation artifacts.

## Published comparison protocol

The public benchmark table was refreshed on July 31, 2026, on an Apple M4 Pro
using pdf-inspector 0.2.6, LiteParse 2.10.1, OpenDataLoader 2.2.1,
PyMuPDF4LLM 0.2.0, and MarkItDown 0.1.5. Every engine processed the same 200
PDFs sequentially in a single process with OCR disabled. Reported speed is the
median of five alternating or rotating complete corpus runs after an excluded
warm-up run; quality scores come from the benchmark evaluator over all 200
outputs. Raw timings, predictions, evaluations, and charts are available in the
[results branch](https://github.com/firecrawl/opendataloader-bench/tree/abi/pdf-parser-benchmark-results).

## Optional backend evidence probe

The evidence probe compares positioned `pdf2md` items with MuPDF structured
text on the same pages. It is intended to find deterministic extraction or
layout evidence that could justify a future native implementation; it does not
merge MuPDF output into Markdown, invoke OCR, or add a runtime dependency.

Install MuPDF's `mutool`, build `pdf2md`, then run:

```bash
python3 scripts/probe_backend_evidence.py document.pdf \
  --pdf2md target/release/pdf2md \
  --json-output /tmp/backend-evidence.json
```

The report flags pages when MuPDF exposes a material net token gain, repeated
alignment anchors absent from local evidence, or additional image blocks. The
JSON includes bounded token samples and page-level counts so promising cases
can be inspected without treating backend disagreement as automatically
correct. Thresholds are configurable with `--min-token-gain`,
`--min-alternate-only-ratio`, and `--min-anchor-gain`.
