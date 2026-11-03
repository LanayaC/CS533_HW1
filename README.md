# CS533_HW1
Information Retrieval: Word Embeddings, Semantic Drift, and News Retrieval

## Project status

Development and submission scaffold only. No assignment algorithms, entry points, experiments, or results have been implemented.

## Layout

- `src/`: assignment source code.
- `runs/`: required TREC-format retrieval run files, with up to 100 documents per query (`qid Q0 docno rank score tag`).
- `tests/`: future correctness checks and small synthetic fixtures.
- `notes/`: development notes, assumptions, and experiment planning.
- `report/`: report source, selected figures and tables, and the final PDF.
- `requirements.txt`: initial software dependencies listed on page 2 of the assignment PDF.
- `.gitignore`: exclusions for local environments, datasets, models, caches, and bulk generated artifacts.
- `.gitkeep` files: empty placeholders so Git can retain the initial folders.

## Environment and data

Use Python 3.10–3.12. Dependencies have not been installed or compatibility-tested.

The official test collection is stored separately at `C:\Projects\IR\data\nyt-tc-2026`. Use the official collection for experiments; the assignment also requires rebuilding it and reporting differences. Do not copy datasets into this repository.

Keep trained models, corpora, indexes, virtual environments, and large generated artifacts outside Git. Keep required text run files in `runs/` and the final report PDF in `report/` eligible for version control. Review new files before staging; `.gitignore` matches names and paths, not file sizes.

## Running and submission

There are no runnable assignment commands or measured runtimes yet. Document exact commands and timings on your machine here as development progresses.

Submit the code with `src/`, `requirements.txt`, and this README; include the required run files and a report PDF with one section per assignment part. Include the required student details and an AI-use paragraph at the end of the report, disclosing Codex assistance with PDF review and repository scaffolding.
