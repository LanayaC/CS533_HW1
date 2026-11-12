
"""Export the Part 4.2 retrieval runs to TREC format."""

import csv
import math
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "data"
RUN_SOURCES = DATA / "part3_results"
EVALUATION = DATA / "part4_results"
OUTPUT = ROOT / "runs"

TEST_SETS = {
    "ki1_c1": ("KI1", 1000, "C1"),
    "ki2_c2": ("KI2", 1000, "C2"),
    "topics_c1": ("T", 32, "C1"),
    "topics_c2": ("T", 32, "C2"),
}


def expected_queries(query_type, count):
    if query_type == "T":
        return {f"T{i:02d}" for i in range(1, count + 1)}
    return {f"{query_type}-{i:04d}" for i in range(1, count + 1)}


def export_run(source, destination, query_ids, collection, tag):
    by_query = defaultdict(list)

    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["qid", "docno", "rank", "score"]:
            raise ValueError(f"Unexpected CSV columns: {source}")

        for row in reader:
            qid = row["qid"]
            if qid not in query_ids:
                continue

            docno = row["docno"]
            rank = int(row["rank"])
            score = float(row["score"])

            if not docno.startswith(f"{collection}-"):
                raise ValueError(f"Incorrect collection: {qid}, {docno}")
            if not math.isfinite(score):
                raise ValueError(f"Invalid score: {qid}, {docno}")

            by_query[qid].append((rank, docno, row["score"]))

    if set(by_query) != query_ids:
        raise ValueError(f"Missing queries in {source.name}")

    lines = []

    for qid in sorted(query_ids):
        results = sorted(by_query[qid])

        if [rank for rank, _, _ in results] != list(range(1, 101)):
            raise ValueError(f"Ranks are not 1-100: {source.name}, {qid}")

        if len({docno for _, docno, _ in results}) != 100:
            raise ValueError(f"Duplicate documents: {source.name}, {qid}")

        for rank, docno, score in results:
            lines.append(f"{qid} Q0 {docno} {rank} {score} {tag}\n")

    destination.write_text("".join(lines), encoding="utf-8")
    return len(lines)


def main():
    OUTPUT.mkdir(exist_ok=True)
    total_files = 0
    total_lines = 0

    for test_set, (query_type, count, collection) in TEST_SETS.items():
        table = EVALUATION / f"part42_{test_set}.csv"

        with table.open(newline="", encoding="utf-8") as handle:
            methods = list(csv.DictReader(handle))

        if len(methods) != 15:
            raise ValueError(f"Expected 15 methods in {table.name}")

        names = [row["method"] for row in methods]
        if len(set(names)) != 15:
            raise ValueError(f"Duplicate methods in {table.name}")

        query_ids = expected_queries(query_type, count)

        for row in methods:
            method = row["method"]
            source_name = row["source_file"]

            if Path(source_name).name != source_name:
                raise ValueError(f"Unexpected source path: {source_name}")

            source = RUN_SOURCES / source_name
            if not source.is_file():
                raise FileNotFoundError(source)

            method_slug = re.sub(
                r"[^a-z0-9]+", "_", method.lower()
            ).strip("_")

            destination = OUTPUT / f"{test_set}_{method_slug}.trec"
            tag = f"cs533_{method_slug}"

            if destination.exists():
                raise FileExistsError(
                    f"Output already exists: {destination}"
                )

            line_count = export_run(
                source, destination, query_ids, collection, tag
            )

            print(f"{destination.name}: {line_count:,} lines")
            total_files += 1
            total_lines += line_count

    if total_files != 60 or total_lines != 3_096_000:
        raise ValueError("Unexpected export totals")

    print(f"\nSUCCESS: {total_files} TREC files")
    print(f"Total ranking records: {total_lines:,}")


if __name__ == "__main__":
    main()
