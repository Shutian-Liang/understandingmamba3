"""Download the public assets used by evaluation and interventions."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from zipfile import ZipFile

from evaluate.common import ROOT, TASKS


SNAPSHOTS = {
    "tokenizer": (
        "NousResearch/Meta-Llama-3.1-8B", None,
        "1f47e50cdbe801ad8a5174156ec3a0655108fb9f",
        ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"),
        ROOT / "evaluate/assets/tokenizer",
    ),
    "siso": (
        "state-spaces/mamba3-siso-1.5b", None,
        "5cfc721542ec9ccee768088b2fd6b7e8101219d8",
        ("config.json", "pytorch_model.bin"),
        ROOT / "checkpoints/officialpretrained/mamba3-siso-1.5b",
    ),
    "mimo": (
        "state-spaces/mamba3-mimo-1.5b", None,
        "bc6b5d0f7994fe4cb3478242e92da8daf9ee29ec",
        ("config.json", "pytorch_model.bin"),
        ROOT / "checkpoints/officialpretrained/mamba3-mimo-1.5b",
    ),
    "math_hard": (
        "lighteval/MATH-Hard", "dataset",
        "cf0716b8bafa192bcb6b455b0679538787dc43f0",
        ("train/*.jsonl",), ROOT / "data/eval/math_hard",
    ),
    "codeparrot": (
        "codeparrot/codeparrot-clean-valid", "dataset",
        "4db92d2ec0c1b4c41eeb439cfae16854511d9dcd",
        ("*.json.gz",), ROOT / "data/eval/codeparrot_clean_valid",
    ),
    "trivia_qa": (
        "mandarjoshi/trivia_qa", "dataset",
        "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f",
        ("rc/validation-*.parquet",), ROOT / "data/eval/trivia_qa_rc",
    ),
    "slimpajama": (
        "fla-hub/slimpajama-test", "dataset",
        "cbfff1661ee798cdf3d911cf27ecd4fb2ea8028d",
        ("data/*.parquet",), ROOT / "data/eval/slimpajama_test",
    ),
    "pg19": (
        "fla-hub/pg19", "dataset",
        "217f9837c7bc0f95e57984ffbfead40939abc451",
        ("data/validation-*.parquet", "data/test-*.parquet"),
        ROOT / "data/eval/pg19",
    ),
    "longbench": (
        "zai-org/LongBench", "dataset",
        "5e628be450b7e67fb7ae6e201bd6d8f7056f7672",
        ("data.zip",), ROOT / "data/eval/longbench_e",
    ),
}

GROUPS = {
    "models": ("tokenizer", "siso", "mimo"),
    "ppl": ("math_hard", "codeparrot", "trivia_qa", "slimpajama"),
    "all": tuple(SNAPSHOTS),
}


def download_snapshot(name: str) -> None:
    from huggingface_hub import snapshot_download

    repository, repo_type, revision, patterns, destination = SNAPSHOTS[name]
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repository, repo_type=repo_type, revision=revision,
        allow_patterns=list(patterns), local_dir=destination,
    )
    if name == "pg19":
        for source in sorted((destination / "data").glob("*.parquet")):
            target = destination / source.name
            shutil.copy2(source, target)
    if name == "longbench":
        with ZipFile(destination / "data.zip") as archive:
            for task in TASKS:
                filename = f"{task}_e.jsonl"
                matches = [member for member in archive.namelist()
                           if Path(member).name == filename]
                if len(matches) != 1:
                    raise ValueError(f"Expected one LongBench member for {filename}: {matches}")
                target = destination / "data" / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(matches[0]))
    print(f"READY {name}: {destination.relative_to(ROOT)}", flush=True)


def expand(values: list[str]) -> list[str]:
    output = []
    for value in values:
        selected = GROUPS.get(value, (value,))
        for name in selected:
            if name not in output:
                output.append(name)
    return output


def main() -> None:
    names = tuple(SNAPSHOTS) + ("models", "ppl", "all")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", nargs="+", choices=names, required=True)
    args = parser.parse_args()
    for name in expand(args.assets):
        download_snapshot(name)


if __name__ == "__main__":
    main()
