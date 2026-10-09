"""Reproduce the Relation and LAMBADA phase-removal accuracy sweep in Figure 2."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import json
import math
from pathlib import Path
import unicodedata

import torch
import torch.nn.functional as F

from evaluate.common import atomic_json, exclusive_lock, sha256, signature
from interventions.pg19 import ROOT, intervention_software_identity, load_model
from interventions.phase_zero_accuracy.run import ZeroAngles
from mamba_ssm.utils.paper_kernel_policy import apply_paper_kernel_policy


PAPER_RELATIONS = (
    "country_capital_city",
    "person_occupation",
    "person_plays_pro_sport",
    "company_hq",
    "product_by_company",
)
MAX_LENGTH = 2048


@dataclass(frozen=True)
class Request:
    index: int
    input_ids: tuple[int, ...]
    continuation_ids: tuple[int, ...]


def parse_widths(value: str) -> tuple[int, ...]:
    widths: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lower, upper = (int(part) for part in item.split("-", 1))
            if lower > upper:
                raise argparse.ArgumentTypeError(f"descending range: {item}")
            widths.update(range(lower, upper + 1))
        else:
            widths.add(int(item))
    if not widths or min(widths) < 1:
        raise argparse.ArgumentTypeError("widths must be positive")
    return tuple(sorted(widths))


def windows(layers: int, widths: tuple[int, ...]):
    if max(widths) > layers:
        raise ValueError(f"window width exceeds the model's {layers} layers")
    return [
        (width, start, tuple(range(start, start + width)))
        for width in widths
        for start in range(layers - width + 1)
    ]


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().strip().split())


def object_prefix(token: str, target: str) -> bool:
    prediction, target = normalized(token), normalized(target)
    return bool(prediction) and target.startswith(prediction)


def encode_pair(tokenizer, context: str, continuation: str):
    """Match lm-evaluation-harness v0.4.2 TemplateLM._encode_pair."""
    trailing = len(context) - len(context.rstrip())
    if trailing:
        continuation = context[-trailing:] + continuation
        context = context[:-trailing]
    if not context:
        context_ids = [int(tokenizer.eos_token_id)]
        continuation_ids = tokenizer.encode(continuation, add_special_tokens=False)
    else:
        whole = tokenizer.encode(context + continuation, add_special_tokens=False)
        context_ids = tokenizer.encode(context, add_special_tokens=False)
        continuation_ids = whole[len(context_ids):]
    if not context_ids or not continuation_ids:
        raise ValueError("empty context or continuation tokenization")
    return tuple(context_ids), tuple(continuation_ids)


def load_relation(data_root: Path, tokenizer, max_per_relation: int, limit: int):
    examples, paths = [], []
    for relation_name in PAPER_RELATIONS:
        path = data_root / f"{relation_name}.json"
        paths.append(path)
        relation = json.loads(path.read_text(encoding="utf-8"))
        samples = relation["samples"]
        if max_per_relation:
            samples = samples[:max_per_relation]
        template = relation["prompt_templates"][0]
        for relation_index, sample in enumerate(samples):
            prompt = template.format(sample["subject"])
            ids = tokenizer.encode(prompt, add_special_tokens=True)
            examples.append({
                "index": len(examples),
                "relation": relation_name,
                "relation_index": relation_index,
                "subject": sample["subject"],
                "object": sample["object"],
                "input_ids": tuple(ids),
            })
    return examples[:limit] if limit else examples, paths


def load_lambada(path: Path, tokenizer, limit: int):
    examples, requests = [], []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            text = str(json.loads(line)["text"])
            pieces = text.split(" ")
            if len(pieces) < 2 or not pieces[-1]:
                raise ValueError(f"invalid LAMBADA row {len(examples)}")
            context, target = " ".join(pieces[:-1]), " " + pieces[-1]
            context_ids, continuation_ids = encode_pair(tokenizer, context, target)
            combined = context_ids + continuation_ids
            input_ids = combined[-(MAX_LENGTH + 1):][:-1]
            if len(continuation_ids) > MAX_LENGTH or not input_ids:
                raise ValueError(f"LAMBADA row {len(examples)} exceeds the scoring window")
            index = len(examples)
            examples.append({
                "index": index,
                "target": target,
                "context_tokens": len(context_ids),
                "target_tokens": len(continuation_ids),
            })
            requests.append(Request(index, tuple(input_ids), continuation_ids))
            if limit and len(examples) >= limit:
                break
    return examples, requests


def manager(model, layers):
    return contextlib.nullcontext() if layers is None else ZeroAngles(model, layers)


@torch.inference_mode()
def relation_scores(model, tokenizer, examples, target_layers, batch_size, device):
    by_length: dict[int, list[dict]] = {}
    for example in examples:
        by_length.setdefault(len(example["input_ids"]), []).append(example)
    rows = []
    vocabulary = min(model.config.vocab_size, len(tokenizer))
    for length in sorted(by_length):
        group = by_length[length]
        for offset in range(0, len(group), batch_size):
            batch = group[offset:offset + batch_size]
            ids = torch.tensor([row["input_ids"] for row in batch], device=device)
            with manager(model, target_layers):
                logits = model(ids, num_last_tokens=1).logits[:, -1, :vocabulary]
            predicted = logits.float().argmax(dim=-1).tolist()
            for example, token_id in zip(batch, predicted):
                token = tokenizer.decode([token_id])
                rows.append({
                    "index": example["index"],
                    "relation": example["relation"],
                    "relation_index": example["relation_index"],
                    "predicted_token_id": int(token_id),
                    "correct": object_prefix(token, example["object"]),
                })
    rows.sort(key=lambda row: row["index"])
    return rows


@torch.inference_mode()
def lambada_scores(model, requests, target_layers, batch_size, pad_multiple, device):
    ordered = sorted(requests, key=lambda row: (-len(row.input_ids), row.input_ids))
    rows = []
    vocabulary = int(model.config.vocab_size)
    for offset in range(0, len(ordered), batch_size):
        batch = ordered[offset:offset + batch_size]
        longest = max(len(row.input_ids) for row in batch)
        padded = math.ceil(longest / pad_multiple) * pad_multiple if pad_multiple else longest
        if padded > MAX_LENGTH:
            raise ValueError(f"padded length {padded} exceeds {MAX_LENGTH}")
        ids = torch.zeros((len(batch), padded), dtype=torch.long, device=device)
        for batch_index, request in enumerate(batch):
            ids[batch_index, :len(request.input_ids)] = torch.tensor(
                request.input_ids, dtype=torch.long, device=device,
            )
        with manager(model, target_layers):
            hidden = model.backbone(ids)
        max_target = max(len(row.continuation_ids) for row in batch)
        selected = hidden.new_zeros((len(batch), max_target, hidden.shape[-1]))
        for batch_index, request in enumerate(batch):
            count, end = len(request.continuation_ids), len(request.input_ids)
            selected[batch_index, -count:] = hidden[batch_index, end-count:end]
        # Keep the BF16 log-softmax used by lm-evaluation-harness for exact
        # agreement with the reported baseline and intervention sweep.
        log_probs = F.log_softmax(model.lm_head(selected)[..., :vocabulary], dim=-1)
        for batch_index, request in enumerate(batch):
            targets = torch.tensor(request.continuation_ids, dtype=torch.long, device=device)
            scores = log_probs[batch_index, -len(targets):]
            rows.append({
                "index": request.index,
                "loglikelihood": float(scores.gather(1, targets[:, None]).sum()),
                "correct": bool((scores.argmax(dim=-1) == targets).all()),
            })
    rows.sort(key=lambda row: row["index"])
    return rows


def summarize(task: str, rows: list[dict]):
    correct = sum(row["correct"] for row in rows)
    result = {"examples": len(rows), "correct": correct, "accuracy": correct / len(rows)}
    if task == "lambada":
        mean_ll = sum(row["loglikelihood"] for row in rows) / len(rows)
        result.update(mean_target_loglikelihood=mean_ll, perplexity=math.exp(-mean_ll))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("relation", "lambada"), required=True)
    parser.add_argument("--model", choices=("siso", "mimo"), required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "evaluate/assets/tokenizer")
    parser.add_argument("--relation-data", type=Path, default=ROOT / "data/eval/relation")
    parser.add_argument(
        "--lambada-data", type=Path,
        default=ROOT / "data/eval/benchmarks/lambada_openai/test.jsonl",
    )
    parser.add_argument("--widths", type=parse_widths, default=parse_widths("1-24"))
    parser.add_argument(
        "--batch-size", type=int,
        help="default: 32 for Relation and 64 for LAMBADA, matching the paper runs",
    )
    parser.add_argument("--pad-multiple", type=int, default=64)
    parser.add_argument("--max-samples-per-relation", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-kernel-policy", action="store_true")
    args = parser.parse_args()
    if args.batch_size is None:
        args.batch_size = 32 if args.task == "relation" else 64
    if (
        args.batch_size < 1
        or args.pad_multiple < 0
        or args.limit < 0
        or args.max_samples_per_relation < 0
    ):
        parser.error(
            "batch size must be positive; pad multiple and sample limits "
            "must be nonnegative"
        )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, clean_up_tokenization_spaces=False,
    )
    if args.task == "relation":
        examples, data_paths = load_relation(
            args.relation_data, tokenizer, args.max_samples_per_relation, args.limit,
        )
        requests = None
    else:
        examples, requests = load_lambada(args.lambada_data, tokenizer, args.limit)
        data_paths = [args.lambada_data]
    if not examples:
        raise ValueError("the selected dataset is empty")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(device)
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model_path = args.model_path or (
        ROOT / f"checkpoints/officialpretrained/mamba3-{args.model}-1.5b"
    )
    model, checkpoint = load_model(args.model, model_path, device)
    policy = None if args.no_kernel_policy else apply_paper_kernel_policy(args.model)
    conditions = [("native", None, None, None)] + [
        (f"width_{width:02d}_start_{start:02d}", width, start, layers)
        for width, start, layers in windows(len(model.backbone.layers), args.widths)
    ]
    output = args.output or ROOT / "interventions/results/figure2" / args.model / args.task
    output.mkdir(parents=True, exist_ok=True)
    spec = {
        "version": 1, "paper_figure": 2, "task": args.task, "model": args.model,
        "checkpoint": checkpoint, "tokenizer": str(args.tokenizer.resolve()),
        "tokenizer_sha256": sha256(args.tokenizer / "tokenizer.json"),
        "datasets": {str(path.resolve()): sha256(path) for path in data_paths},
        "examples": len(examples), "widths": list(args.widths),
        "conditions": len(conditions), "batch_size": args.batch_size,
        "pad_multiple": args.pad_multiple if args.task == "lambada" else None,
        "max_samples_per_relation": args.max_samples_per_relation,
        "limit": args.limit, "kernel_policy": policy,
        "software": intervention_software_identity(),
        "relation_metric": "decoded top-1 token is a non-empty casefolded prefix of gold object",
        "lambada_metric": "all continuation tokens equal greedy argmax",
    }
    run_signature = signature(spec)
    with exclusive_lock(output / "run.lock"):
        metadata_path = output / "metadata.json"
        if metadata_path.exists():
            old = json.loads(metadata_path.read_text())
            if old["signature"] != run_signature:
                raise ValueError(f"output belongs to a different run: {output}")
        else:
            atomic_json(metadata_path, {"signature": run_signature, "spec": spec})
        summaries = []
        for number, (condition, width, start, target_layers) in enumerate(conditions, 1):
            shard = output / f"{condition}.json"
            if shard.exists():
                payload = json.loads(shard.read_text())
                shard_is_valid = (
                    payload.get("signature") == run_signature
                    and payload.get("condition") == condition
                    and payload.get("width") == width
                    and payload.get("start_layer") == start
                    and len(payload.get("examples", [])) == len(examples)
                    and payload.get("summary", {}).get("examples") == len(examples)
                )
                if not shard_is_valid:
                    raise ValueError(f"incompatible shard: {shard}")
            else:
                rows = (
                    relation_scores(
                        model,
                        tokenizer,
                        examples,
                        target_layers,
                        args.batch_size,
                        device,
                    )
                    if args.task == "relation"
                    else lambada_scores(
                        model, requests, target_layers, args.batch_size, args.pad_multiple, device,
                    )
                )
                payload = {
                    "signature": run_signature, "condition": condition,
                    "width": width, "start_layer": start,
                    "end_layer_exclusive": None if width is None else start + width,
                    "summary": summarize(args.task, rows), "examples": rows,
                }
                atomic_json(shard, payload)
            summaries.append({
                "condition": condition, "width": width, "start_layer": start,
                **payload["summary"],
            })
            atomic_json(output / "summary.json", {
                "signature": run_signature, "complete": number == len(conditions),
                "completed_conditions": number, "expected_conditions": len(conditions),
                "native_accuracy": summaries[0]["accuracy"], "conditions": [
                    {**row, "delta_accuracy": row["accuracy"] - summaries[0]["accuracy"]}
                    for row in summaries
                ],
            })
            print(
                f"{number}/{len(conditions)} task={args.task} model={args.model} "
                f"condition={condition} accuracy={payload['summary']['accuracy']:.6f}",
                flush=True,
            )
    print(f"COMPLETE: {output}", flush=True)


if __name__ == "__main__":
    main()
