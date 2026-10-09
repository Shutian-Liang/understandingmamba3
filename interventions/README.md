# Paper intervention evaluation

Run every command from the repository root after preparing the released checkpoints and evaluation data described in the main README. The default paths are:

- `checkpoints/officialpretrained/mamba3-{siso,mimo}-1.5b`
- `evaluate/assets/tokenizer`
- `data/eval/pg19/{test,validation}-00000-of-00001.parquet`
- `data/eval/benchmarks/lambada_openai/test.jsonl`
- `data/eval/relation/*.json`

All reported pretrained-model results use BF16 weights, batch size 1 for PG-19, and the bundled paper-time Triton policy. Each output directory contains input hashes, the resolved protocol, raw per-window or per-book records, and an aggregate summary. An interrupted run resumes only when its stored signature matches the requested configuration.

## Figure 2: phase removal by layer window

`phase_zero_accuracy.ZeroAngles` zeros the learned phase increment at every token in selected layers. Figure 2 evaluates every contiguous layer window: for width `w`, the start layer is `0..24-w`, giving 300 windows per model across widths 1--24.

Relation uses the first prompt template from each of the five bundled LRE relations and batch size 32. A prediction is correct when the decoded top-1 next token is a non-empty, case-insensitive prefix of the gold object. LAMBADA uses batch size 64 and follows the zero-shot `lambada_openai` protocol: the final literal-space-delimited word is the continuation, and accuracy requires greedy agreement on every continuation token. Run every width and position with:

```bash
python -m interventions.accuracy --task relation --model siso
python -m interventions.accuracy --task lambada --model siso
```

Repeat with `--model mimo`. Each completed condition is an atomic JSON shard, so rerunning the same command resumes the sweep. The intervention itself can also be applied directly:

```python
from interventions.phase_zero_accuracy.run import ZeroAngles

with ZeroAngles(model, layers=range(start_layer, start_layer + width)):
    logits = model(input_ids).logits
```

## Table 5: phase and decay interventions on PG-19

Run each condition for both `siso` and `mimo`:

```bash
python -m interventions.pg19 --table 5 --model siso --condition native
python -m interventions.pg19 --table 5 --model siso --condition decay_permutation
python -m interventions.pg19 --table 5 --model siso --condition phase_shuffle
python -m interventions.pg19 --table 5 --model siso --condition phase_removal
python -m interventions.pg19 --table 5 --model siso --condition phase_reversal
```

The contexts are 2K, 16K, and 64K. For a book of `L` tokens and context `C`, starts are `range(0, L-C, max(10, (L-C)//10))`; exact-fit books are excluded. The evaluator scores the final 100 next-token targets in every window and reports `exp(total NLL / total target tokens)`.

The paper labels map to code as follows:

| Paper condition | Implementation |
|---|---|
| decay permutation | `rope_a_mix_1` |
| phase shuffle | `phase_within_token_oscillator_shuffle` |
| phase removal | `zero_angles` |
| phase reversal | `phase_reverse` |

Phase shuffle independently permutes rotary-pair increments at every layer, head, and token. Decay permutation uses one fixed cyclic donor-head shift per layer and applies it only to rotary state coordinates; non-rotary coordinates remain native.

## Table 6: recurrent-state statistics

Run native and phase-removed trajectories for both models:

```bash
python -m interventions.state_statistics --model siso --condition native
python -m interventions.state_statistics --model siso --condition phase_removal
```

The evaluator uses every validation book with at least 64K tokens. Each book is decoded continuously from a zero recurrent state. It excludes the first 2K tokens from measurements without resetting the state, then records the same trajectory at 16K and 64K. Statistics are temporal means per layer/head followed by the median over layer/head; books receive equal weight. `state_rms` is the square root of the median temporal mean squared Frobenius norm. CUDA graph replay is enabled by default; `--eager` is the equivalent slower path.

## Table 7: B/C phase removal over the predictor interval

Run each condition for both models:

```bash
python -m interventions.pg19 --table 7 --model siso --condition native
python -m interventions.pg19 --table 7 --model siso --condition remove_b
python -m interventions.pg19 --table 7 --model siso --condition remove_c
python -m interventions.pg19 --table 7 --model siso --condition remove_both
```

Table 7 uses the 64K PG-19 windows from the Table 5 protocol. B/write and C/read phase are removed only at predictor positions for the final 100 targets.

## Short execution checks

Use `--smoke` to run one deterministic PG-19 window per requested context:

```bash
python -m interventions.pg19 \
  --table 5 --model siso --condition phase_shuffle \
  --contexts 2000 --smoke
```

Use one eligible book and a shorter trajectory for the state-statistics execution path:

```bash
python -m interventions.state_statistics \
  --model siso --condition native \
  --lengths 4000 --warmup 2000 --max-books 1
```

For a short Figure 2 execution check, select the all-layer window and a small example limit:

```bash
python -m interventions.accuracy \
  --task lambada --model siso --widths 24 --limit 8 \
  --output /tmp/mamba3-figure2-smoke
```
