# Mamba-3 experiments

This codebase is built on the official [Mamba implementation](https://github.com/state-spaces/mamba) developed by Albert Gu, Tri Dao, and other contributors. The shared model, CUDA/Triton, and package structure originate from that project; this repository adds state tracking, pretraining, evaluation, and phase-intervention implementations. See [LICENSE](LICENSE) for licensing terms.

## 1. Environment

All experiments share one Python 3.10, Linux x86-64, NVIDIA CUDA environment:

~~~bash
conda create -n name --override-channels \
  -c nvidia/label/cuda-12.8.1 -c conda-forge \
  python=3.10 cuda-nvcc cuda-cudart-dev cuda-cccl \
  'gcc_linux-64<14' 'gxx_linux-64<14' -y
conda activate name

export CUDA_HOME="$CONDA_PREFIX"
export TILELANG_CACHE_DIR="$PWD/.cache/tilelang"
export TRITON_CACHE_DIR="$PWD/.cache/triton"
export HF_HOME="$PWD/.cache/huggingface"

python -m pip install --upgrade pip
python -m pip install -c requirements-lock.txt setuptools wheel ninja
python -m pip install torch==2.9.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -c requirements-lock.txt \
  -e '.[training,evaluation]' --no-build-isolation
python -m pip check
~~~

`requirements-lock.txt` records the tested versions. CUDA 12.8 compiler headers are required by TileLang 0.1.8; the PyTorch CUDA wheel alone does not provide them. Export `CUDA_HOME` again after activating the environment in a new shell.

Check the installation:

~~~bash
python - <<'PY'
import torch, triton, tilelang, mamba_ssm
print("CUDA available:", torch.cuda.is_available())
print("PyTorch:", torch.__version__)
print("Triton:", triton.__version__)
print("Mamba source:", mamba_ssm.__file__)
PY
~~~

## 2. Model downloads

Download the Llama-3.1 tokenizer and official Mamba-3 SISO/MIMO 1.5B checkpoints:

~~~bash
python -m evaluate.download --assets models
~~~

Files are written to:

~~~text
evaluate/assets/tokenizer/
checkpoints/officialpretrained/mamba3-siso-1.5b/
checkpoints/officialpretrained/mamba3-mimo-1.5b/
~~~

The official 1.5B checkpoints can be used with the phase-intervention modules. The 187M/444M experiments produce local `.pt` checkpoints during pretraining. Use `hf auth login` first if a Hugging Face repository requires authentication.

## 3. Data downloads

The downloader pins the explicit dataset revisions and writes each dataset to the path expected by the code:

~~~bash
# MATH-Hard, CodeParrot, TriviaQA, and SlimPajama
python -m evaluate.download --assets ppl

# PG-19 validation and test
python -m evaluate.download --assets pg19

# LongBench-E
python -m evaluate.download --assets longbench
~~~

To download the tokenizer, official models, and every dataset above in one command:

~~~bash
python -m evaluate.download --assets all
~~~

Prepare the four PPL corpora and fixed PG-19 probe:

~~~bash
python -m evaluate.prepare \
  --datasets math_hard codeparrot trivia_qa slimpajama
python -m pretrain.prepare pg19
~~~

The prepared paths are:

~~~text
evaluate/cache/codeparrot/
evaluate/cache/slimpajama/
evaluate/cache/paper_math_hard/
evaluate/cache/paper_trivia_qa/
data/pg19_test/mamba3_llama31_fixed_probe/
data/eval/longbench_e/data/
~~~

LM Evaluation Harness downloads its standard task data on first use. FineWeb-Edu is streamed and tokenized by `python -m pretrain.prepare tokens` in the pretraining section.

## 4. State tracking

State-tracking data is generated during training, so this experiment needs no downloaded dataset or pretrained checkpoint.

List the retained recipes:

~~~bash
bash state_tracking/scripts/sweep.sh --list
~~~

Run one model:

~~~bash
bash state_tracking/scripts/train.sh \
  --task mod3 --model underdamped --seed 42 \
  --lr 0.0007196856730011514
~~~

Launch a sweep:

~~~bash
bash state_tracking/scripts/sweep.sh \
  --task mod3 --model underdamped
~~~

Available tasks are `parity`, `mod3`, and `arithmetic`. Available models are `mamba2`, `no_rotation`, `static_rotation`, `overdamped`, `critical`, and `underdamped`.

The recipes train for 10,000 steps, grow the training maximum length from 40 to 160, and evaluate at lengths 160, 192, and 224. Warmup is 500 steps for Parity/Mod-3 and 1,000 steps for Arithmetic. `train.sh` and `sweep.sh` use `nohup`; logs are written under `state_tracking/logs/`.

Each run writes checkpoints and metrics to `state_tracking/outputs/<task>/<model>/.../`. Evaluate an existing run with:

~~~bash
python -m state_tracking.evaluate \
  /absolute/path/to/run/checkpoints/best.pt \
  --lengths 160,192,224
~~~

## 5. Pretraining

Four configurations cover MIMO 187M/444M with RoPE fractions 0.0/0.5. They train for 100B tokens with sequence length 2,048:

~~~text
pretrain/configs/
├── mamba3_mimo_187m_rope0_fineweb_edu_llama31_100b.json
├── mamba3_mimo_187m_rope05_fineweb_edu_llama31_100b.json
├── mamba3_mimo_444m_rope0_fineweb_edu_llama31_100b.json
└── mamba3_mimo_444m_rope05_fineweb_edu_llama31_100b.json
~~~

Stream the pinned FineWeb-Edu source and build the training/validation token shards:

~~~bash
python -m pretrain.prepare tokens
~~~

The default recipe retains 100B training tokens and 10M validation tokens under `data/fineweb_edu_llama31_100B/`. Token files use flat uint32 Llama-3.1 IDs. The full token set requires about 400 GB. To use local Parquet instead, pass `--input '/path/to/*.parquet'`.

Select a config and output directory, then submit training:

~~~bash
RUN_NAME=mamba3_mimo_187m_rope0_fineweb_edu_llama31_100b
CONFIG="pretrain/configs/$RUN_NAME.json"
OUT_DIR="$PWD/training_outputs/$RUN_NAME"

export MAMBA_PYTHON="$(command -v python)"
bash pretrain/slurm/submit.sh "$CONFIG" --output-dir "$OUT_DIR"
~~~

Submit all four configs with:

~~~bash
bash pretrain/slurm/submit.sh all
~~~

The Slurm scripts contain no account or site-specific partition. Adjust their resource settings for the target cluster. Training writes `last.pt`, `best.pt`, and `metrics.jsonl` directly under `OUT_DIR`; `last.pt` supports automatic resume. The fixed PG-19 probe is evaluated every 1B training tokens by default.

### Evaluate a trained checkpoint

Set the checkpoint once:

~~~bash
RUN_NAME=mamba3_mimo_187m_rope0_fineweb_edu_llama31_100b
CKPT="$PWD/training_outputs/$RUN_NAME/last.pt"
EVAL_NAME="$RUN_NAME-last"
~~~

Run the four-corpus PPL length extrapolation at 1K, 2K, and 4K:

~~~bash
python -m evaluate.ppl \
  --checkpoint "$CKPT" --run-name "$EVAL_NAME" \
  --datasets math_hard codeparrot trivia_qa slimpajama \
  --sequence-lengths 1024,2048,4096 \
  --bucket-size 128 --batch-size 1
~~~

Run the fixed PG-19 extrapolation at 2K, 16K, and 64K:

~~~bash
python -m evaluate.pg19 \
  --checkpoint "$CKPT" --run-name "$EVAL_NAME" --batch-size 1
~~~

Run the standard zero-shot suite:

~~~bash
python -m evaluate.harness \
  --checkpoint "$CKPT" --run-name "$EVAL_NAME" \
  --tasks lambada_openai hellaswag piqa arc_easy \
          arc_challenge winogrande openbookqa \
  --batch-size 8
~~~

Run LongBench-E:

~~~bash
python -m evaluate.longbench \
  --checkpoint "$CKPT" --run-name "$EVAL_NAME" \
  --max-prompt-tokens 65536 --batch-size 1
~~~

PPL, PG-19, and LongBench-E results are written under `evaluate/results/<eval-name>/`. LM Evaluation Harness writes `evaluate/results/harness/<eval-name>/results.json`. Add `--allow-incomplete` when intentionally evaluating a checkpoint saved before its configured token budget.

## 6. Phase interventions

The intervention modules wrap a model forward pass. Dataset loading and metrics stay in the calling evaluation code.

Full-sequence phase interventions:

~~~python
from interventions.phase_interventions.run import intervention

with intervention(model, "zero_angles", sequence_length=input_ids.shape[1]):
    logits = model(input_ids).logits
~~~

Other conditions are `phase_reverse`, `phase_oscillator_shuffle`, `rope_a_mix_1`, and `phase_within_token_oscillator_shuffle`.

Remove B/C phase over the full sequence or a selected predictor interval:

~~~python
from interventions.phase_final_100.run import BCPhase, TailBCPhase, predictor_bounds

with BCPhase("siso", "b_only"):
    logits = model(input_ids).logits

begin, end = predictor_bounds(context=input_ids.shape[1], targets=100)
with TailBCPhase("siso", "both", begin, end):
    logits = model(input_ids).logits
~~~

Remove phase from selected layers:

~~~python
from interventions.phase_zero_accuracy.run import ZeroAngles

with ZeroAngles(model, layers=range(12, 18)):
    logits = model(input_ids).logits
~~~

Observe recurrent-state statistics or remove phase for the hidden-state analysis:

~~~python
from interventions.similarity.run import StateObserver, ZeroAngleProjection

with StateObserver(model, is_mimo=True) as observer:
    output = model.generate(input_ids, max_length=16000)
statistics = observer.summary()

with ZeroAngleProjection(model):
    logits = model(input_ids).logits
~~~
