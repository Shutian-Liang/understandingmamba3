"""Add token-local phase switches to the audited whole-sequence B/C kernels."""
from pathlib import Path
import difflib
import hashlib
import json

HERE = Path(__file__).resolve().parent
BASE = HERE.parent


def replace_once(text, old, new):
    assert text.count(old) == 1, (old[:100], text.count(old))
    return text.replace(old, new, 1)


def build():
    manifest = {}
    for name in ("siso", "mimo"):
        source = HERE / f"{name}_fwd.py"
        original = source.read_text()
        text = original
        if name == "siso":
            text = replace_once(text, '"ROTATE_Q", "ROTATE_K"],',
                                '"ROTATE_Q", "ROTATE_K", "PHASE_START", "PHASE_END"],')
            text = replace_once(text, "    ROTATE_K: tl.constexpr,\n", "    ROTATE_K: tl.constexpr,\n    PHASE_START: tl.constexpr,\n    PHASE_END: tl.constexpr,\n")
            text = replace_once(text, "    rotate_k: bool = True,\n", "    rotate_k: bool = True,\n    phase_start: int = 0,\n    phase_end: int = 2147483647,\n")
            text = replace_once(text, "        ROTATE_K=rotate_k,\n", "        ROTATE_K=rotate_k,\n        PHASE_START=phase_start,\n        PHASE_END=phase_end,\n")
            for side, flag in (("k", "K"), ("q", "Q")):
                old = (f"        if ROTATE_{flag}:\n"
                       f"            {side}o0 = {side}0 * cos_block - {side}1 * sin_block\n"
                       f"            {side}o1 = {side}0 * sin_block + {side}1 * cos_block\n"
                       f"        else:\n            {side}o0, {side}o1 = {side}0, {side}1\n")
                new = (f"        {side}o0 = {side}0 * cos_block - {side}1 * sin_block\n"
                       f"        {side}o1 = {side}0 * sin_block + {side}1 * cos_block\n"
                       f"        if not ROTATE_{flag}:\n"
                       "            active_phase = (offs_seqlen >= PHASE_START) & (offs_seqlen < PHASE_END)\n"
                       f"            {side}o0 = tl.where(active_phase[:, None], {side}0, {side}o0)\n"
                       f"            {side}o1 = tl.where(active_phase[:, None], {side}1, {side}o1)\n")
                text = replace_once(text, old, new)
            text = replace_once(text,
                "actual_qk, mask=offs_seqlen < seqlen)",
                "actual_qk, mask=(offs_seqlen < seqlen) & (offs_seqlen >= PHASE_START) & (offs_seqlen < PHASE_END))")
        else:
            text = replace_once(text, "    rotate_k: bool = True,\n", "    rotate_k: bool = True,\n    phase_start: int = 0,\n    phase_end: int = 2147483647,\n")
            text = replace_once(text, "num_stages=0, rotate_q=True, rotate_k=True):",
                                "num_stages=0, rotate_q=True, rotate_k=True, phase_start=0, phase_end=2147483647):")
            text = replace_once(text, "num_stages=num_stages, rotate_q=rotate_q, rotate_k=rotate_k)",
                                "num_stages=num_stages, rotate_q=rotate_q, rotate_k=rotate_k, phase_start=phase_start, phase_end=phase_end)")
            # Cache the native diagonal for all tokens; overwrite only affected
            # same-token blocks after unilateral phase removal.
            text = replace_once(text,
                "                if rotate_q == rotate_k:\n                    T.gemm(q_shared, k_shared, qk_dot_frag, transpose_B=True, clear_accum=True)\n                    T.copy(qk_dot_frag, qk_dot_full_shared)\n",
                "                T.gemm(q_shared, k_shared, qk_dot_frag, transpose_B=True, clear_accum=True)\n                T.copy(qk_dot_frag, qk_dot_full_shared)\n")
            for side in ("q", "k"):
                old = (f"                if rotate_{side}:\n"
                       "                    for cs, r, n in T.Parallel(chunk_size, R, N//rotary_dim_divisor):\n"
                       f"                        {side}_shared[cs*R + r, n] = T.cos(angles_frag[cs, n]) * {side}_first_half_frag[cs, r, n] - T.sin(angles_frag[cs, n]) * {side}_second_half_frag[cs, r, n]\n"
                       f"                        {side}_shared[cs*R + r, N//2 + n] = T.sin(angles_frag[cs, n]) * {side}_first_half_frag[cs, r, n] + T.cos(angles_frag[cs, n]) * {side}_second_half_frag[cs, r, n]\n")
                new = ("                for cs, r, n in T.Parallel(chunk_size, R, N//rotary_dim_divisor):\n"
                       f"                    if rotate_{side} or chunk_start + cs < phase_start or chunk_start + cs >= phase_end:\n"
                       f"                        {side}_shared[cs*R + r, n] = T.cos(angles_frag[cs, n]) * {side}_first_half_frag[cs, r, n] - T.sin(angles_frag[cs, n]) * {side}_second_half_frag[cs, r, n]\n"
                       f"                        {side}_shared[cs*R + r, N//2 + n] = T.sin(angles_frag[cs, n]) * {side}_first_half_frag[cs, r, n] + T.cos(angles_frag[cs, n]) * {side}_second_half_frag[cs, r, n]\n")
                text = replace_once(text, old, new)
            text = replace_once(text,
                "                if rotate_q != rotate_k:\n                    T.gemm(q_shared, k_shared, qk_dot_frag, transpose_B=True, clear_accum=True)\n                    T.copy(qk_dot_frag, qk_dot_full_shared)\n",
                """                if rotate_q != rotate_k:
                    if chunk_start + chunk_size > phase_start and chunk_start < phase_end:
                        T.gemm(q_shared, k_shared, qk_dot_frag, transpose_B=True, clear_accum=True)
                        for a, b in T.Parallel(fused_chunk_size, fused_chunk_size):
                            if chunk_start + a // R >= phase_start and chunk_start + a // R < phase_end:
                                qk_dot_full_shared[a, b] = qk_dot_frag[a, b]
""")
        output = HERE / f"tail_{name}_fwd.py"
        output.write_text(text)
        (HERE / f"{name}_fwd.diff").write_text("".join(difflib.unified_diff(
            original.splitlines(True), text.splitlines(True), fromfile=str(source.relative_to(HERE)), tofile=output.name)))
        manifest[name] = {"source": str(source.relative_to(HERE)),
                          "source_sha256": hashlib.sha256(original.encode()).hexdigest(),
                          "generated": output.name, "generated_sha256": hashlib.sha256(text.encode()).hexdigest()}
    (HERE / "kernel_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    build()
