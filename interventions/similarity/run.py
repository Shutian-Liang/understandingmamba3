import contextlib

import torch


GROUPS = ("all", "rope", "non_rope")
FIELDS = ("propagated", "input", "cross", "cosine", "state")


def _state_coordinates(is_mimo, d_state, device):
    if d_state % 4:
        raise ValueError("d_state must be divisible by four")
    if is_mimo:
        rope = list(range(d_state // 4)) + list(range(d_state // 2, 3 * d_state // 4))
    else:
        rope = list(range(d_state // 2))
    rope_set = set(rope)
    non_rope = [index for index in range(d_state) if index not in rope_set]
    return (
        torch.tensor(rope, device=device),
        torch.tensor(non_rope, device=device),
    )


class ZeroAngleProjection(contextlib.AbstractContextManager):
    """Set the raw angle projection to zero in every Mamba-3 layer."""

    def __init__(self, model):
        self.model = model
        self.handles = []

    def __enter__(self):
        for block in self.model.backbone.layers:
            count = int(block.mixer.num_rope_angles)

            def zero_angles(_module, _inputs, output, *, _count=count):
                output = output.clone()
                output[..., -_count:] = 0
                return output

            self.handles.append(block.mixer.in_proj.register_forward_hook(zero_angles))
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in self.handles:
            handle.remove()
        return False


class PhaseScale(contextlib.AbstractContextManager):
    """Scale phase increments during recurrent decoding without changing decay."""

    def __init__(self, scale):
        import mamba_ssm.modules.mamba3 as implementation

        if not 0 <= scale <= 1:
            raise ValueError("scale must be in [0, 1]")
        self.implementation = implementation
        self.original = implementation.apply_rotary_qk_inference_fwd
        self.scale = float(scale)

    def __enter__(self):
        def forward(*, q, k, angle_state, angle_proj, dt, **kwargs):
            return self.original(
                q=q,
                k=k,
                angle_state=angle_state,
                angle_proj=angle_proj,
                dt=dt * self.scale,
                **kwargs,
            )

        self.implementation.apply_rotary_qk_inference_fwd = forward
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.implementation.apply_rotary_qk_inference_fwd = self.original
        return False


class StateObserver(contextlib.AbstractContextManager):
    """Accumulate propagated-state, input-state, cosine, and RMS statistics."""

    def __init__(self, model, is_mimo):
        import mamba_ssm.modules.mamba3 as implementation

        if implementation.mamba3_step_fn is None:
            raise RuntimeError("the Mamba-3 recurrent step kernel is required")
        self.implementation = implementation
        self.original = implementation.mamba3_step_fn
        self.layers = [block.mixer for block in model.backbone.layers]
        self.num_layers = len(self.layers)
        self.nheads = int(self.layers[0].nheads)
        self.d_state = int(self.layers[0].d_state)
        self.device = next(model.parameters()).device
        self.rope, self.non_rope = _state_coordinates(is_mimo, self.d_state, self.device)
        self.sums = torch.zeros(
            self.num_layers,
            self.nheads,
            len(GROUPS),
            len(FIELDS),
            dtype=torch.float64,
            device=self.device,
        )
        self.counts = torch.zeros(self.num_layers, dtype=torch.int64, device=self.device)
        self.calls = 0

    @staticmethod
    def _select(values, indices):
        return values if indices is None else values.index_select(-1, indices)

    def __enter__(self):
        def step(state, b_state, x_state, A, B, C, D, x, dt, trap, xproj,
                 outproj=None, state_out=None, out=None, z=None, zproj=None,
                 tile_D=64, num_warps=2):
            layer = self.calls % self.num_layers
            self.calls += 1
            previous = state.detach().clone()
            alpha = torch.exp(A.float() * dt.float())
            beta = (1 - trap.float()) * dt.float() * alpha
            gamma = trap.float() * dt.float()
            current_x = x.float()[:, None] * xproj.float()[None]
            previous_x = x_state.float()[:, None] * xproj.float()[None]
            update = torch.einsum(
                "brhp,brhn->bhpn", current_x * gamma[:, None, :, None], B.float()
            ) + torch.einsum(
                "brhp,brhn->bhpn", previous_x * beta[:, None, :, None], b_state.float()
            )
            result = self.original(
                state, b_state, x_state, A, B, C, D, x, dt, trap, xproj,
                outproj=outproj, state_out=state_out, out=out, z=z, zproj=zproj,
                tile_D=tile_D, num_warps=num_warps,
            )
            current = state if state_out is None else state_out
            propagated = previous.float() * alpha[..., None, None]
            groups = []
            for indices in (None, self.rope, self.non_rope):
                p = self._select(propagated, indices)
                u = self._select(update, indices)
                h = self._select(current.float(), indices)
                p2 = p.square().sum(dim=(-1, -2))
                u2 = u.square().sum(dim=(-1, -2))
                dot = (p * u).sum(dim=(-1, -2))
                groups.append(torch.stack((
                    p2,
                    u2,
                    2 * dot,
                    dot / (torch.sqrt(p2 * u2) + 1e-12),
                    h.square().sum(dim=(-1, -2)),
                ), dim=-1))
            values = torch.stack(groups, dim=-2)
            self.sums[layer] += values.sum(dim=0, dtype=torch.float64)
            self.counts[layer] += values.shape[0]
            return result

        self.implementation.mamba3_step_fn = step
        return self

    def summary(self):
        counts = self.counts.clamp_min(1).view(-1, 1, 1, 1)
        means = self.sums / counts
        result = {name: means[..., index].detach().cpu() for index, name in enumerate(FIELDS)}
        result["state_rms"] = result["state"].clamp_min(0).sqrt()
        return result

    def reset(self):
        self.sums.zero_()
        self.counts.zero_()

    def __exit__(self, exc_type, exc, traceback):
        self.implementation.mamba3_step_fn = self.original
        return False


__all__ = ["GROUPS", "FIELDS", "PhaseScale", "StateObserver", "ZeroAngleProjection"]
