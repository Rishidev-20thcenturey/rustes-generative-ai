import torch


def sample_flow_pair(x0: torch.Tensor):
    """Create a linear noise/data interpolation for flow-matching training.

    x0 is a clean latent. t=0 is clean data and t=1 is Gaussian noise.
    Returns x_t, t and the target velocity (noise - data).
    """
    noise = torch.randn_like(x0)
    t = torch.rand(x0.shape[0], device=x0.device)
    shape = (x0.shape[0],) + (1,) * (x0.ndim - 1)
    xt = (1 - t.view(shape)) * x0 + t.view(shape) * noise
    target = noise - x0
    return xt, t, target


def predict_velocity_tensor(model, x, t, text, text_mask=None):
    """Velocity tensor from any DiT, for sampling.

    Models with a ``predict_velocity`` method (RAYDiT_v5) return the tensor directly.
    Otherwise ``forward`` is called, and a tuple result (prediction, aux losses...) is
    reduced to its first element. Sampling never sees auxiliary losses.
    """
    if hasattr(model, "predict_velocity"):
        return model.predict_velocity(x, t, text, text_mask=text_mask)
    out = model(x, t, text, text_mask=text_mask)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def euler_sample(model, text, shape, steps=20, device=None, text_mask=None,
                 cfg_scale=1.0, null_text=None, null_mask=None):
    """Generate a latent by integrating the learned velocity field.

    Issue #7 (classifier-free guidance): when ``cfg_scale > 1.0`` the model is
    evaluated twice per step, once with ``text`` and once with the null prompt
    ``null_text``, and the velocity is ``uncond + cfg_scale * (cond - uncond)``.
    ``cfg_scale <= 1.0`` runs the original single-pass sampler unchanged.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    use_cfg = cfg_scale > 1.0
    if use_cfg and (null_text is None or null_mask is None):
        raise ValueError("cfg_scale > 1.0 requires null_text and null_mask")
    device = device or text.device
    x = torch.randn(shape, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((shape[0],), 1.0 - i * dt, device=device)
        # Training uses data->noise; sampling follows noise->data.
        v = predict_velocity_tensor(model, x, t, text, text_mask)
        if use_cfg:
            v_uncond = predict_velocity_tensor(model, x, t, null_text, null_mask)
            v = v_uncond + cfg_scale * (v - v_uncond)
        x = x - dt * v
    return x
