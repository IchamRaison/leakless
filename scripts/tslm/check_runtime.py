"""Vérification réelle du GPU, sans données métier ni chargement de poids."""
import json
import platform

import torch


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponible")
    torch.manual_seed(20260912)
    x = torch.randn(128, 128, device="cuda", requires_grad=True)
    loss = (x @ x.T).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
    assert x.grad.norm() > 0
    print(json.dumps({"python": platform.python_version(), "torch": torch.__version__,
                      "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                      "compute_capability": torch.cuda.get_device_capability(),
                      "loss": loss.item(), "gradient_norm": x.grad.norm().item(),
                      "allocated_bytes": torch.cuda.max_memory_allocated()}, indent=2))


if __name__ == "__main__":
    main()
