"""Phase 6 step 1b: verify every downloaded checkpoint loads into the registry model."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path("scripts/models")))
from registry import ALL_MODELS, PRETRAINED, PRETRAINED_AVAILABLE, build_model, count_params  # noqa: E402

MODELS = ["convnext_tiny", "convnextv2_tiny", "fastvit_sa12", "efficientvit_b1", "repvit_m1_5"]
failures = 0
for name in MODELS:
    ckpt = Path(f"results/{name}/best.pt")
    try:
        blob = torch.load(ckpt, map_location="cpu", weights_only=True)
        if isinstance(blob, dict) and "model" in blob:
            state = blob["model"]
            extra = {k: blob[k] for k in ("epoch", "val_macro_f1") if k in blob}
        else:
            state, extra = blob, {}
        model = build_model(name, num_classes=15, pretrained=False)
        model.load_state_dict(state, strict=True)
        n = count_params(model)
        print(f"{name:<18} loads OK, strict match, {n / 1e6:6.2f} M params  "
              f"epoch={extra.get('epoch', '?')}  val_macroF1={extra.get('val_macro_f1', '?')}")
    except Exception as exc:  # noqa: BLE001
        failures += 1
        print(f"{name:<18} FAILED: {type(exc).__name__}: {exc}")

print(f"\nclassifier checkpoints: {len(MODELS) - failures}/{len(MODELS)} load")
raise SystemExit(1 if failures else 0)