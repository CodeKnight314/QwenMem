"""
python scripts/export_full_safetensors.py --base_model RichardGTang/Qwen2_5_VL-3B-WithMemory --checkpoint_dir /projects/vig/tangri/saves/qwen2_5vl-3b/sft/memory/m1/nframes_16/checkpoint-236 --output_dir /projects/vig/tangri/QwenMem/saves/m1/ --memory_type base --forward_type m1 --load_extra_bin
python scripts/export_full_safetensors.py --base_model RichardGTang/Qwen2_5_VL-3B-WithMemory --checkpoint_dir /projects/vig/tangri/saves/qwen2_5vl-3b/sft/memory/m2/nframes_16/checkpoint-236 --output_dir /projects/vig/tangri/QwenMem/saves/m2/--memory_type base --forward_type m2 --load_extra_bin
python scripts/export_full_safetensors.py --base_model RichardGTang/Qwen2_5_VL-3B-WithMemory --checkpoint_dir /projects/vig/tangri/saves/qwen2_5vl-3b/sft/memory/m3/nframes_16/checkpoint-236 --output_dir /projects/vig/tangri/QwenMem/saves/m3/ --memory_type base --forward_type m3 --load_extra_bin
"""


from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_repo_to_syspath() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _parse_dtype(s: str):
    s = s.lower().strip()
    if s in {"auto"}:
        return None
    if s in {"fp32", "float32"}:
        import torch
        return torch.float32
    if s in {"fp16", "float16"}:
        import torch
        return torch.float16
    if s in {"bf16", "bfloat16"}:
        import torch
        return torch.bfloat16
    raise ValueError(f"Unknown dtype: {s}. Use one of: auto, fp32, fp16, bf16.")


def main() -> None:
    def stage(msg: str) -> None:
        print(f"[export] {msg}", flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="HF repo id or local path for the base Qwen2.5-VL model weights.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the PEFT adapter checkpoint folder (contains adapter_model.safetensors).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output folder to write the merged full model (will contain model.safetensors).",
    )
    parser.add_argument(
        "--memory_type",
        type=str,
        default="base",
        choices=["none", "base", "mr1", "mr2"],
        help="Which memory module to instantiate in the WithMemory model.",
    )
    parser.add_argument(
        "--forward_type",
        type=str,
        default="base",
        help='Forward variant used during training (e.g. "base", "m1", "m2", "m3").',
    )
    parser.add_argument(
        "--num_state_tokens",
        type=int,
        default=0,
        help='Only used when forward_type == "m3".',
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Model dtype for loading/merging: auto|fp32|fp16|bf16",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="cpu",
        help='Device map for loading the base model. Common values: "cpu", "auto".',
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True when loading base model configs/weights from Hub.",
    )
    parser.add_argument(
        "--load_extra_bin",
        action="store_true",
        help="If checkpoint_dir contains pytorch_model.bin, load it into the base model with strict=False before applying LoRA.",
    )
    args = parser.parse_args()

    _add_repo_to_syspath()

    stage("Importing dependencies (torch/peft/transformers)...")
    try:
        import torch  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency `torch` in your current environment.\n"
            "Install it first, then rerun this script."
        ) from e

    try:
        from peft import PeftModel
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency `peft`. Install it (and `safetensors`) in your env before running.\n"
            "Example: pip install peft safetensors"
        ) from e

    from src.configuration_qwen2_5_vl import Qwen2_5_VLConfig
    from src.modeling_qwen2_5_vl_with_memory import Qwen2_5_VLForConditionalGenerationWithMemory

    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    stage(f"Validating adapter checkpoint dir: {checkpoint_dir}")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"checkpoint_dir does not exist: {checkpoint_dir}")
    if not (checkpoint_dir / "adapter_model.safetensors").exists() and not (checkpoint_dir / "adapter_model.bin").exists():
        raise FileNotFoundError(
            f"checkpoint_dir does not look like a PEFT adapter folder (missing adapter_model.*): {checkpoint_dir}"
        )

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stage(f"Output dir: {out_dir}")

    torch_dtype = _parse_dtype(args.dtype)
    stage(f"Requested dtype: {args.dtype} -> {torch_dtype}")
    stage(f"device_map: {args.device_map}")

    # Load base config and inject our WithMemory-specific fields.
    stage("Loading base config...")
    config = Qwen2_5_VLConfig.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    config.memory_type = args.memory_type
    config.forward_type = args.forward_type
    if args.forward_type == "m3":
        config.num_state_tokens = int(args.num_state_tokens)
    stage(f"Config patched: memory_type={config.memory_type}, forward_type={config.forward_type}")

    # Avoid surprise behavior from HF caches when exporting.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    stage("Loading base model (this can take a while)...")
    model = Qwen2_5_VLForConditionalGenerationWithMemory.from_pretrained(
        args.base_model,
        config=config,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )

    extra_bin = checkpoint_dir / "pytorch_model.bin"
    if args.load_extra_bin and extra_bin.exists():
        stage(f"Loading extra weights from pytorch_model.bin: {extra_bin}")
        import torch

        obj = torch.load(str(extra_bin), map_location="cpu")
        # Common formats:
        # - raw state_dict: {name: tensor, ...}
        # - wrapper dict: {"state_dict": {...}} or {"model": {...}} or {"module": {...}}
        state_dict = None
        if isinstance(obj, dict):
            if "state_dict" in obj and isinstance(obj["state_dict"], dict):
                state_dict = obj["state_dict"]
            elif "model" in obj and isinstance(obj["model"], dict):
                state_dict = obj["model"]
            elif "module" in obj and isinstance(obj["module"], dict):
                state_dict = obj["module"]
            else:
                # Heuristic: treat as state_dict if it looks like one.
                if all(isinstance(k, str) for k in obj.keys()):
                    state_dict = obj
        if not isinstance(state_dict, dict):
            raise RuntimeError(
                f"Unrecognized pytorch_model.bin format at {extra_bin}. "
                "Expected a state_dict-like dict or a dict containing 'state_dict'/'model'/'module'."
            )

        incompatible = model.load_state_dict(state_dict, strict=False)
        # torch returns IncompatibleKeys(missing_keys, unexpected_keys)
        stage(
            f"Loaded extra_bin with strict=False: "
            f"{len(incompatible.missing_keys)} missing keys, {len(incompatible.unexpected_keys)} unexpected keys"
        )

    stage("Loading PEFT adapter...")
    model = PeftModel.from_pretrained(model, str(checkpoint_dir), is_trainable=False)
    stage("Merging adapter into base weights (merge_and_unload)...")
    model = model.merge_and_unload()
    model.eval()

    stage("Saving merged model to safetensors...")
    model.save_pretrained(str(out_dir), safe_serialization=True)

    print(f"[ok] Saved merged full safetensors model to: {out_dir}")


if __name__ == "__main__":
    main()
