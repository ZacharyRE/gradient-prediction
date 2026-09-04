from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


_COUNT_SKETCH_CACHE: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}


SYSTEM_PROMPT = "You are a helpful assistant and an expert mathematician."
USER_TEMPLATE = """Solve the following mathematics problem. Show your reasoning clearly and place only the final answer inside \\boxed{{}}.

Problem:
{problem}"""


@dataclass
class ExtractionOutput:
    hidden: np.ndarray | None
    gradient: np.ndarray
    raw_gradient: np.ndarray
    metadata: dict


def _chat_template_ids(tokenizer, messages: list[dict]) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    elif isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return list(encoded)


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def load_model_and_tokenizer(config: dict, device: str):
    model_path = config["model"]["path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    dtype = torch.bfloat16 if config["model"]["dtype"] == "bfloat16" else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        local_files_only=True,
    )
    lora = config["lora"]
    if lora["checkpoint"] == "init":
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora["rank"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            bias=lora["bias"],
            target_modules=list(lora["target_modules"]),
            layers_to_transform=list(lora["target_layers"]),
            layers_pattern="layers",
            init_lora_weights=True,
        )
        model = get_peft_model(base_model, lora_config)
    else:
        model = PeftModel.from_pretrained(
            base_model,
            lora["checkpoint"],
            is_trainable=True,
        )
    # Keep trainable LoRA master weights and their accumulated gradients in FP32,
    # independently of the frozen base-model dtype.
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and ("lora_A" in name or "lora_B" in name):
            parameter.data = parameter.data.float()
    model.config.use_cache = False
    model.eval()
    model.to(device)
    return model, tokenizer


def trainable_lora_parameters(model) -> list[tuple[str, torch.nn.Parameter]]:
    parameters = sorted(
        [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad],
        key=lambda item: item[0],
    )
    if not parameters:
        raise RuntimeError("No trainable LoRA parameters found")
    unexpected = [name for name, _ in parameters if "lora_A" not in name and "lora_B" not in name]
    if unexpected:
        raise RuntimeError(f"Unexpected trainable parameters: {unexpected}")
    return parameters


def build_supervised_example(tokenizer, problem: str, solution: str, max_length: int):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(problem=problem)},
    ]
    prompt_ids = _chat_template_ids(tokenizer, messages)
    solution_ids = tokenizer(solution, add_special_tokens=False)["input_ids"]
    eos_id = tokenizer.eos_token_id
    response_ids = solution_ids + ([eos_id] if eos_id is not None else [])
    available = max_length - len(prompt_ids)
    if available <= 0:
        raise ValueError(f"Prompt length {len(prompt_ids)} exceeds max_length={max_length}")
    truncated = len(response_ids) > available
    response_ids = response_ids[:available]
    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "prompt_token_count": len(prompt_ids),
        "response_token_count": len(response_ids),
        "truncated": truncated,
    }


def _pool_prompt_hidden(sequence_hidden: torch.Tensor, config: dict) -> torch.Tensor:
    """Pool an unpadded prompt sequence into one predictor feature vector."""
    pooling = config["model"].get("hidden_token", "last_prompt_token")
    if pooling == "last_prompt_token":
        return sequence_hidden[-1]
    if pooling == "mean_prompt_tokens":
        return sequence_hidden.mean(dim=0)
    raise ValueError(f"Unknown hidden_token pooling: {pooling}")


def extract_one(
    model, tokenizer, row: dict, config: dict, device: str, include_hidden: bool = True
) -> ExtractionOutput:
    encoded = build_supervised_example(
        tokenizer,
        row["problem"],
        row["solution"],
        int(config["model"]["max_sequence_length"]),
    )
    tensor_inputs = {
        key: value.to(device)
        for key, value in encoded.items()
        if isinstance(value, torch.Tensor)
    }
    parameters = trainable_lora_parameters(model)
    model.zero_grad(set_to_none=True)
    tuple_index = int(config["model"]["hidden_states_tuple_index"])
    use_bf16_autocast = bool(config["model"].get("forward_autocast_bfloat16", False))
    hidden = None
    if include_hidden:
        hidden_source = config["model"].get("hidden_source", "hidden_states_tuple")
        captured_inputs = []
        hook = None
        if hidden_source == "target_module_input":
            suffix = config["model"]["hidden_module_suffix"]
            matches = [(name, module) for name, module in model.named_modules()
                       if name.endswith(suffix)]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected one module ending in {suffix!r}, found {[name for name, _ in matches]}"
                )

            def capture_module_input(_module, inputs):
                captured_inputs.append(
                    _pool_prompt_hidden(inputs[0][0], config).detach().float().cpu()
                )

            hook = matches[0][1].register_forward_pre_hook(capture_module_input)
        try:
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16,
                enabled=use_bf16_autocast and str(device).startswith("cuda"),
            ):
                prompt_outputs = model(
                    input_ids=tensor_inputs["input_ids"][:, : encoded["prompt_token_count"]],
                    attention_mask=tensor_inputs["attention_mask"][:, : encoded["prompt_token_count"]],
                    output_hidden_states=hidden_source == "hidden_states_tuple",
                    return_dict=True,
                )
        finally:
            if hook is not None:
                hook.remove()
        if hidden_source == "target_module_input":
            if len(captured_inputs) != 1:
                raise RuntimeError(f"Expected one captured module input, found {len(captured_inputs)}")
            hidden = captured_inputs[0].numpy()
        elif hidden_source == "hidden_states_tuple":
            hidden = (
                _pool_prompt_hidden(prompt_outputs.hidden_states[tuple_index][0], config)
                .detach()
                .float()
                .cpu()
                .numpy()
            )
        else:
            raise ValueError(f"Unknown hidden_source: {hidden_source}")
    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16,
        enabled=use_bf16_autocast and str(device).startswith("cuda"),
    ):
        outputs = model(**tensor_inputs, output_hidden_states=False, return_dict=True)
    loss = outputs.loss
    backward_scale = float(config.get("loss", {}).get("backward_scale", 1.0))
    (loss * backward_scale).backward()
    if backward_scale != 1.0:
        for _, parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.div_(backward_scale)

    gradient_parts = []
    a_parts = []
    b_parts = []
    parameter_layout = []
    offset = 0
    for name, parameter in parameters:
        if parameter.grad is None:
            raise RuntimeError(f"Missing gradient for {name}")
        flat = parameter.grad.detach().float().reshape(-1).cpu()
        gradient_parts.append(flat)
        factor = "A" if "lora_A" in name else "B"
        (a_parts if factor == "A" else b_parts).append(flat)
        parameter_layout.append(
            {"name": name, "factor": factor, "shape": list(parameter.shape), "start": offset, "end": offset + flat.numel()}
        )
        offset += flat.numel()
    raw_gradient = torch.cat(gradient_parts).numpy()
    sketch_config = config.get("gradient_sketch", {})
    if sketch_config.get("enabled", False):
        sketch_dim = int(sketch_config["dimension"])
        sketch_seed = int(sketch_config["seed"])
        cache_key = (len(raw_gradient), sketch_dim, sketch_seed)
        if cache_key not in _COUNT_SKETCH_CACHE:
            indices = np.arange(len(raw_gradient), dtype=np.uint64)
            mixed = indices * np.uint64(0x9E3779B185EBCA87) + np.uint64(sketch_seed)
            buckets = (mixed % np.uint64(sketch_dim)).astype(np.int32)
            signs = np.where((mixed >> np.uint64(63)) == 0, 1.0, -1.0).astype(np.float32)
            _COUNT_SKETCH_CACHE[cache_key] = buckets, signs
        buckets, signs = _COUNT_SKETCH_CACHE[cache_key]
        gradient = np.bincount(
            buckets, weights=raw_gradient * signs, minlength=sketch_dim
        ).astype(np.float32)
    else:
        gradient = raw_gradient
    grad_a = torch.cat(a_parts)
    grad_b = torch.cat(b_parts)
    metadata = {
        "sample_id": row["sample_id"],
        "loss": float(loss.detach().float().cpu()),
        "backward_scale": backward_scale,
        "prompt_token_count": encoded["prompt_token_count"],
        "response_token_count": encoded["response_token_count"],
        "truncated": bool(encoded["truncated"]),
        "gradient_norm": float(np.linalg.norm(raw_gradient)),
        "stored_gradient_norm": float(np.linalg.norm(gradient)),
        "gradient_a_norm": float(torch.linalg.vector_norm(grad_a)),
        "gradient_b_norm": float(torch.linalg.vector_norm(grad_b)),
        "hidden_norm": None if hidden is None else float(np.linalg.norm(hidden)),
        "hidden_source": config["model"].get("hidden_source", "hidden_states_tuple"),
        "hidden_token": config["model"].get("hidden_token", "last_prompt_token"),
        "raw_gradient_dim": int(raw_gradient.shape[0]),
        "stored_gradient_dim": int(gradient.shape[0]),
        "gradient_representation": (
            "signed_count_sketch" if sketch_config.get("enabled", False) else "raw"
        ),
        "parameter_layout": parameter_layout,
    }
    model.zero_grad(set_to_none=True)
    return ExtractionOutput(
        hidden=hidden,
        gradient=gradient,
        raw_gradient=raw_gradient,
        metadata=metadata,
    )


def extract_prompt_only_hidden(model, tokenizer, row: dict, config: dict, device: str) -> np.ndarray:
    sequence = extract_prompt_hidden_sequence(model, tokenizer, row, config, device)
    return _pool_prompt_hidden(torch.from_numpy(sequence), config).numpy()


def extract_prompt_hidden_sequence(
    model, tokenizer, row: dict, config: dict, device: str
) -> np.ndarray:
    """Return every prompt-token vector from the configured hidden source.

    Unlike ``extract_prompt_only_hidden``, this function never pools the token
    dimension.  In particular, ``target_module_input`` is captured with the
    same forward-pre-hook convention used by ``extract_one`` so last, mean,
    and sequence-level predictors can be derived from one identical tensor.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(problem=row["problem"])},
    ]
    prompt_ids = _chat_template_ids(tokenizer, messages)
    inputs = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(inputs)
    hidden_source = config["model"].get("hidden_source", "hidden_states_tuple")
    captured_inputs = []
    hook = None
    if hidden_source == "target_module_input":
        suffix = config["model"]["hidden_module_suffix"]
        matches = [
            (name, module) for name, module in model.named_modules() if name.endswith(suffix)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one module ending in {suffix!r}, "
                f"found {[name for name, _ in matches]}"
            )

        def capture_module_input(_module, module_inputs):
            captured_inputs.append(module_inputs[0][0].detach().float().cpu())

        hook = matches[0][1].register_forward_pre_hook(capture_module_input)
    use_bf16_autocast = bool(config["model"].get("forward_autocast_bfloat16", False))
    try:
        with torch.no_grad(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=use_bf16_autocast and str(device).startswith("cuda"),
        ):
            outputs = model(
                input_ids=inputs,
                attention_mask=attention_mask,
                output_hidden_states=hidden_source == "hidden_states_tuple",
                return_dict=True,
            )
    finally:
        if hook is not None:
            hook.remove()
    if hidden_source == "target_module_input":
        if len(captured_inputs) != 1:
            raise RuntimeError(f"Expected one captured module input, found {len(captured_inputs)}")
        sequence = captured_inputs[0]
    elif hidden_source == "hidden_states_tuple":
        tuple_index = int(config["model"]["hidden_states_tuple_index"])
        sequence = outputs.hidden_states[tuple_index][0].detach().float().cpu()
    else:
        raise ValueError(f"Unknown hidden_source: {hidden_source}")
    if sequence.ndim != 2 or sequence.shape[0] != len(prompt_ids):
        raise RuntimeError(
            f"Unexpected prompt hidden shape {tuple(sequence.shape)} for {len(prompt_ids)} tokens"
        )
    return sequence.numpy()


def extract_teacher_forced_prompt_hidden(
    model, tokenizer, row: dict, config: dict, device: str
) -> np.ndarray:
    encoded = build_supervised_example(
        tokenizer,
        row["problem"],
        row["solution"],
        int(config["model"]["max_sequence_length"]),
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
    tuple_index = int(config["model"]["hidden_states_tuple_index"])
    prompt_hidden = outputs.hidden_states[tuple_index][0, : encoded["prompt_token_count"]]
    return _pool_prompt_hidden(prompt_hidden, config).detach().float().cpu().numpy()


def extract_rows(model, tokenizer, rows: list[dict], config: dict, device: str, progress=None):
    hidden, gradients, metadata = [], [], []
    iterable = progress(rows) if progress else rows
    for row in iterable:
        output = extract_one(model, tokenizer, row, config, device)
        hidden.append(output.hidden)
        gradients.append(output.gradient)
        metadata.append(output.metadata)
    return np.stack(hidden), np.stack(gradients), metadata
