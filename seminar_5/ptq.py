"""Учебная версия: калибровка, sensitivity, INT8 Q/DQ и оценка Pythia."""
import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "EleutherAI/pythia-160m"
REVISION = "50f5173d932e8e61f858120bcb800b97af589f46"


def get_blocks(tokenizer, split, count, context):
    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(x for x in data["text"] if x.strip())
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    blocks = [ids[i:i + context] for i in range(0, len(ids) - context + 1, context)]
    if count > len(blocks):
        raise ValueError(f"Недостаточно блоков в {split}: {len(blocks)}")
    return blocks[:count]


def observe(model, blocks, device):
    layers = {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear) and n not in ("lm_head", "embed_out")}
    ranges = {n: [float("inf"), float("-inf")] for n in layers}
    handles = []
    for name, layer in layers.items():
        def hook(module, inputs, name=name):
            x = inputs[0].detach()
            ranges[name][0] = min(ranges[name][0], x.min().item())
            ranges[name][1] = max(ranges[name][1], x.max().item())
        handles.append(layer.register_forward_pre_hook(hook))
    with torch.inference_mode():
        for block in blocks:
            model(input_ids=block[None].to(device), use_cache=False)
    for handle in handles:
        handle.remove()
    return layers, ranges


def quant_weight(weight):
    scale = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127
    quantized = (weight / scale).round().clamp(-127, 127).to(torch.int8)
    return quantized, scale


def sensitivity(model, layers, ranges, blocks, device):
    scores = {name: [0.0, 0.0] for name in layers}
    handles = []
    for name, layer in layers.items():
        lo, hi = ranges[name]
        activation_scale = max(abs(lo), abs(hi), 1e-12) / 127
        qweight, weight_scale = quant_weight(layer.weight.detach().float())
        weight = qweight.float() * weight_scale
        def hook(module, inputs, output, name=name, scale=activation_scale, weight=weight):
            x = inputs[0]
            x = (x / scale).round().clamp(-127, 127) * scale
            quantized_output = F.linear(x, weight, module.bias)
            scores[name][0] += output.float().square().sum().item()
            scores[name][1] += (output.float() - quantized_output).square().sum().item()
        handles.append(layer.register_forward_hook(hook))
    with torch.inference_mode():
        for block in blocks:
            model(input_ids=block[None].to(device), use_cache=False)
    for handle in handles:
        handle.remove()
    ranked = [(name, 10 * math.log10(max(signal, 1e-30) / max(error, 1e-30)))
              for name, (signal, error) in scores.items()]
    return sorted(ranked, key=lambda item: item[1])


class QuantLinear(nn.Module):
    def __init__(self, layer, limits, mode):
        super().__init__()
        qweight, weight_scale = quant_weight(layer.weight.detach().float())
        self.register_buffer("qweight", qweight)
        self.register_buffer("weight_scale", weight_scale)
        self.register_buffer("bias", None if layer.bias is None else layer.bias.detach().clone())
        lo, hi = limits
        if mode == "symmetric":
            scale, zero, self.qmin, self.qmax = max(abs(lo), abs(hi), 1e-12) / 127, 0, -127, 127
        else:
            lo, hi = min(lo, 0), max(hi, 0)
            scale = max(hi - lo, 1e-12) / 255
            zero, self.qmin, self.qmax = max(0, min(255, round(-lo / scale))), 0, 255
        self.register_buffer("activation_scale", torch.tensor(scale, device=layer.weight.device))
        self.register_buffer("zero", torch.tensor(zero, device=layer.weight.device))

    def forward(self, x):
        q = (x / self.activation_scale + self.zero).round().clamp(self.qmin, self.qmax)
        x = (q - self.zero) * self.activation_scale
        return F.linear(x, self.qweight.float() * self.weight_scale, self.bias)


def evaluate(model, blocks, device):
    nll, correct, tokens = 0.0, 0, 0
    with torch.inference_mode():
        for block in blocks:
            x = block[None].to(device)
            logits = model(input_ids=x, use_cache=False).logits[:, :-1].float()
            target = x[:, 1:]
            nll += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="sum").item()
            correct += (logits.argmax(-1) == target).sum().item()
            tokens += target.numel()
    return {"perplexity": math.exp(nll / tokens), "accuracy": correct / tokens, "tokens": tokens}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fp32", "symmetric", "affine"], default="symmetric")
    parser.add_argument("--skip", choices=["none", "sensitive", "random"], default="sensitive")
    parser.add_argument("--cal-blocks", type=int, default=32)
    parser.add_argument("--eval-blocks", type=int, default=32)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--eval-split", choices=["validation", "test"], default="validation")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.float32, attn_implementation="eager").eval().to(device)
    calibration = get_blocks(tokenizer, "train", args.cal_blocks, args.context)
    evaluation = get_blocks(tokenizer, args.eval_split, args.eval_blocks, args.context)
    ranking_blocks = get_blocks(tokenizer, "validation", 32, args.context)
    layers, ranges = observe(model, calibration, device)
    ranking = sensitivity(model, layers, ranges, ranking_blocks, device) if args.skip != "none" and args.mode != "fp32" else []
    names = [name for name, _ in ranking]
    skipped = set(names[:8] if args.skip == "sensitive" else random.Random(42).sample(names, 8)) if names else set()
    if args.mode != "fp32":
        for name, layer in layers.items():
            if name in skipped:
                continue
            parent, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent), child, QuantLinear(layer, ranges[name], args.mode))
    result = {"metrics": evaluate(model, evaluation, device), "mode": args.mode,
              "skip": args.skip, "skipped_layers": sorted(skipped), "ranking": ranking,
              "calibration_ranges": ranges, "head": "fp32", "context": args.context,
              "calibration_blocks": args.cal_blocks, "evaluation_split": args.eval_split}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
