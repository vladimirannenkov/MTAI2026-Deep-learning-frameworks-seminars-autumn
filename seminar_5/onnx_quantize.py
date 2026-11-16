"""Экспорт Pythia в ONNX и калиброванная Q/DQ-квантизация."""
import argparse
from pathlib import Path

import numpy as np
import onnx
import torch
from datasets import load_dataset
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "EleutherAI/pythia-160m"
REVISION = "50f5173d932e8e61f858120bcb800b97af589f46"


class Model64(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mask", torch.triu(torch.full((1, 1, 64, 64), -10000.0), diagonal=1))

    def forward(self, input_ids):
        return self.model(input_ids=input_ids, attention_mask=self.mask, use_cache=False).logits


class Reader(CalibrationDataReader):
    def __init__(self, blocks):
        self.blocks = iter(blocks)

    def get_next(self):
        block = next(self.blocks, None)
        return None if block is None else {"input_ids": block}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cal-blocks", type=int, default=32)
    parser.add_argument("--nodes", nargs="*", default=None, help="Имена MatMul/Gemm; без аргумента квантуются все")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.float32, attn_implementation="eager").eval()
    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(x for x in data["text"] if x.strip())
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    blocks = [ids[i:i + 64][None].numpy().astype(np.int64) for i in range(0, len(ids) - 63, 64)][:args.cal_blocks]
    if len(blocks) != args.cal_blocks:
        raise ValueError("Недостаточно калибровочных блоков")
    fp32 = out / "pythia_fp32.onnx"
    quantized = out / "pythia_int8_qdq.onnx"
    wrapped = Model64(model).eval()
    sample = torch.from_numpy(blocks[0])
    with torch.inference_mode():
        if not torch.allclose(model(input_ids=sample, use_cache=False).logits, wrapped(sample), rtol=1e-4, atol=1e-4):
            raise RuntimeError("Маска изменила выход модели")
    torch.onnx.export(wrapped, (sample,), str(fp32), input_names=["input_ids"], output_names=["logits"], opset_version=17, dynamo=False)
    onnx.checker.check_model(str(fp32))
    quantize_static(str(fp32), str(quantized), Reader(blocks), quant_format=QuantFormat.QDQ,
                    per_channel=True, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax, op_types_to_quantize=["MatMul", "Gemm"],
                    nodes_to_quantize=args.nodes or [],
                    extra_options={"ActivationSymmetric": True, "WeightSymmetric": True})
    onnx.checker.check_model(str(quantized))
    graph = onnx.load(str(quantized)).graph
    initializers = {x.name: x for x in graph.initializer}
    zeros = [node.input[2] for node in graph.node if node.op_type in ("QuantizeLinear", "DequantizeLinear") and len(node.input) > 2]
    if not zeros or any(z not in initializers or np.any(onnx.numpy_helper.to_array(initializers[z]) != 0) for z in zeros):
        raise RuntimeError("Обнаружен ненулевой или динамический zero-point")
    print("ONNX FP32:", fp32, "\nONNX Q/DQ:", quantized, "\nПроверено zero-point:", len(zeros))


if __name__ == "__main__":
    main()
