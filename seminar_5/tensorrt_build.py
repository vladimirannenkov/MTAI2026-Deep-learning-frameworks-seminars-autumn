"""Сборка TensorRT-движка из подготовленного ONNX-графа."""
import argparse
from pathlib import Path

try:
    import tensorrt as trt
except ImportError:
    import tensorrt_bindings as trt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx")
    parser.add_argument("engine")
    args = parser.parse_args()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(args.onnx):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    engine = builder.build_serialized_network(network, config)
    Path(args.engine).write_bytes(bytes(engine))
    print("Engine:", args.engine, "размер:", Path(args.engine).stat().st_size, "байт")


if __name__ == "__main__":
    main()
