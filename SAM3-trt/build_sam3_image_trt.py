import argparse
import os

import tensorrt as trt


def _parse_dims(value):
    parts = str(value).replace(",", "x").split("x")
    dims = tuple(int(part) for part in parts if part != "")
    if not dims:
        raise ValueError(f"Invalid shape: {value}")
    return dims


def parse_profile_spec(value):
    try:
        name, shapes = str(value).split(":", 1)
    except ValueError as exc:
        raise ValueError(
            "--profile must be name:min,opt,max or name:min:opt:max"
        ) from exc

    shape_parts = shapes.split(":")
    if len(shape_parts) == 1:
        min_shape = opt_shape = max_shape = _parse_dims(shape_parts[0])
    elif len(shape_parts) == 3:
        min_shape, opt_shape, max_shape = (_parse_dims(part) for part in shape_parts)
    else:
        raise ValueError(f"Invalid --profile spec: {value}")
    return name, min_shape, opt_shape, max_shape


def _network_input_shapes(network):
    return {
        network.get_input(i).name: tuple(int(dim) for dim in network.get_input(i).shape)
        for i in range(network.num_inputs)
    }


def build_engine(
    onnx_path,
    output_path,
    precision="fp16",
    workspace_gb=8,
    tf32=True,
    profile_specs=None,
):
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file does not exist: {onnx_path}")
    onnx_size = os.path.getsize(onnx_path)
    if onnx_size <= 0:
        raise RuntimeError(f"ONNX file is empty: {onnx_path}")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    explicit_batch = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
    network_flags = 0 if explicit_batch is None else 1 << int(explicit_batch)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    onnx_dir = os.path.dirname(os.path.abspath(onnx_path))
    onnx_name = os.path.basename(onnx_path)
    old_cwd = os.getcwd()
    try:
        # TensorRT resolves ONNX external-data weights relative to the current
        # directory, so parse from the ONNX folder.
        os.chdir(onnx_dir)
        with open(onnx_name, "rb") as f:
            if not parser.parse(f.read()):
                errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
                raise RuntimeError(f"Failed to parse ONNX:\n{errors}")
    finally:
        os.chdir(old_cwd)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))

    input_shapes = _network_input_shapes(network)
    dynamic_inputs = {
        name for name, shape in input_shapes.items()
        if any(dim < 0 for dim in shape)
    }
    profile_specs = profile_specs or []
    if dynamic_inputs and not profile_specs:
        formatted = ", ".join(f"{name}{shape}" for name, shape in input_shapes.items())
        raise ValueError(
            "Dynamic ONNX inputs require at least one --profile. "
            f"Network inputs: {formatted}"
        )
    if profile_specs:
        profile = builder.create_optimization_profile()
        provided = set()
        profile_used = False
        for spec in profile_specs:
            name, min_shape, opt_shape, max_shape = parse_profile_spec(spec)
            if name not in input_shapes:
                raise ValueError(
                    f"--profile input does not exist in ONNX graph: {name}. "
                    f"Available inputs: {sorted(input_shapes)}"
                )
            if name not in dynamic_inputs:
                static_shape = input_shapes[name]
                if any(shape != static_shape for shape in (min_shape, opt_shape, max_shape)):
                    raise ValueError(
                        f"--profile for static input {name} must match {static_shape}, "
                        f"got min={min_shape} opt={opt_shape} max={max_shape}"
                    )
                print(f"Skipping optimization profile for static input {name}{static_shape}")
                continue
            profile.set_shape(name, min_shape, opt_shape, max_shape)
            provided.add(name)
            profile_used = True
        missing = sorted(dynamic_inputs - provided)
        if missing:
            raise ValueError(
                "Missing --profile for dynamic inputs: "
                + ", ".join(missing)
            )
        if profile_used:
            config.add_optimization_profile(profile)

    precision = str(precision).lower()
    if not tf32:
        config.clear_flag(trt.BuilderFlag.TF32)
    has_fp16_flag = hasattr(trt.BuilderFlag, "FP16")
    has_bf16_flag = hasattr(trt.BuilderFlag, "BF16")
    if precision == "fp16" and has_fp16_flag and getattr(builder, "platform_has_fast_fp16", True):
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "bf16" and has_bf16_flag:
        config.set_flag(trt.BuilderFlag.BF16)
    elif precision != "fp32":
        print(
            f"TensorRT {trt.__version__} does not expose a {precision.upper()} builder flag; "
            "using the ONNX tensor data types as exported."
        )

    print(f"Building TensorRT engine from {onnx_path} ({onnx_size} bytes)")
    print(f"Network inputs: {input_shapes}")
    print(f"Network outputs: {[network.get_output(i).name for i in range(network.num_outputs)]}")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed.")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(serialized)
    print(f"Saved TensorRT engine: {output_path} ({os.path.getsize(output_path)} bytes)")


def main():
    parser = argparse.ArgumentParser(description="Build SAM3 image TensorRT engine from static ONNX.")
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workspace-gb", type=float, default=8.0)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument("--fp32", action="store_true", help="Shortcut for --precision fp32.")
    parser.add_argument("--no-tf32", action="store_true", help="Clear TensorRT TF32 builder flag.")
    parser.add_argument(
        "--profile",
        action="append",
        default=None,
        help=(
            "Optimization profile for dynamic ONNX inputs. "
            "Use name:shape for fixed min/opt/max, or name:min:opt:max. "
            "Shapes accept 1x3x1008x1008 or comma-separated dimensions."
        ),
    )
    args = parser.parse_args()
    precision = "fp32" if args.fp32 else args.precision
    build_engine(
        args.onnx,
        args.output,
        precision=precision,
        workspace_gb=args.workspace_gb,
        tf32=not args.no_tf32,
        profile_specs=args.profile,
    )


if __name__ == "__main__":
    main()
