import argparse
import statistics
import time

import onnxruntime as ort
import numpy as np
import onnx

# Level of precision for inference
INFERENCE_TYPE = onnx.TensorProto.DOUBLE
# Precision data types for inference
PRECISION_OPS = (onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE)


def load_onnx_model(model_path):
    model = onnx.load(model_path)
    # onnx.checker.check_model(model)
    return model


def get_numpy_dtype(ort_type):
    mapping = {
        "tensor(float)": np.float32,
        "tensor(double)": np.float64,
        "tensor(int32)": np.int32,
        "tensor(int64)": np.int64,
        "tensor(bool)": np.bool_,
        "tensor(uint8)": np.uint8,
        "tensor(uint16)": np.uint16,
        "tensor(uint32)": np.uint32,
        "tensor(uint64)": np.uint64,
    }
    return mapping.get(ort_type, np.float32)


def random_input(inference_session, input_data):
    input_feed = {}
    for input_info in inference_session.get_inputs():
        name = input_info.name
        shape = input_info.shape
        shape = [dim if isinstance(dim, int) else 1 for dim in shape]
        dtype = get_numpy_dtype(input_info.type)

        if input_data is not None and name in input_data:
            data = input_data[name]
        else:
            data = np.random.rand(*shape).astype(dtype)

        input_feed[name] = data
    return input_feed


def evaluate_onnx_model(model_or_path, input_data=None, sess_options=None):
    session = create_session(model_or_path, sess_options=sess_options)
    input_feed = random_input(session, input_data)
    outputs = session.run(None, input_feed)
    return outputs, input_feed


def create_session(model_or_path, sess_options=None):
    if isinstance(model_or_path, str):
        return ort.InferenceSession(
            model_or_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
    return ort.InferenceSession(
        model_or_path.SerializeToString(),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )


def bench_evaluate_onnx_model(model_or_path, sess_options=None, input_data=None):
    session = create_session(model_or_path, sess_options=sess_options)
    input_feed = random_input(session, input_data)
    start = time.perf_counter()
    _ = session.run(None, input_feed)
    end = time.perf_counter()
    # Returns time in milliseconds to run the model, excluding session creation time and input generation time
    return (end - start) * 1000


def benchmark_onnx_model(
    model_or_path,
    sess_options=None,
    input_data=None,
    warmup_samples=1,
    measured_samples=100,
    inner_iterations_per_sample=1,
):
    session = create_session(model_or_path, sess_options=sess_options)
    input_feed = random_input(session, input_data)

    for _ in range(warmup_samples):
        for _ in range(inner_iterations_per_sample):
            _ = session.run(None, input_feed)

    sample_times_ms = []
    for _ in range(measured_samples):
        start = time.perf_counter()
        for _ in range(inner_iterations_per_sample):
            _ = session.run(None, input_feed)
        end = time.perf_counter()
        sample_times_ms.append(
            ((end - start) * 1000.0) / inner_iterations_per_sample
        )
    return sample_times_ms


def model_info(model):
    ops = set()
    for node in model.graph.node:
        ops.add(node.op_type)
    print(ops)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("export_path")
    parser.add_argument("batch_size", nargs="?", type=int, default=1)
    parser.add_argument("--warmup-samples", type=int, default=1)
    parser.add_argument("--measured-samples", type=int, default=100)
    parser.add_argument("--inner-iterations", type=int, default=1)
    args = parser.parse_args()

    export_path = args.export_path
    batch_size = args.batch_size
    original_model_path = f"{export_path}/original_model.onnx"
    model_name = export_path.split("/")[-1]

    # Turn off all optimizations
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    model = load_onnx_model(original_model_path)
    sample_times_ms = benchmark_onnx_model(
        model,
        sess_options=sess_options,
        warmup_samples=args.warmup_samples,
        measured_samples=args.measured_samples,
        inner_iterations_per_sample=args.inner_iterations,
    )
    sample_times_ms = [time_ms * batch_size for time_ms in sample_times_ms]
    mean_time_ms = statistics.mean(sample_times_ms)
    stddev_time_ms = statistics.stdev(sample_times_ms) if len(sample_times_ms) > 1 else 0.0
    stddev_ratio = (stddev_time_ms / mean_time_ms) if mean_time_ms else float("inf")
    within_5_percent = stddev_ratio <= 0.05 if mean_time_ms else False

    print(f"Model: {model_name}")
    print("Batch size:", batch_size)
    print("Warmup samples:", args.warmup_samples)
    print("Measured samples:", args.measured_samples)
    print("Inner iterations per sample:", args.inner_iterations)
    print(f"Inference time mean: {mean_time_ms}ms")
    print(f"Inference time stddev: {stddev_time_ms}ms")
    print(f"Stddev / mean: {stddev_ratio}")
    print(f"Within 5% of mean: {within_5_percent}")
    print(f"Inference time: {mean_time_ms}ms")
