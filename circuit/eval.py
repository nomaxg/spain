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
    print("Model loaded and checked.")
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
    if isinstance(model_or_path, str):
        session = ort.InferenceSession(
            model_or_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
    else:
        session = ort.InferenceSession(
            model_or_path.SerializeToString(),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
    input_feed = random_input(session, input_data)
    outputs = session.run(None, input_feed)
    return outputs, input_feed


def bench_evaluate_onnx_model(model_or_path, sess_options=None, input_data=None):
    if isinstance(model_or_path, str):
        session = ort.InferenceSession(
            model_or_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
    else:
        session = ort.InferenceSession(
            model_or_path.SerializeToString(),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
    input_feed = random_input(session, input_data)
    start = time.time()
    _ = session.run(None, input_feed)
    end = time.time()
    # Returns time in milliseconds to run the model, excluding session creation time and input generation time
    return (end - start) * 1000


def model_info(model):
    ops = set()
    for node in model.graph.node:
        ops.add(node.op_type)
    print(ops)


if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) < 2:
        print("Usage: python eval.py <export_path> [batch_size]")
        sys.exit(1)

    export_path = sys.argv[1]
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    original_model_path = f"{export_path}/original_model.onnx"
    model_name = export_path.split("/")[-1]
    num_samples = 100
    
    # Turn off all optimizations
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    total_time = 0
    model = load_onnx_model(original_model_path)

    for _ in range(num_samples):
        total_time += bench_evaluate_onnx_model(model, sess_options=sess_options)

    total_time /= num_samples
    total_time *= batch_size

    print(f"Model: {model_name}")
    print("Batch size:", batch_size)
    print(f"Inference time: {total_time}ms")
