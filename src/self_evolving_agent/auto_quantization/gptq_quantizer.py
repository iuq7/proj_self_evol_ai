from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import torch

class GPTQQuantizer:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        self.model = AutoModelForCausalLM.from_pretrained(self.config["model_name"], torch_dtype=torch.float16)

    def quantize_model(self, bits=4, group_size=128, desc_act=False, output_dir="./quantized_model"):
        quantize_config = BaseQuantizeConfig(
            bits=bits,
            group_size=group_size,
            desc_act=desc_act,
        )

        # Load and quantize the model
        quantized_model = AutoGPTQForCausalLM.from_pretrained(
            self.model,
            quantize_config,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        # Save quantized model
        quantized_model.save_pretrained(output_dir, use_safetensors=True)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Quantized model saved to {output_dir}")

        # Benchmark latency
        self.benchmark_latency(output_dir)

    def benchmark_latency(self, model_path):
        """
        Benchmarks the latency of the quantized model.
        """
        print("Benchmarking latency...")
        model = AutoGPTQForCausalLM.from_quantized(model_path, device="cuda:0", use_safetensors=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Create a dummy input
        dummy_input = "This is a dummy input to test the model's latency."
        inputs = tokenizer(dummy_input, return_tensors="pt").to("cuda:0")

        # Warm-up
        for _ in range(10):
            _ = model.generate(**inputs)

        # Measure latency
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        start_time.record()
        _ = model.generate(**inputs)
        end_time.record()

        torch.cuda.synchronize()

        latency = start_time.elapsed_time(end_time)
        print(f"Latency: {latency:.4f} ms")

    def export_to_onnx(self, model_path, output_path="./onnx_model"):
        """
        Exports the quantized model to ONNX format.
        """
        print("Exporting to ONNX...")
        from optimum.onnxruntime import ORTModelForCausalLM

        model = ORTModelForCausalLM.from_pretrained(model_path, export=True)
        model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print(f"ONNX model saved to {output_path}")

    def benchmark_onnx_latency(self, model_path):
        """
        Benchmarks the latency of the ONNX model.
        """
        print("Benchmarking ONNX latency...")
        from optimum.onnxruntime import ORTModelForCausalLM
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        model = ORTModelForCausalLM.from_pretrained(model_path, session_options=sess_options)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Create a dummy input
        dummy_input = "This is a dummy input to test the model's latency."
        inputs = tokenizer(dummy_input, return_tensors="pt")

        # Warm-up
        for _ in range(10):
            _ = model.generate(**inputs)

        # Measure latency
        import time
        start_time = time.time()
        _ = model.generate(**inputs)
        end_time = time.time()

        latency = (end_time - start_time) * 1000
        print(f"ONNX Latency: {latency:.4f} ms")
