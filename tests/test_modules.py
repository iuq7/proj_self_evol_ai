import unittest
import os
import shutil

from auto_rl.orchestrator import AutoRLOrchestrator
from dynamic_context.rag_baseline import RAGBaseline
from auto_fine_tuning.drift_detector import DriftDetector
from auto_fine_tuning.lora_finetuner import LoRAFinetuner, DummyDataset
from auto_quantization.gptq_quantizer import GPTQQuantizer

class TestModules(unittest.TestCase):

    def setUp(self):
        # Clean up any previous test artifacts
        if os.path.exists("lora_model_test"): shutil.rmtree("lora_model_test")
        if os.path.exists("quantized_model_test"): shutil.rmtree("quantized_model_test")
        if os.path.exists("onnx_model_test"): shutil.rmtree("onnx_model_test")
        if os.path.exists("reference_data.csv"): os.remove("reference_data.csv")
        if os.path.exists("new_data.csv"): os.remove("new_data.csv")

    def test_auto_rl_orchestrator(self):
        print("\nTesting Auto-RL Orchestrator...")
        search_space = {
            "env": "CartPole-v0",
            "lr": 0.001,
            "gamma": 0.99,
        }
        orchestrator = AutoRLOrchestrator(search_space)
        # We can't run the full tune.run here without a Ray cluster, so we'll just test initialization
        self.assertIsNotNone(orchestrator)
        print("Auto-RL Orchestrator initialized successfully.")

    def test_rag_baseline(self):
        print("\nTesting RAG Baseline...")
        rag = RAGBaseline()
        query = "What is the capital of France?"
        response = rag.retrieve_and_generate(query)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        print(f"RAG Response: {response}")

    def test_drift_detector(self):
        print("\nTesting Drift Detector...")
        import pandas as pd
        reference_data = pd.DataFrame({'feature1': [1, 2, 3, 4, 5], 'feature2': [6, 7, 8, 9, 10]})
        new_data = pd.DataFrame({'feature1': [11, 12, 13, 14, 15], 'feature2': [16, 17, 18, 19, 20]})
        reference_data.to_csv("reference_data.csv", index=False)
        new_data.to_csv("new_data.csv", index=False)

        drift_detector = DriftDetector("reference_data.csv", "new_data.csv")
        is_drift_detected = drift_detector.detect_drift()
        self.assertIsInstance(is_drift_detected, bool)
        print(f"Drift detected: {is_drift_detected}")

    def test_lora_finetuner(self):
        print("\nTesting LoRA Finetuner...")
        finetuner = LoRAFinetuner()
        train_dataset = DummyDataset(finetuner.tokenizer)
        eval_dataset = DummyDataset(finetuner.tokenizer)
        finetuner.fine_tune(train_dataset, eval_dataset, output_dir="lora_model_test")
        self.assertTrue(os.path.exists("lora_model_test"))
        print("LoRA Finetuner completed successfully.")

    def test_gptq_quantizer(self):
        print("\nTesting GPTQ Quantizer...")
        quantizer = GPTQQuantizer()
        quantizer.quantize_model(output_dir="quantized_model_test")
        self.assertTrue(os.path.exists("quantized_model_test"))
        quantizer.export_to_onnx("quantized_model_test", output_path="onnx_model_test")
        self.assertTrue(os.path.exists("onnx_model_test"))
        print("GPTQ Quantizer completed successfully.")

    def tearDown(self):
        # Clean up test artifacts
        if os.path.exists("lora_model_test"): shutil.rmtree("lora_model_test")
        if os.path.exists("quantized_model_test"): shutil.rmtree("quantized_model_test")
        if os.path.exists("onnx_model_test"): shutil.rmtree("onnx_model_test")
        if os.path.exists("reference_data.csv"): os.remove("reference_data.csv")
        if os.path.exists("new_data.csv"): os.remove("new_data.csv")

if __name__ == '__main__':
    unittest.main()
