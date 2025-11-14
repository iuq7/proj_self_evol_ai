import torch
import torch.nn as nn


class ModelAnalyzer:
    def __init__(self, model):
        self.model = model

    def analyze_layer_sensitivity(self):
        """
        Analyzes the sensitivity of each layer to quantization.
        This is a placeholder and should be replaced with a more sophisticated method.
        """
        sensitivity_report = {}

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # A simple sensitivity metric: the magnitude of the weights
                sensitivity = torch.mean(torch.abs(module.weight.data))
                sensitivity_report[name] = sensitivity.item()

        return sensitivity_report
