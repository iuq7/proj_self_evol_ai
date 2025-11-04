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

if __name__ == "__main__":
    # Create a dummy model for demonstration
    class DummyModel(nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 5)

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    model = DummyModel()
    analyzer = ModelAnalyzer(model)
    sensitivity_report = analyzer.analyze_layer_sensitivity()

    print("Layer Sensitivity Report:")
    for layer, sensitivity in sensitivity_report.items():
        print(f"- {layer}: {sensitivity:.4f}")
