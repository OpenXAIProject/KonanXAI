from ...utils import *
from ...lib.algorithm import *
from ...models import XAIModel
from .explain import ExplainData

# Kernel Private
def _calculate(algorithm, model, dataset, platform) -> list:
    results = []
    explain: Algorithm = algorithm(model, dataset, platform)
    for data in dataset:
        explain.target_input = data
        results.append((explain.calculate(), data.raw))
    return results

# Kernel API
def request_algorithm(xai) -> ExplainData:
    # Platform
    platform = xai.model.platform
    # Dataset
    dataset = xai.dataset
    # algorithm
    explain = xai.algorithm
    
    # results
    results = {}
    for algorithm in explain:
        explain_class = None
        if algorithm == ExplainType.GradCAM:
            explain_class = GradCAM
        # elif algorithm == ExplainType.GradCAMpp:
        #     explain_class = GradCAMpp
        assert explain_class is not None, "Unsupported XAI algorithm."
        results[explain_class.__name__] = _calculate(explain_class, xai.model, dataset, platform)

    return ExplainData(results)