from ...utils import *
from ...lib.attribution import *
from ...models import XAIModel
from .explain import ExplainData
import numpy as np
# Kernel Private
def _calculate(algorithm, model, dataset, platform) -> list:
    results = []
    explain: Algorithm = algorithm(model, dataset, platform)
    for data in dataset:
        # Pytorch
        if isinstance(data,tuple):
            import torch
            data = data[0].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).type(model.device)
            explain.target_input = data
            data = np.transpose(data.detach().cpu().squeeze(0),(1,2,0))*255
            results.append((explain.calculate(),data))
        #DarkNet
        else:
            explain.target_input = data# 구조 변경 필요
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
        elif algorithm == ExplainType.GradCAMpp:
            explain_class = GradCAMpp
        elif algorithm == ExplainType.EigenCAM:
            explain_class = EigenCAM
        elif algorithm[0] == ExplainType.LRP:
            explain_class = LRP
            xai.model.rule = algorithm[1].name
            xai.model.yaml_path = algorithm[2]
            #모드 설정 필요

        assert explain_class is not None, "Unsupported XAI algorithm."
        results[explain_class.__name__] = _calculate(explain_class, xai.model, dataset, platform)

    return ExplainData(results, xai.model.mtype.name.lower(), platform.name.lower())