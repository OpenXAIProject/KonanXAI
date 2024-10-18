from KonanXAI.evaluation.base import *
from KonanXAI.evaluation.pixel_flipping import *
from KonanXAI.evaluation.sensitivity import *

PIXEL_FLIPPING_METRICS = [
    MoRF,
    LeRF,
    AbPC,
]

AVAILABLE_METRICS = PIXEL_FLIPPING_METRICS

def load_metric(config, model, baseline):
    evaluation_result = {}
    metrics = config['metric']
    if isinstance(metrics,(list,tuple)):
        metrics = [metric.lower() for metric in metrics]
        for metric in metrics:
            evaluation_result[metric] = []
    else:
        metrics = metrics.lower()
        evaluation_result[metrics] = []
    return object_metric(metrics, model, baseline), evaluation_result

def object_metric(metric, model, baseline):
    metrics = []
    if isinstance(metric,(tuple,list)):
        for name in metric:
            metrics.append(parser_metric(name, model, baseline))
    else:
        metrics.append(parser_metric(metric, model, baseline))
    return metrics
        
def parser_metric(metrics, model, baseline):
    if metrics == 'abpc':
        metric = AbPC(model=model, baseline_fn=baseline)
    elif metrics == 'sensitivity':
        metric = Sensitivity(model=model)
    else:
        print("Not Support Metric")
    metric.type = metrics
    return metric
        