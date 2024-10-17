from KonanXAI.attribution.manager import explain, train
from KonanXAI.models import model_import
from KonanXAI.datasets import load_dataset
from KonanXAI.utils import *
def run_xai(job_type, model_param, datasets_param, arg_param, output):
    dataset = load_dataset(**datasets_param)
    model = model_import(**model_param, source='local')
    if job_type == "explain":
        handle = explain(model, dataset, arg_param, output)
        return handle
    elif job_type == "train":
        trainer = train(model, dataset, arg_param, output)
        handle = trainer.xai_train()
        return handle
    elif job_type == "evaluation":
        pass
    else:
        print("err")
