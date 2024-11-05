import numpy as np
np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))
import KonanXAI._core
import KonanXAI.attribution
import KonanXAI.datasets
import KonanXAI.model_improvement
import KonanXAI.models
import KonanXAI.utils
import KonanXAI.evaluation
import os
import json

def preprocessing(json_file):
    with open(json_file, 'r') as f:
        response = json.load(f)
    response = response['job_infos'][0]
    xai_type = response['job_type']
    _input, _output = response['inputs'], response['outputs']
    save_dir = _output[0]['volume'] + "/" + _output[0]['file_path']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    model_param = _input[0]['model_info']
    datasets_param = _input[0]['data_info']
    datasets_param['framework'] = model_param['framework']
    datasets_param['mode'] = xai_type
    arg_param = _input[0]['type_config']
    arg_param['framework'] = model_param['framework']
    arg_param['model_name'] = model_param['model_name']
    return xai_type, model_param, datasets_param, arg_param, _output[0]