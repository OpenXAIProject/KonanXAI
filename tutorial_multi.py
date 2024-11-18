from KonanXAI import autoXAI, preprocessing
json_path = "D:/xai_refactoring/piper_test_yaml/evaluation/resnet_multi_test_dict.json"
job_type, model_param, datasets_param, arg_param, output = preprocessing(json_path)

work_handle = autoXAI.run_xai(job_type, model_param, datasets_param, arg_param, output)

for progress in work_handle:
    pass