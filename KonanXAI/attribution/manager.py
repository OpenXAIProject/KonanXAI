import torch
from KonanXAI.attribution import load_attribution
from KonanXAI.evaluation import load_metric
from KonanXAI.model_improvement import load_train
from KonanXAI.utils.data_convert import convert_tensor
from KonanXAI.utils.evaluation import ZeroBaselineFunction, heatmap_postprocessing
from KonanXAI.utils.manager import save_image, set_seed
import torchvision.transforms as transforms
import json
import matplotlib.pyplot as plt
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
__all__ = ["load_attribution", "parser_attribution"]
def explain(model, dataset, arg_param, output):
    framework, algorithm, save_path = explain_init(dataset, arg_param, output)
    for index, data in enumerate(dataset):
        origin_img, img_size, output = preprocessing(model, framework, data, algorithm[0].data_type)
        for algo in algorithm:
            img_save_path = image_path(framework, save_path, dataset, index, algo)
            calc_algo = algo(framework = framework, model = model, input = data, config = arg_param).calculate(targets=output)
            save_image(model_name = model.model_name, algorithm_type = algo.type, origin_img = origin_img, heatmap = calc_algo, img_save_path = img_save_path, img_size = img_size, framework= framework)
        yield round(index / len(dataset), 2)
        
def xai_eval(model, dataset, arg_param, output):
    framework, algorithms, save_path = explain_init(dataset, arg_param, output)
    metrics, evaluation_result = load_metric(arg_param, model, ZeroBaselineFunction())
    for index, data in enumerate(dataset):
        origin_img, img_size, output = preprocessing(model, framework, data, algorithms[0].data_type)
        for metric in metrics:
            metric_li = {}
            for algo in algorithms:
                img_save_path = image_path(framework, save_path, dataset, index, algo)
                algorithm = algo(framework = framework, model = model, input = data, config = arg_param)
                calc_algo = algorithm.calculate(targets = output) 
                post_heatmap = heatmap_postprocessing(algo.type, img_size, calc_algo)
                if metric.type == 'abpc':
                    score = round(metric.evaluate(inputs=(data[0].to('cuda')),targets=output, attributions=post_heatmap).item(),4)
                    #json 저장
                    record_evaluation(evaluation_result,dataset.image_name[index], metric.type, metric_li, algorithm.type, score)
                    save_image(model_name = model.model_name, algorithm_type = algo.type, origin_img = origin_img, heatmap = calc_algo, img_save_path = img_save_path, img_size = img_size, framework= framework, metric=metric.type, score=score)
                if metric.type == 'sensitivity':
                    score = round(metric.evaluate(inputs= origin_img.to(device),targets=output, attributions=post_heatmap, explainer = algorithm).item(),4)
                    #json 저장
                    record_evaluation(evaluation_result,dataset.image_name[index], metric.type, metric_li, algorithm.type, score)
                    save_image(model_name = model.model_name, algorithm_type = algo.type, origin_img = origin_img, heatmap = calc_algo, img_save_path = img_save_path, img_size = img_size, framework= framework, metric=metric.type, score=score)
            evaluation_result[metric.type].append(metric_li)
        yield round(index / len(dataset), 2)
    with open(f"{save_path}/resutl.json", "w") as f:
        json.dump(postprocessing_eval(evaluation_result, metrics),f, indent=4)

def train(model, dataset, arg_param, output):
    save_step, save_path = train_init(arg_param, output)
    model, optimizer, criterion, trainer, gpus, hyperparam = load_train(arg_param)
    model = model(num_classes=dataset.classes)
    optimizer = optimizer(model.parameters(), lr = hyperparam['lr'])
    criterion = criterion()
    trainer = trainer(model, optimizer, criterion, dataset,hyperparam['lr'],hyperparam['batch'],hyperparam['epoch'],save_path=save_path)
    trainer.set_device(gpus)
    trainer.set_checkpoint_step(save_step)
    if trainer.transfer_weights != None:
        trainer.model_load(trainer.transfer_weights)
    if trainer.model_algorithm == "domaingeneralization":
        trainer.set_freq(trainer.frequency)
        target = model
        for m in trainer.target_layer:
            if m.isdigit():
                target = target[int(m)]
            else:
                target = getattr(target,m)
        trainer.set_target_layer(target)
    return trainer
    

def algoritmn_init(algorithm, dataset):
    for algo in algorithm:
        algo.data_type = dataset.dataset_name
    return algorithm

def preprocessing(model, framework, data, data_type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if framework == 'darknet':
        origin_img = data.origin_img
        img_size = data.im_size
        output = None
    else:
        infer_data = data[0]
        img_size = data[3]
        compose_resize = transforms.Compose([
                                transforms.Resize(img_size),
                                transforms.ToTensor()
                            ])
        origin_img = compose_resize(data[4]).unsqueeze(0)
        if data[1] == -1:
            if "yolo" in model.model_name:
                output = None
            else:
                output = model(infer_data.to(device)).argmax(-1).item()
        else: 
            output = data[1]
    return origin_img, img_size, output

def image_path(framework, save_path, dataset, index, algo):
    if framework == "dtrain":
        img_path = ['data/',dataset.image_name[index]]
    if os.name() == "nt":
            img_path = dataset.image_name[index].split('\\')
    else:
        img_path = dataset.image_name[index].split('/')
    if len(img_path)>2:
        root = f"{save_path}{algo.type}_result/{img_path[-2]}"
    else:
        root = f"{save_path}{algo.type}_result"
    if os.path.isdir(root) == False:
        os.makedirs(root)
    img_save_path = f"{root}/{img_path[-1]}"
    return img_save_path

def train_init(arg_param, output):
    if arg_param.get('seed') != None:
        set_seed(arg_param['seed'])
    else:
        set_seed(777)
    if isinstance(output, dict):
        save_path = output['volume'] + "/" + output['file_path'] + "/"
    if os.path.isdir(save_path) == False:
        os.makedirs(save_path)
    save_step = arg_param['save_step']
    return save_step ,save_path

def explain_init(dataset, arg_param, output):
    if isinstance(output, dict):
        save_path = output['volume'] + "/" + output['file_path'] + "/"
    algorithm = load_attribution(arg_param)
    algorithm = algoritmn_init(algorithm, dataset)
    framework = arg_param['framework']
    if arg_param.get('seed') != None:
        set_seed(arg_param['seed'])
    else:
        set_seed(777)
    return framework, algorithm, save_path
    
def record_evaluation(dict_origin, image_path, metric_type, metric_li, attr_tpye, score):
    metric_li["image_path"] = image_path
    metric_li[attr_tpye] = score
    return metric_li

def postprocessing_eval(result_json, metrics):
    for metric in metrics:
        metric = metric.type
        score_li = []
        for value_dict in result_json[metric]:
            score_li.append(list(value_dict.values())[1:])
            keys = list(value_dict.keys())[1:]
        score_average = [list(map(lambda x: sum(x) / len(score_li), zip(*score_li)))]
        metric_dict = {}
        metric_name = f"result_{metric}"
        result_json[metric_name] = []
        for key, value in zip(keys, score_average[0]):
            metric_dict[key] = round(value,4)
        metric_dict = dict(sorted(metric_dict.items(), key=lambda item: item[1], reverse=True))
        result_json[metric_name].append(metric_dict)
    return result_json