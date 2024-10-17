import torch
from KonanXAI.attribution import load_attribution
from KonanXAI.model_improvement import load_train
from KonanXAI.utils.data_convert import convert_tensor
from KonanXAI.utils.manager import save_image
import os
__all__ = ["load_attribution", "parser_attribution"]
def explain(model, dataset, arg_param, output):
    if isinstance(output, dict):
        save_path = output['volume'] + "/" + output['file_path'] + "/"
    algorithm = load_attribution(arg_param)
    algorithm = algoritmn_init(algorithm, dataset)
    framework = arg_param['framework']
    for index, data in enumerate(dataset):
        origin_img, img_size, output = preprocessing(model, framework, data, algorithm[0].data_type)
        for algo in algorithm:
            img_save_path = image_path(framework, save_path, dataset, index, algo)
            calc_algo = algo(framework = framework, model = model, input = data, config = arg_param).calculate()
            save_image(model_name = model.model_name, algorithm_type = algo.type, origin_img = origin_img, heatmap = calc_algo, img_save_path = img_save_path, img_size = img_size, framework= framework)
        yield round(index / len(dataset), 2)
        
def evaluation(model, dataset, arg_param, output):
    pass

def train(model, dataset, arg_param, output):
    if isinstance(output, dict):
        save_path = output['volume'] + "/" + output['file_path'] + "/"
    if os.path.isdir(save_path) == False:
        os.makedirs(save_path)
    save_step = arg_param['save_step']
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
        origin_img = data[0]
        img_size = data[3]
        if data[1] == -1:
            if data_type == "imagenet":
                infer_data = convert_tensor(data[4], data_type, img_size).unsqueeze(0)
            else:
                infer_data = data[0]
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
    else:
        img_path = dataset.image_name[index].split('\\')
    if len(img_path)>2:
        root = f"{save_path}{algo.type}_result/{img_path[-2]}"
    else:
        root = f"{save_path}{algo.type}_result"
    if os.path.isdir(root) == False:
        os.makedirs(root)
    img_save_path = f"{root}/{img_path[-1]}"
    return img_save_path