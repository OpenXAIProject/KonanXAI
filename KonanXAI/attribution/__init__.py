from KonanXAI.attribution.deeplift import *
from KonanXAI.attribution.gradcam import *
from KonanXAI.attribution.gradcampp import *
from KonanXAI.attribution.eigencam import *
from KonanXAI.attribution.guided_gradcam import *
from KonanXAI.attribution.layer_wise_propagation import *
from KonanXAI.attribution.integrated_gradient import IG
from KonanXAI.attribution.kernel_shap import *
from KonanXAI.attribution.lime_image import *
from KonanXAI.attribution.gradient import *
from KonanXAI.attribution.gradientxinput import *
from KonanXAI.attribution.smoothgrad import *

def load_attribution(config):
    yolov4_calc = ['gradcam', 'gradcampp', 'eigencam']
    attributions = config['algorithm_name']
    if isinstance(attributions, (tuple, list)):
        attributions = [attribution.lower() for attribution in attributions]
    else:
        attributions = attributions.lower()
    if config['framework'] == 'darknet' and attributions == 'all':
        return object_algorithm(yolov4_calc)
    # elif config['framework'] == 'darknet':
    #     return object_algorithm(attributions)
    else:
        return object_algorithm(attributions)
    
def object_algorithm(algorithm_name):
    algo = []
    if isinstance(algorithm_name, (tuple,list)):
        for name in algorithm_name:
            algo.append(parser_attributions(name))
    else:
        algo.append(parser_attributions(algorithm_name))
    return algo

def parser_attributions(algorithm_name):
    if algorithm_name == 'gradcam':
        algorithm = GradCAM
    elif algorithm_name == 'gradcampp':
        algorithm = GradCAMpp
    elif algorithm_name == "guidedgradcam":
        algorithm = GuidedGradCAM
    elif algorithm_name == 'eigencam':
        algorithm = EigenCAM
    elif algorithm_name == 'lrp':
        algorithm = LRP
    elif algorithm_name == 'lrpyolo':
        algorithm = LRPYolo
    elif algorithm_name == 'ig':
        algorithm = IG
    elif algorithm_name == "lime":
        algorithm = LimeImage
    elif algorithm_name == "kernelshap":
        algorithm = KernelShap
    elif algorithm_name == "gradient":
        algorithm = Gradient
    elif algorithm_name == "gradientxinput":
        algorithm = GradientxInput
    elif algorithm_name == "smoothgrad":
        algorithm = SmoothGrad
    elif algorithm_name == "deeplift":
        algorithm = DeepLIFT
    else:
        print("Not support algorithm")
    algorithm.type = algorithm_name
    return algorithm
