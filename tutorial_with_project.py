import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from project.make_project import Project
# config_path = './project/example_gradcam/config_darknet.yaml'
# config_path = './project/example_gradcampp/config_darknet_gradcampp.yaml'
# config_path = './project/example_eigencam/config_darknet_eigencam.yaml'
# config_path = './project/example_lrp/config_efficientnet_lrp.yaml'
# config_path = './project/example_gradcam/config_efficientnet_gradcam.yaml'
# config_path = './project/example_lrp/config_resnet_lrp_epsilon.yaml'
# config_path = './project/example_lrp/config_resnet_lrp_alphabeta.yaml'
# config_path = './project/example_lime/config_resnet_lime.yaml'
# config_path = './project/example_kernelshap/config_resnet_kernelshap.yaml'
# config_path = './project/example_gradcam/config_resnet_gradcam.yaml'
# config_path = './project/example_gradcampp/config_resnet_gradcampp.yaml'
# config_path = './project/example_guided_gradcam/config_resnet_guidedgradcam.yaml'
# config_path = './project/example_eigencam/config_resnet_eigencam.yaml'
# config_path = './project/example_ig/config_resnet_ig.yaml'
# config_path = './project/example_lrp/config_vgg_lrp.yaml'
# config_path = './project/example_gradcam/config_vgg_gradcam.yaml'
# config_path = './project/example_gradcampp/config_vgg_gradcampp.yaml'
# config_path = './project/example_guided_gradcam/config_vgg_guidedgradcam.yaml'
# config_path = './project/example_eigencam/config_vgg_eigencam.yaml'
# config_path = './project/example_lrp/config_yolo_lrp.yaml'
# config_path = './project/example_lrp/config_yolo_lrp_alphabeta.yaml'
# config_path = './project/example_gradcam/config_yolo.yaml'
# config_path = './project/example_gradcampp/config_yolo_gradcampp.yaml'
# config_path = './project/example_guided_gradcam/config_yolo_guidedgradcam.yaml'
# config_path = './project/example_eigencam/config_yolo_eigencam.yaml'
# config_path = './projec0t/example_ig/config_yolo_ig.yaml'
# config_path = './project/example_gradcam/config_resnet_DG_gradcam.yaml'
# config_path = './project/example_gradcam/config_resnet_ABN_gradcam.yaml'
# config_path = './project/example_lrp/config_resnet_DG_lrp.yaml'
# config_path = './project/example_train/config_fgsm_resnet_train.yaml'
# config_path = './project/example_train/config_abn_resnet_train.yaml'
# config_path = './project/example_train/config_abn_vgg_train.yaml'
# config_path = './project/example_train/config_dg_resnet_train.yaml'
# config_path = './project/example_train/config_dg_vgg_train.yaml'
# config_path = './project/example_train/config_default_resnet_train.yaml'
# config_path = './project/example_train/config_default_vgg_train.yaml'
# config_path = './project/example_train/config_dann_resnet_train.yaml'
# config_path = './project/example_train/config_dann_grad_resnet_train.yaml'
# config_path = './project/example_lrp/config_resnet_lrp_alphabeta.yaml'
# config_path = './project/example_gradient/config_resnet_gradient.yaml'
# config_path = './project/example_gradientxinput/config_resnet_gradientxinput.yaml'
config_path = './project/example_smoothgrad/config_resnet_smoothgrad.yaml'
# config_path = './project/config_resnet_mnist_CF_prototype.yaml'


project = Project(config_path)
project.run()