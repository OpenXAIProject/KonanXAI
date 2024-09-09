from project.make_project import Project

# config_path = './project/config_darknet.yaml'
# config_path = './project/config_darknet_gradcampp.yaml'
# config_path = './project/config_darknet_eigencam.yaml'
# config_path = './project/config_efficientnet_lrp.yaml'
# config_path = './project/config_efficientnet_gradcam.yaml'
# config_path = './project/config_resnet_lrp.yaml'
config_path = './project/config_resnet_mnist_CF_prototype.yaml'
# config_path = './project/config_resnet_gradcam_clustering.yaml'
# config_path = './project/config_resnet_smoothgrad.yaml'
# config_path = './project/config_resnet_gradient.yaml'
# config_path = './project/config_resnet_gradcam.yaml'
# config_path = './project/config_resnet_gradcampp.yaml'
# config_path = './project/config_resnet_guidedgradcam.yaml'
# config_path = './project/config_resnet_eigencam.yaml'
# config_path = './project/config_vgg_lrp.yaml'
# config_path = './project/config_vgg_gradcam.yaml'
# config_path = './project/config_yolo_lrp.yaml'
# config_path = './project/config_yolo.yaml'
# config_path = './project/config_abn_resnet_train.yaml'
# config_path = './project/config_abn_vgg_train.yaml'
# config_path = './project/config_dg_resnet_train.yaml'
# config_path = './project/config_dg_vgg_train.yaml'
# config_path = './project/config_default_resnet_train.yaml'
# config_path = './project/config_default_vgg_train.yaml'
# config_path = './project/config_dann_resnet_train.yaml'
# config_path = './project/config_dann_grad_resnet_train.yaml'
project = Project(config_path)
project.run()