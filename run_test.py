from project.make_project import make_attirbution
# project = make_attirbution('./config_darknet.yaml')
# project = make_attirbution('./config_yolo.yaml')
# project = make_attirbution('./config_resnet_lrp.yaml')
# project = make_attirbution('./config_vgg_lrp.yaml')
# project = make_attirbution('./config_efficientnet_lrp.yaml')
project = make_attirbution('./config_yolo_lrp.yaml')
project.run()
