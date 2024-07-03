from project.make_project import make_attirbution

project = make_attirbution('./config_yolo.yaml')
# project = make_attirbution('./config_resnet.yaml')
project.run()
