from project.make_project import Project

config_path = './project/config_yolo_lrp.yaml'

project = Project(config_path)
project.run()