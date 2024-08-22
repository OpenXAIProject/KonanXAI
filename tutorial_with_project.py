from project.make_project import Project

config_path = './project/config_yolo.yaml'

project = Project(config_path)
project.run()