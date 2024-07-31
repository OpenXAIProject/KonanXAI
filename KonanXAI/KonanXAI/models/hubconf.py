import sys
import os

import torch
import torch.nn as nn
# 경로 설정 general하게 맞춰야??
path = '/mnt/d/KonanXAI_implement_example2/ultralytics/'
sys.path.append(path)

# 모델별로 경로를 만들까..? 리포지토리별로 경로를 만들까?..


# hubconf.py가 있는 path가 기준 경로
# ex) path = '/mnt/d/KonanXAI_implement_example2/yolov5/'



class Repository:
    def __init__(self, repo_or_dir, model_name):
        self.repo_or_dir = repo_or_dir
        self.model_name = model_name
        self.path = self.repo_or_dir
        
        
        #self._import_path(self.path)


    def _add_to_sys_path(self):
        sys.path.insert(0, self.path)

    def _set_path(self):
        self.path = self.repo_or_dir

    # def _import_path(self):
    #     pass

    def _check_hubconf(self):
        return os.path.isfile(self.path + 'hubconf.py')
    
    def _read_hubconf(self):
        pass
    
    def _load(self):
        pass

    def _load_weight(self):
        pass




class Git(Repository):
    def __init__(self, repo_or_dir, model_name):
        Repository.__init__(self, repo_or_dir, model_name)

    def _download_from_url(self, cache_or_local):
        git_url = 'https://github.com/'+ self.repo_or_dir +'.git'
    
        if cache_or_local == 'cache':
            command = 'git' + ' -C ' + '~/.cache' + ' clone' + git_url
            os.system(command) 
            
        
        else:
            command = 'git' + ' -C ' + cache_or_local + ' clone ' + git_url
            os.system(command)

        repository_name = self.repo_or_dir.split('/')[1]
        self.path = cache_or_local + repository_name + '/'
    
    def _load(self):
        print(self.path)
        self.check_hubconf = self._check_hubconf()
        print(self.check_hubconf)

        self._add_to_sys_path()
        # 아래는 ultralytics/yolov5 리포지토리 코드
        # hubconf 공통으로 작성했을 때 읽어들이는 코드 필요
        if self.check_hubconf == True:
            import hubconf
            model = hubconf._create(self.model_name)
            return model
        # hubconf 규약대로 작성한 경우 어떻게 로드할 지 작성해야    
        
        # 에러로 처리해야
        else:
            print('write hubconf.py')
            return None

        model = None
        return model
    
    def _load_weight(self):
        pass
    
        
    def _save_from_cache(self):
        pass


class Local(Repository):
    def __init__(self, repo_or_dir, model_name):
        Repository.__init__(self, repo_or_dir, model_name)


# 예전 import 코드    
# class Ensemble(nn.ModuleList):
#     def __init__(self):
#         super().__init__()
        
#     def forward(self, x, augment=False, profile=False, visualize=False):
#         y = []
#         for module in self:
#             y.append(module(x, augment, profile, visualize)[0])
        
#         y = torch.cat(y,1)
#         return y, None



# 클래스 이름을 어떻게 할지 고민이네
class Yolov5(Local):
    def __init__(self, repo_or_dir, model_name):
        Local.__init__(self, repo_or_dir, model_name)

    # 이거 안되네?
    # def _import_path(self):
    #     import hubconf

    # common.py에서 import ultralytics 에러 
    def _load(self):
        self.check_hubconf = self._check_hubconf()
        self._add_to_sys_path()
        if self.check_hubconf == True:
            import hubconf
            model = hubconf._create(self.model_name)
            return model
        # hubconf 규약대로 작성한 경우 어떻게 로드할 지 작성해야    
        
        # 에러로 처리해야
        else:
            print('write hubconf.py')
            return None
        
class Ultralytics(Local):
    def __init__(self, repo_or_dir, model_name):
        Local.__init__(self, repo_or_dir, model_name)

    def _load(self):
        self.check_hubconf = self._check_hubconf()
        self._add_to_sys_path()
        from ultralytics.models.yolo.model import YOLO
        model = YOLO(self.model_name)
        return model 



    # def _load(self):
    #     from models.yolo import Detect
    #     from models.yolo import Model as YoloModel
    #     from models.common import C3, Conv, Bottleneck, Concat, SPPF
    #     from models.experimental import Ensemble
    #     from models.experimental import Ensemble



            

    
        


