from piper import RedisController, Status, Infomation
from time import sleep
from atexit import register
from traceback import format_exc
import sys
from KonanXAI.autoXAI import run_xai
import os
## 임시용: piper에 들어갈 내용
def preprocessing(rc, job_id, job_type, model_type):
    response = rc.get_redis() # API Server를 통해 들어온 작업 정보
    _status, _input, _output = response['status'], response['input'], response['output']
    print(_input)
    # XAI TRAINING
    if job_type == "train" and model_type == "xai":
        save_dir = _output[0]['volume'] + "/" + _output[0]['file_path']
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        model_param = _input[0]['model_info']
        datasets_param = _input[0]['data_info']
        datasets_param['framework'] = model_param['framework']
        datasets_param['mode'] = job_type
        arg_param = _input[0]['type_config']
        arg_param['framework'] = model_param['framework']
        arg_param['model_name'] = model_param['model_name']
        _modify_input = [{'model_param': model_param, 'datasets_param': datasets_param, "arg_param": arg_param}]
        rc.set(job_id, Infomation.INPUT, _modify_input)
    # XAI INFERENCE
    elif job_type == "explain" and model_type == "xai":
        save_dir = _output[0]['volume'] + "/" + _output[0]['file_path']
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        model_param = _input[0]['model_info']
        datasets_param = _input[0]['data_info']
        datasets_param['framework'] = model_param['framework']
        datasets_param['mode'] = job_type
        arg_param = _input[0]['type_config']
        arg_param['framework'] = model_param['framework']
        _modify_input = [{'model_param': model_param, 'datasets_param': datasets_param, "arg_param": arg_param}]
        rc.set(job_id, Infomation.INPUT, _modify_input)
    # XAI evaluation
    elif job_type == 'evaluation' and model_type == "xai":
        save_dir = _output[0]['volume'] + "/" + _output[0]['file_path']
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        model_param = _input[0]['model_info']
        datasets_param = _input[0]['data_info']
        datasets_param['framework'] = model_param['framework']
        datasets_param['mode'] = job_type
        arg_param = _input[0]['type_config']
        arg_param['framework'] = model_param['framework']
        _modify_input = [{'model_param': model_param, 'datasets_param': datasets_param, "arg_param": arg_param}]
        rc.set(job_id, Infomation.INPUT, _modify_input)
    # XAI explainer
    elif job_type == "explainer" and model_type == "xai":
        pass
class FirePower:
    def __init__(self):
        self.rc = RedisController()
        self.ch = 'xai' # Redis에 구독 할 Channel Name 입력
        self.job_id:str
        self.job_type:str
        self.except_msg: str
        
    def exception(self):
        # 에러로 인한 프로그램 종료 전, 에러 메세지 남기기 - ERROR_MSG에 ERR_MSG인자 전달
        if self.rc.get(self.job_id, Infomation.STATUS) == Status.SUCCESS: return
        if self.rc.get(self.job_id, Infomation.STATUS) == Status.CANCEL: return
        
        self.rc.set(self.job_id, Infomation.STATUS, Status.FAIL)
        self.rc.set(self.job_id, Infomation.MESSAGE, self.except_msg)
        
    def worker(self):
        register(self.exception) # 에러 발생으로 인한 서버 재부팅 시 에러 남기기 위함
        gen = self.rc.subscribe(ch=self.ch)
        
        try:        
            while True:
                job_type = next(gen)
                job_id = self.rc.job_id
                self.rc.set(job_id, Infomation.STATUS, Status.PREPROCESSING)
                preprocessing(self.rc, job_id, job_type, self.ch)
                self.rc.set(job_id, Infomation.STATUS, Status.PREPROCESSED)
                self.rc.running_check()
                
                response = self.rc.get_redis() # API Server를 통해 들어온 작업 정보
                _status, _input, _output = response['status'], response['input'], response['output']

                model_param = _input[0]['model_param']
                datasets_param = _input[0]['datasets_param']
                arg_param = _input[0]['arg_param']
                output = _output[0]
                work_handle = run_xai(job_type, model_param, datasets_param, arg_param, output)
                for progress in work_handle:
                    if self.rc.get(job_id, Infomation.STATUS) == Status.CANCEL:
                        sys.exit() # CANCEL 명령 오면 프로그램 다운, docker로 restart
                    self.rc.set(job_id, Infomation.AI_PROGRESS, progress)
        
                self.rc.complete_check() # Engine이 작업을 잘 끝냈다는 플래그
        except Exception:
            self.except_msg = format_exc() # 에러 메시지를 저장
            print(self.except_msg)
            sys.exit()
            
if __name__ == '__main__':
    sleep(5) # Server Down을 판단하기 위한 플래그
    fire_power = FirePower()
    fire_power.worker()
