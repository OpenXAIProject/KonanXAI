from redis import StrictRedis
from datetime import datetime
import os

class Status:
    WATTING = 'WATTING'
    ''' 작업 대기 중 '''
    PREPROCESSING = 'PREPROCESSING'
    ''' 전처리 진행 중 '''
    PREPROCESSED = 'PREPROCESSED'
    ''' 전처리 진행 완료 '''
    RUNNING = 'RUNNING'
    ''' Engine 작업 진행 중 '''
    COMPLETE = 'COMPLETE'
    ''' Engine 작업 완료 '''
    POSTPROCESSING = 'POSTPROCESSING'
    ''' 후처리 진행 중 '''
    SUCCESS = 'SUCCESS'
    ''' 작업 완료(성공) '''
    CANCEL = 'CANCEL'
    ''' 작업 취소 '''
    FAIL = 'FAIL'
    ''' 작업 실패(에러) '''
    
class Infomation:
    MESSAGE = 'MESSAGE'
    '''
    type: str\n
    description: 진행중인 작업에 대해 알리고 싶은 메세지
    '''
    STATUS = 'STATUS'
    '''
    type: str\n
    description: 작업이 어떤 상태인지
    '''
    PROGRESS = 'PROGRESS'
    '''
    type: int\n
    description: 진행중인 작업의 진행도
    '''
    AI_PROGRESS = 'AI_PROGRESS'
    '''
    type: float\n
    description: 진행중인 Engine 작업의 진행도
    '''
    JOB_TYPE = 'JOB_TYPE'
    '''
    type: str\n
    description: 학습인지, 인식인지, 추천인지
    '''
    MODEL_TYPE = 'MODEL_TYPE'
    '''
    type: str\n
    description: 어떤 모델인지
    '''
    INPUT = 'INPUT'
    '''
    type: list[dict]\n
    description: User가 요청한 Input 정보
    '''
    OUTPUT = 'OUTPUT'
    '''
    type: list[dict]\n
    description: User가 요청한 Output 정보
    '''
    QUEUE = 'QUEUE'
    '''
    type: list[str]\n
    description: 진행중인 작업 및 대기중인 작업들의 job_id
    '''
    
class RedisController:
    def __init__(self):
        self.job_id = None
        self.redis = StrictRedis(host='10.10.30.15',
                               port=3079,
                               charset="utf-8",
                               decode_responses=True)
    
    def iset(self, key, field, value, log=False):
        ''' type에 상관 없이 dict로 감싸서 value로 던짐 '''
        set_item = {'type': str(type(value)), 'data': value}
        self.redis.hset(key, field, str(set_item))
        if log: Logger('REDIS', f"[{key}] SET {field} {set_item}")    
    
    def get(self, key, value, log=False):
        ''' str로 들어온 데이터 타입을 dict로 변환 '''
        if key is None: return None
        get_item = eval(self.redis.hget(key, value))
        if log: Logger('REDIS', f"[{key}] GET {get_item['type']} {get_item['data']}")
        return get_item['data']
    
    def cancel_check(self, key):
        ''' 모든 작업을 하기 전, Cancel 확인 '''
        status = self.get(key, Infomation.STATUS)
        return False if status == 'CANCEL' else True
            
    def set(self, key, field, value, log=False):
        ''' set하기 전에 CANCEL 명령 있는지 확인하고 진행 '''
        if not self.cancel_check(key):
            return Logger('REDIS', f"[{key}] CANCEL")
        self.iset(key, field, value, log)
    
    def qset(self, key, log=False):
        ''' QUEUE에 Job 삽입'''
        self.redis.rpush('QUEUE', key)
        if log: Logger('REDIS', f"[{key}] QSET")
        
    def qget(self, log=False):
        ''' QUEUE에 있는 모든 Job 조회'''
        queue = self.redis.lrange('QUEUE', 0, -1)
        if queue == list(): return queue
        if log: Logger('REDIS', f"[{queue[0]}] QGET ALL: {queue}")
        return queue
    
    def qdel(self, key, log=True):
        ''' QUEUE에 있는 특정 Job_ID 삭제'''
        self.redis.lrem('QUEUE', 0, key)
        if log: Logger('REDIS', f"[{key}] QDEL")
        
    def qpop(self, log=False):
        ''' QUEUE에 대기중인 첫 Job_ID pop'''
        job_id = self.redis.lpop('QUEUE')
        if log: Logger('REDIS', f"[{job_id}] QPOP")
        
    def publish(self, ch, msg):
        ''' 특정 Channel을 구독중인 Container들에게 Publish '''
        self.redis.publish(ch, msg)
        Logger('REDIS', f'CH: {ch} Message: {msg}')
        
    def subscribe(self, ch):
        ''' 특정 Channel을 구독하고 있다가, message 수신'''
        # 실제 AI Engine에서는 이 메서드를 쓰지 말고 함수로 따로 가져가서 사용
        subscriber = self.redis.pubsub()
        subscriber.subscribe(ch)
        Logger('REDIS', f'CH: {ch}')
        for message in subscriber.listen():
            if message['type'] != 'subscribe':
                Logger('REDIS', f'MESSAGE: {message}, TYPE: {type(message)}')
                self.job_id = self.qget()[0]
                yield message['data']
    
    def get_redis(self):
        _input = self.get(self.job_id, Infomation.INPUT)
        _output = self.get(self.job_id, Infomation.OUTPUT)
        _status = self.get(self.job_id, Infomation.STATUS)
        return {'input': _input, 'output': _output, 'status':_status}
    
    def running_check(self):
        self.set(self.job_id, Infomation.STATUS, Status.RUNNING) # Engine이 작업을 잘 받았다는 플래그
        self.set(self.job_id, Infomation.MESSAGE, 'Engine Process Start!')
    
    def complete_check(self):
        self.set(self.job_id, Infomation.STATUS, Status.COMPLETE) # Engine이 작업을 잘 끝냈다는 플래그
        self.set(self.job_id, Infomation.MESSAGE, 'Engine Process End!')
            
class Logger:
    def __init__(self, option:str, msg:str):
        """_summary_
        Args:
            option (str): [INFO, ERROR, REDIS]
            msg (str): Log Message
        """
        self.log_path = f"/home/server/logs" # LOG를 저장할 특정 PATH 지정 필요
        ymd = datetime.now().strftime('%Y-%m-%d')
        hms = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        os.makedirs(f"{self.log_path}/{ymd}", exist_ok=True)
        
        with open(f"{self.log_path}/{ymd}/{ymd}.log",mode='a') as write_log:
            log_data = f"[{hms}]\t{option}\t{msg}\n"
            write_log.write(log_data)
            print(log_data.rstrip())
