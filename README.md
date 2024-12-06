# KonanXAI
- Explain(GradCAM, GradCAM++, EigenCAM, LRP, YOLOLRP)
- Train(ResNet50, VGG19)
  - Default, ABN, DomainGeneralization
## Dtrain, DarkNet 사용시 
- Dtrain 설치 방법
  - git clone http://10.10.16.165:6272/vision-ai-lab/kai2021.git
  - cd kai2021
  - git checkout releases_xai
  - visual studio 2019를 통한 빌드 혹은 빌드된 압축파일을 kai2021/out/ 경로에 압축해제 하여 아래와 같이 경로를 맞춤(ex: C:\Users\user\kai2021\out\Release)
  - C:\Users\user\kai2021\source\python 경로로 이동하여 pip install . 명령어를 통해 프레임워크 및 API 설치

- DarkNet 설치 방법
  - git clone http://10.10.18.132:6118/vision-recognition-team/xai_darknet.git
  - cd xai_darknet
  - cuda 11환경의 경우: git checkout dev_cuda_11x_wt
  - cuda 12환경의 경우: git checkout dev_cuda_12x_wt 
  - pip install . 명령어를 통해 Darknet 프레임워크 및 API 설치

## 사용법
- project 파일에 사용할 yaml파일 작성
- tutorial_with_project.py에 생성한 yaml 파일 경로 지정
- 최상단 루트에서 tutorial_with_project.py 실행
 
# Yaml 파일 작성 요령
- ## yaml파일은 크게 2가지로 구성 됩니다.
  - ### head
  - ### train or explain
- ## 만약 해당 파라미터가 불필요한 경우 ~ 를 사용합니다.(None 처리를 위함)

- ## head의 필수 구성은 다음과 같습니다.
    - ### project_type:
        - 학습을 원할 경우 train, attribution을 원할 경우 explain을 지정합니다.
        - ex) "train"
    - ### save_path: 
        - 저장될 경로를 지정합니다.
        - ex) "./heatmap/"
    - ### weight_path: 
        - 사용자가 사용할 모델의 경로를 지정합니다.
        - ex) "./yolov5s.pt"
    - ### cfg_path:
        - yolo모델을 선택할 경우 yaml파일 혹은 cfg파일의 경로를 지정합니다.
        - ex) "./yolov5s.yaml"
    - ### data_resize: 
        - 모델에 맞게 데이터 크기를 재설정 합니다. tuple 혹은 list 타입으로 지정합니다.
        - ex) [224,224] or (224,224)
    - ### data_path:
        - 데이터를 불러올 경로를 지정합니다.
        - ex) "D:/Datasets/ai_fire/train"
    - ### model_name:
        - 사용할 모델을 선택합니다. (torchvision을 사용할 경우 풀네임을 작성해야 합니다)
        - ex) vgg19, resnet50, efficientnet_b0, yolov5s
    - ### framework:
        - 어떤 프레임 워크를 사용할지 지정합니다.
        - ex) torch, darknet
    - ### source:
        - 프레임워크를 다운로드 할지 선택합니다.
        - ex) torchvision, github, local
            - torchvision의 경우 사용자 폴더의 .cache폴더에 다운로드 됩니다.
            - github을 선택하게 된다면 해당 프레임워크를 .cache 폴더에 다운로드 됩니다.
            - local을 선택하게 될 경우 repo_or_dir 파라미터의 경로를 참고하여 모델을 load 합니다.
    - ### repo_or_local:
        - repo 경로를 사용할지 local 경로를 사용할지 정합니다.
        - 내용 추가 필요
    - ### cache_or_local:
        - .cache 경로에 저장할지, 사용자가 지정한 경로에 저장할지 선택합니다.
        - ex) cache, "저장될 경로"
    - ### data_type:
        - datasets을 상속받아 구현된 dataloader를 지정합니다.
        - 사용자가 자신의 데이터셋에 맞게 커스텀해서 사용합니다.
        - ex) CUSTOM, AI_FIRE 등

- ## explain의 필수 구성은 다음과 같습니다.
    -  ### algorithm:
        - 알고리즘을 선택합니다. 
        - ex) gradcam, eigencam, gradcampp, lrp, lrpyolo 등
    - ### target_layer:
        - cam방식일 경우 hook의 대상을 지정합니다.
        - ex) [layer4,'2',relu]
    - ### rule:
        - lrp의 경우 어떤 rule을 사용할지 지정합니다.
        - ex) Epsilon

- ## train의 필수 구성은 다음과 같습니다.
    - ### epoch:
        - 학습할 횟수를 지정 합니다.
        - ex) 50
    - ### learning_rate:
        - 학습률에 대한 하이퍼 파라미터를 지정합니다.
        - ex) 0.0001
    - ### batch_size:
        - 한번에 몇장씩 처리할 지 지정합니다.
        - ex) 128
    - ### optimizer:
        - optimizer를 선택합니다.
        - ex) adam
    - ### loss_function:
        - 손실함수를 선택합니다.
        - ex) crossentropyloss
    - ### save_step:
        - 학습된 모델을 n번 간격으로 저장합니다.
        - ex) 10
    - ### improvement_algorithm:
        - ABN, DG 등 해당 알고리즘에 필요한 하이퍼 파라미터를 지정합니다.
        - 하위 모듈에는 algorithm, transefer_weights, gpu_count 가 있습니다.
        - algorithm은 abn, default, domaingeneralization 선택합니다.
            - ex) abn
        - trainsfer_weights는 전이학습 시킬 모델의 경로를 지정합니다.
            - ex) "./checkpoint/default_resnet50_10ep.pt"
        - gpu_count 는 학습에 사용될 gpu의 숫자를 지정합니다.
            - 현재는 0번부터 n번까지 숫자를 지정합니다.
            - ex) 3일 경우 0,1,2 gpu 사용

# Example YAML[single explain] and JSON[multi explain]
- yaml 파일의 경우 main.py 실행
- json의 경우 main_multi.py 실행
- ## Explain GradCAM for ResNet50 [YAML]
    ```
    head:
    project_type: 'explain'
    save_path: "./heatmap/"
    weight_path: "./resnet50-0676ba61.pth"
    cfg_path: ~
    data_path: "./data"
    data_resize: [224,224]
    model_name: resnet50
    framework: torch
    source: torchvision
    repo_or_dir: ~
    cache_or_local: cache
    data_type: CUSTOM

    explain:
    algorithm: GradCAM
    model_algorithm: Default
    target_layer: [layer4,'2',relu]
    ```
- ## Train for ResNet50 [YAML]
    ```
    head:
    project_type: 'train'
    save_path: "./checkpoint"
    weight_path: ~
    # weight_path: "./resnet50-0676ba61.pth"
    cfg_path: ~
    data_resize: [224,224]
    data_path: "D:/Datasets/ai_fire/train_lite"
    model_name: resnet50
    framework: torch
    source: torchvision
    repo_or_dir: ~
    cache_or_local: cache
    data_type: AI_FIRE
    
    train:
    epoch: 10
    learning_rate: 0.0001
    batch_size: 128
    optimizer: 'adam'
    loss_function: 'CrossEntropyLoss'
    save_step: 1
    improvement_algorithm: { 
        algorithm: Default,
        transfer_weights: ~,
        gpu_count: 1
    }
    ```
- ## Multi Explain and Evaluation [JSON]
   - explain: gradcam, lime, guidedgradcam
   - evaluation: AbPC, Sensitivity
    ```
    {
    "job_infos": [
        {
        "job_type": "evaluation",
        "model_type": "xai",
        "inputs": [
            {
            "model_info": {
                "framework": "torch",
                "weight_path": "D:/xai_refactoring/resnet50-0676ba61.pth",
                "cfg_path": null,
                "num_classes": 1000,
                "repo_or_dir": "C:/Users/jaehyeok/.cache/torch/hub/pytorch_vision_v0.11.0",
                "model_algorithm": "default",
                "model_name": "resnet50"
            },
            "data_info": {
                "data_path": "D:/xai_refactoring/data",
                "data_type": "CUSTOM",
                "resize": [224,224]
            },
            "type_config": {
                "algorithm_name": ["gradcam","lime","guidedgradcam"],
                "metric": ["ABPC", "sensitivity"],
                "gradcam": {"model_algorithm": "Default",
                            "target_layer": ["layer4","2","relu"]},
                "lime": {"model_algorithm": "Default",
                            "segments": {"algo_type": "slic", "n_segments": 40, "compactnes": 2, "sigma": 3},
                            "seed": 415,
                            "num_samples": 40,
                            "num_features": 10,
                            "positive_only": true,
                            "hide_rest": true},
                "guidedgradcam": {"model_algorithm": "Default",
                                    "target_layer": ["layer4","2","relu"]}
            }
            }
        ],
        "outputs": [
            {
            "volume": "D:/",
            "file_path": "save_test"
            }
        ]
        }
    ]
    }

    ```