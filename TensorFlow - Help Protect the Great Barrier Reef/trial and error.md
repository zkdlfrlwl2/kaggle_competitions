#### 2021 - 12 - 20 ~ 26

* 시도해볼만한 것들

  * Unlabeled data 활용 방안
    * 0 or 1 값으로 starfish 유무 판단 label 추가
      * 0 or 1 값 분류하는 classification head 추가 - Aux loss 
      
    * 각 image에 존재하는 starfish 수 label 추가
      * starfish 수 예측하는 regression head 추가 - Aux loss
      
    * 위 두 방법을 적용하려면 YOLOX 코드를 뜯어서 입맛에 맞게 변경해야할 듯
      * 이 방법이 잘 먹히면 YOLOv5도 동일하게 해서 nms or wbf ensemble 적용
  * Yolov5 loss function 변경
      * F2 score를 인자로 추가 
      * R & P 추가하여 R 값에 가중치 부여 
  
  
  * hyp.yaml 수정 



#### 2021 - 12 - 27 ~ 

* WBF 시도
  * yolov5
  * yolox
  * faster rcnn
* Tracking 시도
* 이미지 augmentation
  * yolo hyp.yaml 이외의 augmentation 적용

    * 모델에 입력으로 넣기 전 수중 이미지를 좀 더 선명하게 하는 preprocessing
* yolov5는 mAP 0.5 ~ 0.95이다. 하지만 kaggle 대회는 mAP 0.3 ~ 0.8 interval 0.05 이다. yolov5의 코드를 0.3 ~ 0.8, 0.05로 바꾸면 CV와 LB 차이가 더 줄어들지 않을까 싶은데 
* unlabeled data 사용
  * cls: bgr
  * bbox: 0.0
* cls는 cots 밖에 없으니 cls에 각 이미지의 bbox 갯수로 채워 넣는다면 ? 1 ~ 18개니까 18개 분류가 될 듯
* yolov5 augmentation 적용법  
  * 대부분 augmentation 기법은 yolov5/utils/datasets.py와 hyp.yaml에 정의되어 있는 듯
  * 추가하려면 datasets.py에 추가하고 hyp.yaml에 인자값도 추가
