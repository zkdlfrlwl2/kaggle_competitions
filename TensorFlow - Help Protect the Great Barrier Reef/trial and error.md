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
  * 이미지 augmentation
  
      * yolo hyp.yaml 이외의 augmentation 적용
        
        * 모델에 입력으로 넣기 전 수중 이미지를 좀 더 선명하게 하는 preprocessing
  
  
  * hyp.yaml 수정 

