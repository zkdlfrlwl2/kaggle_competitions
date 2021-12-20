#### 2021 - 12 - 20 ~ 26

* Unlabeled data 활용 방안
  * 0 or 1 값으로 starfish 유무 판단 label 추가
    * 0 or 1 값 분류하는 classification head 추가 - Aux loss 

  * 각 image에 존재하는 starfish 수 label 추가
    * starfish 수 예측하는 regression head 추가 - Aux loss

  * 위 두 방법을 적용하려면 YOLOX 코드를 뜯어서 입맛에 맞게 변경해야할 듯
    * 이 방법이 잘 먹히면 YOLOv5도 동일하게 해서 nms or wbf ensemble 적용


