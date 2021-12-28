| Model |  CV   | Public LB | Private LB |
| :---: | :---: | :-------: | :--------: |
| ver1  | 0.447 |   0.324   |            |
| ver2  |   -   |   0.374   |            |
| ver3  | 0.539 |   0.425   |            |
| ver4  | 0.546 |   0.435   |            |
| ver5  |   -   |   0.394   |            |
| ver6  | 0.534 |   0.419   |            |
| ver7  | 0.588 |   0.435   |            |
| ver8  |       |           |            |



* ver1
  * Model: YOLOv5-m6
  * config
    * batchsize 10, epoch 50
    * 80/20 split base subsequence
    * delete unlabeled data
  * result
    * best
      * LB: 0.324



* ver2
  * ver1와 다른점
    * hyper parameter
      * flipud = 0.5
  * 결과
    * best
      * LB: 0.374
    * last
      * LB: 0.374



* ver3
  * ver2와 다른점
    * hyper parameter
      * lr0 = 0.001
      * patience=10
  * 결과
    * best
      * CV: 0.539
      * LB: 0.425
    * last
      * CV: 0.532
      * LB: 0.444



* ver4
  * ver3와 다른점
    * hyper parameter
      * warmup_epochs = 5.0
      * patience=20
  * result
    * best
      * CV: 0.546
      * LB: 0.435



* ver5
  * ver4와 다른점
    * metric의 fitness
      * [0.0 0.0 0.1 0.9] -> [0.0 0.3 0.1 0.6]으로 변경
  * result
    * best
      * LB: 0.394



* ver6
  * ver4와 다른점
    * epoch=10 테스트 목적
    * mAP .5:.95 -> .3:.8로 변경
  * result
    * best
      * CV: 0.534
      * LB: 0.419



* ver7
  * epoch=50
  * patience=15
  * hyper parameter
    * hyp.scratch-med.yaml 적용
  * result
    * best
      * CV: 0.588
      * LB: 0.435



* ver8
  * epoch=50
  * patience=15
  * hyper parameter
    * hyp.scratch-high.yaml 적용





### Reference

* https://www.kaggle.com/remekkinas/yolox-training-pipeline-cots-dataset-lb-0-507?scriptVersionId=81353936
* https://www.kaggle.com/julian3833/reef-a-cv-strategy-subsequences
* https://www.kaggle.com/yamqwe/great-barrier-reef-yolox-yolov5-ensemble
* https://www.kaggle.com/parapapapam/yolox-inference-tracking-on-cots-lb-0-539/notebook
