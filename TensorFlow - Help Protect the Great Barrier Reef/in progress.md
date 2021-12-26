| Model |  CV   | Public LB | Private LB |
| :---: | :---: | :-------: | :--------: |
| ver1  | 0.447 |   0.324   |            |
| ver2  |   -   |   0.374   |            |
| ver3  | 0.539 |   0.425   |            |
| ver4  | 0.546 |   0.435   |            |
|       |       |           |            |



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



### Reference

* https://www.kaggle.com/remekkinas/yolox-training-pipeline-cots-dataset-lb-0-507?scriptVersionId=81353936
* https://www.kaggle.com/julian3833/reef-a-cv-strategy-subsequences
* https://www.kaggle.com/yamqwe/great-barrier-reef-yolox-yolov5-ensemble
