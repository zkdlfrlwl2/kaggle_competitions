| Model |  CV   | mAP .3 | mAP .3:.05:.8 | Public LB | Private LB |
| :---: | :---: | :----: | :-----------: | :-------: | :--------: |
| ver1  | 0.447 |   -    |       -       |   0.324   |            |
| ver2  |   -   |   -    |       -       |   0.374   |            |
| ver3  | 0.539 |   -    |       -       |   0.425   |            |
| ver4  | 0.546 |   -    |       -       |   0.435   |            |
| ver5  |   -   |   -    |       -       |   0.394   |            |
| ver6  | 0.534 | 0.578  |     0.460     |   0.419   |            |
| ver7  | 0.588 | 0.636  |     0.525     |   0.435   |            |
| ver8  | 0.593 | 0.628  |     0.517     |   0.459   |            |
| ver9  | 0.618 | 0.674  |     0.544     |   0.477   |            |
| ver10 | 0.617 | 0.661  |     0.538     |   0.452   |            |
| ver11 | 0.622 | 0.678  |     0.553     |   0.484   |            |
| ver12 | 0.613 | 0.649  |     0.528     |   0.497   |            |
|       |       |        |               |           |            |

※ 목표 Public LB 0.60 ↑



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
  * result
    * best
      * CV: 0.593
      * LB: 0.459



* ver9
  * epochs=50
  * patience=20
  * yolov5m6.pt
  * ver9 hyp
    * hyp.scratch-med.yaml base
    * lr0: 0.01 -> 0.001
    * warmup_epochs: 3.0 -> 5.0
    * flipud: 0.0 -> 0.5
    * mixup: 0.1 -> 0.2
  * result
    * best
      * CV: 0.618
      * LB: 0.477



* ver10
  * epochs=50
  * patience=20
  * yolov5m6.pt
  * ver10 hyp
    * hyp.scratch-high.yaml base
    * lr0: 0.01 -> 0.001
    * warmup_epochs: 3.0 -> 5.0
    * flipud: 0.0 -> 0.5
    * mixup: 0.1 -> 0.2
    * copy_paste: 0.1 -> 0.2
  * result
    * best
      * CV: 0.617
      * LB: 0.452



* ver11
  * epochs=50
  * batch=10
  * patience=20
  * yolov5m6.pt
  * ver9 hyp base
    * 바뀐점
      * hsv_h: 0.015 -> 0.2
      * mixup: 0.2 -> 0.5
  * result
    * best
      * CV: 0.622
      * LB: 0.484



* ver12
  * epochs=30
  * batch=6
  * patience=10
  * yolov5l6.pt
  * ver11 hyp
  * result
    * best
      * CV: 0.613
      * LB: 0.497



### Reference

* https://www.kaggle.com/remekkinas/yolox-training-pipeline-cots-dataset-lb-0-507?scriptVersionId=81353936
* https://www.kaggle.com/julian3833/reef-a-cv-strategy-subsequences
* https://www.kaggle.com/yamqwe/great-barrier-reef-yolox-yolov5-ensemble
* https://www.kaggle.com/parapapapam/yolox-inference-tracking-on-cots-lb-0-539/notebook
