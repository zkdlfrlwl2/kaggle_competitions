|     Model      |  CV   | mAP .3 | mAP .3:.05:.8 | Public LB | Private LB |
| :------------: | :---: | :----: | :-----------: | :-------: | :--------: |
| yolov5-m ver1  | 0.447 |   -    |       -       |   0.324   |            |
| yolov5-m ver2  |   -   |   -    |       -       |   0.374   |            |
| yolov5-m ver3  | 0.539 |   -    |       -       |   0.425   |            |
| yolov5-m ver4  | 0.546 |   -    |       -       |   0.435   |            |
| yolov5-m ver5  |   -   |   -    |       -       |   0.394   |            |
| yolov5-m ver6  | 0.534 | 0.578  |     0.460     |   0.419   |            |
| yolov5-m ver7  | 0.588 | 0.636  |     0.525     |   0.435   |            |
| yolov5-m ver8  | 0.593 | 0.628  |     0.517     |   0.459   |            |
| yolov5-m ver9  | 0.618 | 0.674  |     0.544     |   0.477   |            |
| yolov5-m ver10 | 0.617 | 0.661  |     0.538     |   0.452   |            |
| yolov5-m ver11 | 0.622 | 0.678  |     0.553     |   0.484   |            |
| yolov5-l ver12 | 0.613 | 0.649  |     0.528     |   0.497   |            |
| yolov5-l ver13 | 0.607 | 0.658  |     0.533     |   0.468   |            |
| yolov5-l ver14 | 0.642 | 0.682  |     0.551     |   0.426   |            |
| yolov5-m ver15 | 0.636 | 0.688  |     0.564     |   0.442   |            |
| yolov5-m ver16 | 0.705 | 0.784  |     0.678     | **0.506** |            |
| yolov5-m ver17 | 0.633 | 0.792  |     0.677     |   0.463   |            |
| yolov5-l ver18 | 0.754 | 0.780  |     0.663     |   0.431   |            |
| yolov5-l ver19 | 0.702 | 0.769  |     0.665     |   0.506   |            |
|                |       |        |               |           |            |

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
    * mAP@.3: 0.655, mAP@.3:.8: 0.533
    * **conf thres: 0.2**, iou_thres: 0.3: 0.548, LB: 0.498
    * **conf thres: 0.15**, iou_thres: 0.3: 0.609, LB: 0.497



* ver13
  * epochs=30
  * batch=6
  * patience=10
  * yolov5l6.pt
  * ver11 hyp base
    * 변경점
      * hsv_h: 0.2 -> 0.015
      * shear: 0.0 -> 0.2
      * mixup: 0.5 -> 0.4
  * result
    * best
      * CV: 0.607
      * LB: 0.468



* ver14

  * epochs=30
  * batch=6
  * patience=10
  * yolov5l6.pt
  * ver11 hyp base

    * 변경점

      * degree: 0.0 -> 20.0
  * result
    * best
      * CV: 0.642
      * LB: 0.426



* ver15
  * epochs=50
  * batch=10
  * patience=10
  * yolov5m6.pt
  * ver14 hyp base
  * result
    * **conf thres: 0.2**, iou_thres: 0.3: 0.670, LB: 0.446
      * mAP@.3: 0.759, mAP@.3:.8: 0.656
    * **conf thres: 0.15**, iou_thres: 0.3: 0.645, LB: 0.442
      * mAP@.3: 0.755, mAP@.3:.8: 0.649



* ver16
  * epochs=50
  * batch=10
  * patience=10
  * yolov5m6.pt
  * ver14 hyp base
  * data split 방식 변경
    * .2 split 대신 5fold 사용
    * val: fold=4
  * result
    * best
      * CV
        * 변경된 방법
          * **conf thres: 0.4**, iou_thres: 0.3: 0.669
            * mAP@.3: 0.776, mAP@.3:.8: 0.682 
          * **conf thres: 0.3**, iou_thres: 0.3: 0.768
            * mAP@.3: 0.783, mAP@.3:.8: 0.682 
          * **conf thres: 0.2**, iou_thres: 0.3: 0.705
            * mAP@.3: 0.784, mAP@.3:.8: 0.678
      * LB
        * conf thres: 0.4, iou_thres: 0.3: 0.476
        * conf thres: 0.3, iou_thres: 0.3: 0.494
        * conf thres: 0.2, iou_thres: 0.3: 0.505
        * conf thres: 0.15, iou_thres: 0.3: 0.506



* ver17
  * epochs=40
  * batch=10
  * patience=5
  * yolov5m6.pt
  * ver14 hyp base
    * 변경점
      *   lr0 = 0.0032
      *  lrf = 0.12
      * momentum = 0.843
      * weight_decay = 0.00036
      * warmup_epochs = 2.0
      * warmup_momentum = 0.5
      * warmup_bias_lr = 0.05
  * data split 방식 변경
    * .2 split 대신 5fold 사용
    * val: fold=4
  * result
    * best
      * CV
        * **conf thres: 0.4**, iou_thres: 0.3: 0.667
          * mAP@.3: 0.778, mAP@.3:.8: 0.675
        * **conf thres: 0.3**, iou_thres: 0.3: 0.764
          * mAP@.3: 0.786, mAP@.3:.8: 0.679
        * **conf thres: 0.2**, iou_thres: 0.3: 0.706
          * mAP@.3: 0.792, mAP@.3:.8: 0.679
        * **conf thres: 0.15**, iou_thres: 0.3: 0.633
          * mAP@.3: 0.792, mAP@.3:.8: 0.677
      * LB
        * conf thres: 0.4, iou_thres: 0.3: 0.433
        * conf thres: 0.3, iou_thres: 0.3: 0.446
        * conf thres: 0.2, iou_thres: 0.3: 0.459
        * conf thres: 0.15, iou_thres: 0.3: 0.463



* ver18
  * epochs=40
  * batch=6
  * patience=5
  * yolov5l6.pt
  * ver14 hyp base
  * data split 방식 변경
    * .2 split 대신 5fold 사용
    * val: fold=4
  * result
    * best
      * CV
        * **conf thres: 0.4**, iou_thres: 0.3: 0.640
          * mAP@.3: 0.763, mAP@.3:.8: 0.658
        * **conf thres: 0.3**, iou_thres: 0.3: 0.674
          * mAP@.3: 0.773, mAP@.3:.8: 0.662
        * **conf thres: 0.2**, iou_thres: 0.3: 0.754
          * mAP@.3: 0.78, mAP@.3:.8: 0.663 
        * **conf thres: 0.15**, iou_thres: 0.3: 0.687
          * mAP@.3: 0.784, mAP@.3:.8: 0.662
      * LB
        * conf thres: 0.4, iou_thres: 0.3: 0.420
        * conf thres: 0.3, iou_thres: 0.3: 0.429
        * conf thres: 0.2, iou_thres: 0.3: 0.431
        * conf thres: 0.15, iou_thres: 0.3: 0.429



* ver19
  * ver12와 동일, data split 방식만 fold=4로 변경
  * result
    * CV
      * best
        * **conf thres: 0.2**, iou_thres: 0.3, f2 score: 0.702
          * mAP@.3: 0.769, mAP@.3:.8: 0.665
        * **conf thres: 0.15**, iou_thres: 0.3, f2 score: 0.719
          * mAP@.3: 0.769, mAP@.3:.8: 0.662
      * last
        * **conf thres: 0.2**, iou_thres: 0.3, f2 score: 0.668(new), 0.639
          * mAP@.3: 0.761, mAP@.3:.8: 0.657
        * **conf thres: 0.15**, iou_thres: 0.3, f2 score: 0.686(new), 0.651
          * mAP@.3: 0.763, mAP@.3:.8: 0.656
    * LB
      * best
        * conf thres: 0.2, iou_thres: 0.3: 0.506
        * conf thres: 0.15, iou_thres: 0.3: 0.504 
      * last
        * conf thres: 0.2, iou_thres: 0.3: 0.478
        * conf thres: 0.15, iou_thres: 0.3: 0.478



* ver20
  * ver19와 동일
  * metric의 fitness
    * [0.0 0.0 0.1 0.9] -> [0.0 0.8 0.0 0.2]으로 변경



* ver21
  * LB 가장 잘 나온 모델 config 그대로
  * iou_t: 0.20 -> 0.30으로 변경 




### Reference

* https://www.kaggle.com/remekkinas/yolox-training-pipeline-cots-dataset-lb-0-507?scriptVersionId=81353936
* https://www.kaggle.com/julian3833/reef-a-cv-strategy-subsequences
* https://www.kaggle.com/yamqwe/great-barrier-reef-yolox-yolov5-ensemble
* https://www.kaggle.com/parapapapam/yolox-inference-tracking-on-cots-lb-0-539/notebook
