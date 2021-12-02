|       Model       |             CV             |     Public LB     | Private LB |
| :---------------: | :------------------------: | :---------------: | :--------: |
|  Base - 10 folds  |          18.35465          |       18.49       |            |
|  ver1 - 10folds   |          18.02539          |     18.10953      |            |
|  ver2 - 10folds   |          17.94588          |     18.30072      |            |
|  ver3 - 10folds   |          18.01586          |     18.24908      |            |
|  ver4 - 10folds   |          18.35333          |     18.40091      |            |
|  ver5 - 10folds   |          17.79845          |     18.03773      |            |
|  ver6 - 10folds   |          17.74849          |     18.06732      |            |
|   ver7 - 5folds   |          17.81729          |     18.04533      |            |
|  ver7 - SVR head  | 17.67098 - 1st cv svr head |     18.07251      |            |
|  ver 8 - 10folds  |          17.73651          | 18.02347 - 1st lb |            |
| ver 8 - SVR head  |          17.67373          |     18.07010      |            |
|  ver 9 - 10folds  |  17.69658 - 1st cv orgin   |     18.04895      |            |
| ver 9 - SVR head  |          17.68860          |     18.07064      |            |
| ver 10 - 10folds  |                            |                   |            |
| ver 10 - SVR head |                            |                   |            |



* Base model
  * Model: **'swin_base_patch4_window7_224'** in the timm library
  * Ensemble: 10-fold mean
  * Augmentation: base aug
  * Regression RMSE
  * Augmentation: Resize, HorizontalFlip, VerticalFlip, Normalize
  * Adam: lr=1e-5
  * head_out=128
  * CosineAnnealingWarmRestarts
  * Early stopping patience: 3
  * Epochs: 20
  * Batch size: 4
  * Image size: 224
  * 1 fold cv is  18.06551
  * mean of std oof / target:  0.55



* ver1 model
  * Model: **'swin_small_patch4_window7_224'** in the timm library
  * Augmentation: base aug
  * Ensemble: 10-fold mean
  * Regression RMSE
  * Augmentation: Resize, HorizontalFlip, VerticalFlip, Normalize
  * Adam: lr=1e-5
  * head_out=128
  * CosineAnnealingWarmRestarts
  * Early stopping patience: 5
  * Epochs: 20
  * Batch size: 4
  * Image size: 224
  * **final dense layer의 bias를 38.0으로 초기화**
  * 1 fold cv is  17.72005
  * mean of std oof / target:  0.53



* ver 2 model
  * Model: **'swin_base_patch4_window7_224_in22k'** in the timm library
  * **remove dup and average target**
  * Ensemble: 10-fold mean
  * Regression RMSE
  * Augmentation: base aug
  * Adam: lr=1e-5
  * head_out=192
  * lr scheduler: CosineAnnealingWarmRestarts
  * Early stopping patience: 3
  * Epochs: 10
  * Batch size: 4
  * Image size: 224
  * final dense layer의 bias를 38.0으로 초기화
  * 1 fold cv is  17.73151
  * mean of std oof / target:  0.56



* ver 3 model
  * ver 2 model cofig와 동일
  * 다른점
    * **'swin_large_patch4_window7_224_in22k'**
    *  all aug
    * final dense layer bias 0으로 초기화 
  * fold cv is  17.77913
  * mean of std oof / target:  0.57



* ver 4 model
  * ver1 model과 config 동일
    *  **'swin_small_patch4_window7_224'**
  * 다른점
    * **final dense layer bias 0으로 초기화** 
    * **remove dup and average target**
    * Early stopping patience: 3
    * head_out=192
  * 1 fold cv is  18.71113
  * mean of std oof / target:  0.55



* ver 5 model
  * ver 4 model과 다른점
    * **'swin_large_patch4_window7_224'**
    * **meta data 사용 X**
    * **From RMSE to BCE loss**
    * add RandomResizedCrop



* ver 6 model
  * ver 5 model에서 SVR head 추가
  * Overall CV NN head RSME = 17.79845490677977
  * Overall CV SVR head RSME = 18.12954052168302
  * Overall CV Ensemble heads RSME with 50% NN and 50% SVR = 17.7484913618123



* ver 7 model
  * ver 5 model과 다른점
    * swin_large_patch4_window12_384
    * 5fold



* ver 8 model
  * ver 5 model과 다른점
    * batch size=16



* ver 9 model

  * ver 8과 동일

  * 다른점

    * FCNN Head

      * ```python
        x1 = self.model(image)  # head_out = 192
        x = self.dense1(x1)		# (192, 64)  
        x = self.selu(x)
        x = self.dense2(x) 		# (64, 192) 
        x = x*0.7 + x1*0.3		# skip connection
        x = self.selu(x)
        x = self.dense3(x)		# (192, 32)
        x = self.relu(x)
        x = self.dense4(x)		# (32, 1)
        ```

        

* ver 10 model
  * ver 9와 동일
  * 다른점
    * train_remove_dup_pseudo_10folds_ver2 데이터 (paw data + oxford data) 사용
    * Adam lr = 3e-4, lr scheduler 3e-6



### Reference

* base model
  * https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/276522
  * https://www.kaggle.com/phalanx/train-swin-t-pytorch-lightning
  * https://www.kaggle.com/cdeotte/rapids-svr-boost-17-8
  * https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/277164
  * https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/274303
  * https://www.kaggle.com/abhishek/tez-pawpular-training
  * https://www.kaggle.com/abhishek/tez-pawpular-swin-ference

