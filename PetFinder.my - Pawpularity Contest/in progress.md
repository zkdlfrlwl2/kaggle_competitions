|        Model        |      CV      | Public LB |  Private LB  |
| :-----------------: | :----------: | :-------: | :----------: |
|   Base - 10 folds   |   18.35465   |   18.49   |      -       |
|   ver1 - 10folds    |   18.02539   | 18.10953  |      -       |
|   ver2 - 10folds    |   17.94588   | 18.30072  |      -       |
|   ver3 - 10folds    |   18.01586   | 18.24908  |      -       |
|   ver4 - 10folds    |   18.35333   | 18.40091  |      -       |
|   ver5 - 10folds    |   17.79845   | 18.03773  |   17.31849   |
|   ver6 - 10folds    |   17.74849   | 18.06732  |   17.24896   |
|    ver7 - 5folds    |   17.81729   | 18.04533  |   17.23596   |
| **ver7 - SVR head** | **17.67098** | 18.07251  | **17.12319** |
|   ver 8 - 10folds   |   17.73651   | 18.02347  |   17.30402   |
|  ver 8 - SVR head   |   17.67373   | 18.07010  |   17.26586   |
|   ver 9 - 10folds   |   17.69658   | 18.04895  |   17.28831   |
|  ver 9 - SVR head   |   17.68860   | 18.07064  |   17.28297   |
|  ver 10 - 10folds   |   19.08658   | 18.04916  |   17.43060   |
|  ver 11 - 10folds   |   22.41390   | 17.96100  |   17.25914   |
|  ver 11 - SVR head  |   19.36459   | 18.24712  |   17.42989   |
|  ver 12 - 10folds   |   10.25835   | 18.13238  |   17.35134   |
|  ver 13 - 10folds   |   18.50727   | 18.19801  |   17.46118   |



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
  * **final dense layer??? bias??? 38.0?????? ?????????**
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
  * final dense layer??? bias??? 38.0?????? ?????????
  * 1 fold cv is  17.73151
  * mean of std oof / target:  0.56



* ver 3 model
  * ver 2 model cofig??? ??????
  * ?????????
    * **'swin_large_patch4_window7_224_in22k'**
    *  all aug
    * final dense layer bias 0?????? ????????? 
  * fold cv is  17.77913
  * mean of std oof / target:  0.57



* ver 4 model
  * ver1 model??? config ??????
    *  **'swin_small_patch4_window7_224'**
  * ?????????
    * **final dense layer bias 0?????? ?????????** 
    * **remove dup and average target**
    * Early stopping patience: 3
    * head_out=192
  * 1 fold cv is  18.71113
  * mean of std oof / target:  0.55



* ver 5 model
  * ver 4 model??? ?????????
    * **'swin_large_patch4_window7_224'**
    * **meta data ?????? X**
    * **From RMSE to BCE loss**
    * add RandomResizedCrop



* ver 6 model
  * ver 5 model?????? SVR head ??????
  * Overall CV NN head RSME = 17.79845490677977
  * Overall CV SVR head RSME = 18.12954052168302
  * Overall CV Ensemble heads RSME with 50% NN and 50% SVR = 17.7484913618123



* ver 7 model
  * ver 5 model??? ?????????
    * swin_large_patch4_window12_384
    * 5fold



* ver 8 model
  * ver 5 model??? ?????????
    * batch size=16



* ver 9 model

  * ver 8??? ??????

  * ?????????

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
  * ver 9??? ??????
  * ?????????
    * train_remove_dup_pseudo_10folds_ver2 ????????? (paw data + oxford data) ??????



* ver 11 model
  * ver 9??? ??????
  * ?????????
    * train_remove_dup_pseudo_10folds ????????? (paw data + oxford data) ??????



* ver 12 model
  * ver 11??? ??????
  * ?????????
    * ????????? ?????? ???????????? ?????? ?????? ver9 model??? ???????????? ????????? ?????? ??? ??????



* ver 13 model
  * ver 10??? ??????
  * ?????????
    * ????????? ?????? ???????????? ?????? ?????? ver9 model??? ???????????? ????????? ??? ??????



### Reference

* base model
  * https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/276522
  * https://www.kaggle.com/phalanx/train-swin-t-pytorch-lightning
  * https://www.kaggle.com/cdeotte/rapids-svr-boost-17-8
  * https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/277164
  * https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/274303
  * https://www.kaggle.com/abhishek/tez-pawpular-training
  * https://www.kaggle.com/abhishek/tez-pawpular-swin-ference

