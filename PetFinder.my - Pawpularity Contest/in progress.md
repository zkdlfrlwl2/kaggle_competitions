|      Model      |   CV   | Public LB | Private LB |
| :-------------: | :----: | :-------: | :--------: |
| Base - 10 folds | 16.49  |   18.49   |            |
| ver1 - 10folds  | 16.198 | 18.10953  |            |
| ver2 - 10folds  |        |           |            |

한 달에 Model 1개 목표로 진행 

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
  * 1fold result
    * base valid rmse & batch size 4 -> 16.35 / 18.57
    * add selu after dense1 layer & batch size 4 -> 16.99
    * remove dropout & batch size 4 -> 16.28 / 18.58 = **select**
    * remove dropout & batch size 8 -> 17.65
    * remove dropout & batch size 16 -> 18.01
    * remove dropout & swin large patch4 window7 224 & batch size 4 -> 16.50



* ver1 model
  * Model: **'swin_small_patch4_window7_224'** in the timm library
  * Augmentation: base aug
  * Ensemble: 10-fold mean
  * Regression RMSE
  * Augmentation: Resize, HorizontalFlip, VerticalFlip, Normalize
  * Adam: lr=1e-5
  * head_out=192
  * CosineAnnealingWarmRestarts
  * Early stopping patience: 5
  * Epochs: 20
  * Batch size: 4
  * Image size: 224
  * **final dense layer의 bias를 38.0으로 초기화**



* ver 2 model
  * Model: **'swin_large_patch4_window12_384_in22k'** in the timm library
  * **remove dup and average target**
  * Ensemble: 10-fold mean
  * Regression RMSE
  * Augmentation: **All add aug**
  * Adam: lr=1e-5
  * head_out=192
  * lr scheduler: CosineAnnealingWarmRestarts
  * Early stopping patience: 3
  * Epochs: 5
  * Batch size: 4
  * Image size: 384
  * final dense layer의 bias를 38.0으로 초기화









### Reference

* base model
  * https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/276522
  * https://www.kaggle.com/phalanx/train-swin-t-pytorch-lightning
  * https://www.kaggle.com/cdeotte/rapids-svr-boost-17-8
  * https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/277164
  * https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/274303
  * https://www.kaggle.com/abhishek/tez-pawpular-training
  * https://www.kaggle.com/abhishek/tez-pawpular-swin-ference

