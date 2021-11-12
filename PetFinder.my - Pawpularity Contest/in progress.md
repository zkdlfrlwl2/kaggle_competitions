| Model |  CV  | Public LB | Private LB |
| :---: | :--: | :-------: | :--------: |
| Base  |      |           |            |

한 달에 Model 1개 목표로 진행 

* Base model
  * Model: 'swin_base_patch4_window7_224' in the timm library
  * Ensemble: 10-fold mean
  * Regression
  * Augmentation: Resize, HorizontalFlip, VerticalFlip, Normalize
  * Adam
  * CosineAnnealingWarmRestarts
  * Epochs: 20
  * Batch size: 4
  * Image size: 224
  * 1fold valid rmse: 16.35





* try something
    * add rapids svr head
    * add metadata input layer
    * try other lr scheduler
    * adjust epoch, batch size, lr
    * add other augmentation
    * use pet-centric cropped dataset 
    * use GANs for additinal data





### Reference

* base model
  * https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/276522
  * https://www.kaggle.com/phalanx/train-swin-t-pytorch-lightning
  * https://www.kaggle.com/cdeotte/rapids-svr-boost-17-8
  * https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/277164
  * https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/274303
  * https://www.kaggle.com/abhishek/tez-pawpular-training
  * https://www.kaggle.com/abhishek/tez-pawpular-swin-ference

