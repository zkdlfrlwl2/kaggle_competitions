## Trial and Error Summary

### 1차 정리 - 11/19/2021

* CV와 Public LB 차이가 많이 난다. 16.198 vs 18.10953 
  * CV 계산 방법이 잘못되었음
  * 기존 batch 별 계산하던 것을 fold 별로 변경
  * 한 모델 CV도 전 oof와 target 값으로 계산
* 유사한 이미지를 제거하고 target 값을 평균내서 학습을 돌리니 CV 값이 개선되었음
* final dense layer의 bias를 target 값 평균인 약 38.0으로 초기화하는 것도 CV 개선에 도움이 되었음
  * 수렴 속도가 빨라짐



* 시도 해볼 것 들 
  * add rapids svr head
  * add other augmentation - **完** 
    * RandomResizedCrop
  * from rmse loss to bce loss - **完** 
  * use GANs for additinal data & aux loss 
  * meta data 제외하고 학습 시켜보기 - **完** 
  * add cat or dog label to aux loss
  * 각 만드는 모델마다 oof도 같이 만들어서 실제 target 값과의 hist 및 평균값 비교해보기 - **完** 
    * https://www.kaggle.com/kishalmandal/eda-of-rapids-svr-actual-vs-pred-comparison



---------------------------

### 2차 정리 - 11/26/2021



