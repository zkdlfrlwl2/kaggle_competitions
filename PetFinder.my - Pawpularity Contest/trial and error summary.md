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
  * add rapids svr head - **完** 
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

* In regression
  * init 38.0, meta data = 0, RandomResizedCrop, one loss default target, batchsize=4, lr=1e-5
    * 1 fold cv: 18.00733, 2 fold cv: 18.36872
  * loss = target * 0.8 + target.diff.abs * 0.2, batch size=4, lr=1e-5
    * 1 fold cv is  18.62247, 2 fold cv is  18.35596
  * target을 하나 더 늘려서 시도해봤는데 별로였음
* In classification
  * BCE로 변경했을 때, bias를 38.0으로 초기화하는 건 별로 왜냐 범위가 0 ~ 1 이기 때문에
* SVR head 
  * ver 5 vs ver 6 (적용)
    * CV: 17.79845 vs **17.74849**
    * Public LB: **18.03773** vs 18.06732
  * SVR Head 적용 후 CV 약간 개선, LB는 오히려 증가
* Add other target
  * target을 따로 (4, 1), (4, 1)로 하는 것이 아니라 (4, 2)로 해서 시도
* 학습 데이터에 Dog or Cat labeling은 품이 너무 많이 듬 
  * Labeling 방법 고민 해봐야할 듯 
* Dog or Cat 무료 라이선스 데이터 구해서 Backborn 학습 시도



-------------------------------



### 3차 정리 - 12/6/2021

* model ver9를 사용하여 Oxford cat & dog 데이터로 paw target값 예측
* 기존 paw 이미지 데이터와 paw target값 예측한 Oxford 이미지 데이터를 합하여 model ver 9 config와 동일하게 처음부터 학습 진행 
  *  결과 (ver 11 model)
    * CV: 22.41390
    * LB: 17.96100
* oxford data의 paw target값은 np.random.choice로 기존 paw target값으로 채워넣어서 model ver9 config와 동일하게 처음부터 학습 진행
  * 결과 (ver 10 model)
    * CV: 19.08658
    * LB: 18.04916
