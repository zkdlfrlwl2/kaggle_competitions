* target 값 평균이 약 38.034 이므로 마지막 dense layer의 bias를 38.0으로 초기화해서  수렴 속도를 빠르게 하고 hockey stick loss curve를 없앤다

  * ```python
    self.dense2 = nn.Linear(64, 1)
    self.dense2.bias.data = torch.nn.Parameter(
        torch.Tensor([38.0])
    )
    
    es patience=5
    ```
    
  * 'swin_base_patch4_window7_224'

    * 추가 전 1fold CV 16.28 (best val)
    * 추가 후 1fold CV 8.69 (last train) / 15.982 (best val) / pb 1fold 18.871
    
  * 'swin_large_patch4_window7_224'

    * 추가 후 1fold CV  7.4 (last train) / 16.184 (best val)

  * 'swin_small_patch4_window7_224'

    * 추가 후 1fold CV  9.52 (last train) / 16.082 (best val)

  * 'swin_tiny_patch4_window7_224'

    * 추가 후 1fold CV 11.3 (last train) / 16.76 (best val)
    
  * 'swin_base_patch4_window7_224_in22k'

    * 추가 후 1fold CV 16.2 (best val)

  * 'swin_base_patch4_window12_384'

    * 추가 후 1fold CV 16.007 (best val), 1h 16m

  * 'swin_base_patch4_window12_384_in22k'

    * 추가 후 1fold CV 15.867 (best val)**, 1h 32m**, 18.31311

  * 'swin_large_patch4_window12_384'

    * 추가 후 1fold **CV 15.765 (best val)**, 2h 4m, 18.44981

  * 'swin_large_patch4_window12_384_in22k'

    * 추가 후 1fold CV 15.896 (best val), 2h 4m

* meta data 유무

  * 적용 1fold CV 
    * 8.69 (last train) / 15.982 (best val) / pb 1fold 18.871
  * 미적용 1fold CV
    * 16.264 (best val)
  * transformer 입력에 넣기 전, image와 feature concat 1fold CV
    * ![image](https://user-images.githubusercontent.com/92927837/141882804-d3a398d2-9371-4b83-b2b0-593a6ce7a1ac.png)
    * 보류 
      * RuntimeError: Sizes of tensors must match except in dimension 0. Expected size 3 but got size 1 for tensor number 1 in the list. -> meta도 3차원이어야하나본데

* try augmentation

  * base - HorizontalFlip, VerticalFlip
    * 'swin_base_patch4_window7_224'
    * 1fold CV: 16.094 (best val)
  * base + RandomBrightnessContrast
    * 'swin_base_patch4_window7_224'
    * 1fold CV: 16.254 (best val)
  * base + HueSaturationValue
    * 'swin_base_patch4_window7_224'
    * 1 fold CV: 16.119 (best val)
  * All add aug
    * 'swin_large_patch4_window12_384_in22k'
    * 1 fold of 10 folds CV: **15.869 (best val) / 18.16462 (pb score)** 
      * CosineAnnealingLR 1fold CV: 15.843
    * 1 fold of 5 folds CV:  15.917 (best val) 

* lr scheduler 변경 시도

  * model: 'swin_large_patch4_window12_384'

  * ```python
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6)
    ```

  * CosineAnnealingWarmRestarts - headout 256

    * T_0=10, val cv 16.145
    * T_0=5, val cv 16.308

  * CosineAnnealingLR - headout 192

    * T_max=10, 15.886
    * T_max=5, 15.987

* swin transformer output unit size 변경 시도

  * 128 -> 192
    * 384 model 사용 시, 192로 변경해서 사용 중

  * 192 -> 256

  

* train dataset과 test dataset의 분포가 다르다면 어떻게 학습시켜야 할까

* **add rapids svr head**

* add metadata input layer 

* adjust epoch, batch size, lr, folds, lr scheduler 

* **add other augmentation** 

* **use GANs for additinal data & aux loss**

  * GAN으로 생성한 이미지를 분류하게 CNN attention으로 

* 지금 현재 training dataset에 강아지냐 고양이냐의 label을 하나 더 추가해서 **aux loss로 사용**한다면 ?
  * test set이 노이즈라서 test set까지 labeling하기는 힘들고 training phase때만 aux loss로 학습하고 infer에는 aux head 없어도 상관없나 ?  함 시도해봐야겠다

* from rmse loss to bce loss 



* 유사한 이미지가 있으나 target 값이 다르다
  * https://www.kaggle.com/showeed/annoy-similar-images-ver2?scriptVersionId=79322129
  * 유사한 이미지 중 하나를 없애고 target값을 두 값의 평균으로 치환해서 다시 학습을 해본다면 ?
  * 여러 discussion에서 지적했듯이 target 값인 pawpularity에 대해서 불만이 많다.
    * 동일한 이미지에 target값이 다른 이유를 뒷받침 해줄 데이터가 부족하다.
    * 동일한 프로필 사진이라도 올린 시간이 달라서 target값이 달라졌을지도 모르는데 그런 부가적인 정보가 되게 부족하다. 업로드 시간, 품종, 나이, 성별 등등
    * 그나마 위안이 되는 건 데이터가 아주 적지는 않다는 것
  * 'swin_base_patch4_window7_224_in22k' -> 10 fold 대략 8시간 소요 예상
    * data aug all 1fold CV 15.903, 47m 35s
    * data aug base 1fold CV 15.668, 41m 5s, public lb 18.43586
    * base는 aug가 적은 편이 CV가 잘 나오네
  * 'swin_large_patch4_window7_224_in22k'
    * data aug all 1fold CV 15.744, 1h 5m 27s -> 10 fold 대략 11시간 소요 예상 
    * data aug base 1fold CV 15.905, 50m 50s
    * large는 aug가 많은 편이 CV가 잘 나오고



---------------------------



### 1차 정리 - 11/19/2021

* CV와 Public LB 차이가 많이 난다. 16.198 vs 18.10953 

  * train dataset 분포와 test dataset 분포가 다른걸까 
* Batch size는 작을 수록 CV가 더 잘나왔다. 

  * Batch size 4 고정
* lr

  * start 1e-5, min 1e-6
  * CosineAnnealingWarmRestarts
* 유사한 이미지를 제거하고 target 값을 평균내서 학습을 돌리니 CV 값이 평균적으로 0.2 ~ 0.3 개선되었음.
* 현 시점에서 swin large patch4 window12 384 in 22k model이 가장 CV가 잘 나오지만 1 epoch당 2h 4m이 걸려 10 folds를 다 돌리면 21시간이 걸린다. 근데 Google Colab Pro는 최대 24시간 런타임 이라 되어있지만 실제로 돌려보면 10시간도 안되서 런타임이 끊긴다. 여러 시도를 해보고 정 안되겠다 싶으면 model을 바꾸는 수밖에 

  * swin base or large patch4 window 7 224 in 22k 
* final dense layer의 bias를 target 값 평균인 약 38.0으로 초기화하는 것도 CV 개선에 도움이 되었음
* 'swin_patch4_window7_224_in22k'

  * base vs large
    * large는 data aug가 많을 수록 CV가 잘 나왔고 반면에 base는 aug가 거의 없을 때, CV가 잘 나옴
      * 이유는 모르겠고, window12_384도 마찬가지일까 ? 
    * 384는 aug all이 더 CV 잘나오는 듯
* CV vs LB correlation
  * config
    * remove dup and average target
    * final dense layer의 bias를 38.0으로 초기화
    * Batch size: 4
    * Epochs: 10(224), 5(384)
    * Early stopping patience: 3

  * 'swin_base_patch4_window7_224_in22k'

    * data aug base 1fold CV **15.668,  public lb 18.43586**, 41m 5s
    * data aug all 1fold CV **15.903**, 47m 35s
  * 'swin_large_patch4_window7_224_in22k'
    * data aug base 1fold CV **15.905**, 50m 50s
    * data aug all 1fold CV **15.744**, 1h 5m 27s
  * 'swin_base_patch4_window12_384_in22k'
    * data aug base 1fold CV **15.758**, 1h 16m 6s
    * data aug all 1fold CV **15.742**, 1h 18m 50s
  * 'swin_large_patch4_window12_384_in22k'
    * data aug base 1fold CV 15.551, 2h 4m
    * data aug all 1fold CV **15.439, public lb 18.54264**, 2h 4m
  * 15.668 - 18.436 vs 15.869 - 18.165
    * CV랑 LB 점수랑 비례하지가 않는다.
    * batch size가 지나치게 작아서 그런걸까 
    * RMSE 대신 BCE로 Loss를 변경하면 비례하게 바뀔까 ? 
    * meta data와 target 간의 상관관계가 없어서 더 그럴까 ? 

      * meta data를 제외하고 학습 시켜서 cv와 pb의 상관 관계를 살펴봐야하나 ?
    * train과 test dataset 간의 분포가 달라서 그럴까 ? 
    * training dataset인 이미지와 target 값 간의 연결 고리가 없어서 학습이 제대로 이루어지지 않는건가 

      * 단순 평균 맞추기 -> 운의 요소가 영향을 끼치는 실패한 competition 인가 ?
    * data augmentaion 차이인가 ? 
      * aug 적은 것 대비 aug가 많으면 CV 대비 PB 감소가 두드러지는 건가 ? 


​       

* **시도해볼것들**
  * **bias 38 추가하는거 없애자. test dataset target 평균값이 38이라는 보장이 없으니**
  * lr scheduler 제거 -> lr 1e-5 고정 
  * swin 384 model은 5fold로, 224 model은 10fold로
  * add rapids svr head - 4
  * add other augmentation
  * from rmse loss to bce loss - 2
  * use GANs for additinal data & aux loss
  * 강아지냐 고양이냐의 label을 하나 더 추가해서 aux loss로 사용 - 5
  * head에 attention layer 추가 해보기 - 3
  * meta data 제외하고 학습 시켜보기 - 1
  * 각 만드는 모델마다 oof도 같이 만들어서 실제 target 값과의 hist 및 평균값 비교해보기
    * https://www.kaggle.com/kishalmandal/eda-of-rapids-svr-actual-vs-pred-comparison



---------------------------------



#### 2021 - 11 - 22 ~ 26

* ver 4 model에서 meta data 제외하고 1fold만 학습 시켜서 제출해보기

  * 1fold CV 16.147 / Public LB 18.95612

  * preds paw about 12 ~ 14 - 노이즈 이미지에 대한 예측값

    * 이건 아마도 1fold에 속해 있는 training dataset의 값이 저정도라는 소리겠지.

      ![image](https://user-images.githubusercontent.com/92927837/142787579-6f3cbdbe-bcf5-4341-8307-728208a08c87.png)

      * **확인해보니 12 ~ 14가 아니고 평균은 38 정도임**
      * 앞으로 드러난 test dataset이 단순 noise 이미지라 유추할 수가 없네.

* 각 모델 별 oof vs fold target 분포 비교 해보기

  * https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/289790

  * 20 ~ 80 값만 학습되는 것 같음

  * 0 ~ 20, 80 ~ 100 target 값은 전혀 예측을 못함 

    ![image](https://i.ibb.co/H2pv0rS/142796486-a9425053-8212-446b-8dbd-1da31d9d4ae5.png)

    * 중앙으로 압축된 느낌

* 주최자 측에서 제안한 loss & metric이라 하더라도 cv와 lb의 상관관계가 안보이면 다른 loss & metric을 찾아서라도 상관관계를 linear하게 맞추는게 맞나 ? 

  * maybe yes

* 모든 회귀는 평균으로 돌아가려는 습성이 있다 ? 

  * oof std와 target std를 나눠서 0.6 이상이면 그래도 training data에 useful signal이 있다는 소리
  * 모든 회귀 모델이 완벽하지 않아서 생기는 문제 ? 

* bias를 38.0으로 초기화하는 건 단순히 수렴 속도를 빠르게 해주는 것 이외에는 없는 건가 ..

  * overfit과는 상관없는건가 ..
  * 이번 대회에서 사용해보고 최종 결산 때 overfit이 심하면 다음에는 사용 안하는 걸로 

* **중요**

  * CV 계산 시, batch 별 -> fold 별로 변경 - **完** 
  * tez -> only pytorch 코드로 변경
    * Loss 계산법 고려 필요 
    * BCE Loss 적용 
  * CV 계산법 변경 후 batch size 조절 필요, 현재 4 - RAM 용량 제한 하에서 - 
  * 모델 평가 시, 1fold에서 나아가 추가로 2 fold 결과 체크 필요 - **完** 
  * regression coefficient 란
  * pearson correlation coefficient 란
  * correlation coefficient 란
  * z score 란

* init bias and original meta data 비교

  * model: swin_base_patch4_window7_224_in22k
  * init 38.0, meta data = 0
    * 1 fold cv: 18.47227, 2 fold cv: 18.69408
  * init 0.0, meta data = 0
  * init 38.0, meta data = 12
  * init 0.0, meta data = 12

* 학습 데이터에 target 추가

  * dog or cat label

