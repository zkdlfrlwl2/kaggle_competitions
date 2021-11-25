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
    * BCE Loss 적용 - **完** 
    
  * CV 계산법 변경 후 batch size 조절 필요, 현재 4 - RAM 용량 제한 하에서 - 

  * 모델 평가 시, 1fold에서 나아가 추가로 2 fold 결과 체크 필요 - **完** 

    

* init bias and original meta data 비교 - Google Colab Pro

  * model: swin_base_patch4_window7_224_in22k
    * init 38.0, meta data = 0
      * 1 fold cv: 18.35615, 2 fold cv: 18.93261
    * init 0.0, meta data = 0
      * 1 fold cv: 18.99869, 2 fold cv: 18.91864
    * init 38.0, meta data = 12
      * 1 fold cv: 19.07274, 2 fold cv: 18.45164
    * init 0.0, meta data = 12
      * 1 fold cv: 18.59144, 2 fold cv: 19.15761
    * meta data가 target하고 상관관계가 없어서 그런지 완전히 noise 취급인가보네 
    * **init 38.0, meta data = 0, RandomResizedCrop**, one loss default target, batchsize=4, lr=1e-5
      * 1 fold cv: 18.00733, 2 fold cv: 18.36872
    * init 0.0, meta data = 0, RandomResizedCrop, target.diff().abs()
      * loss = target * 0.6 + target.diff.abs * 0.4
        * 1 fold cv is  19.20116, 2 fold cv is  19.51235
      * loss = target * 0.8 + target.diff.abs * 0.2, batch size=4, lr=1e-5
        * 1 fold cv is  18.62247, 2 fold cv is  18.35596
      * loss = target * 0.8 + target.diff.abs * 0.2, batch size=8, lr=2e-5
        * 1 fold cv is  18.57108, 2 fold cv is  18.43565
      * loss = target * 0.8 + target.diff.abs * 0.2, batch size=16, lr=4e-5
        * 1 fold cv is  18.51706, 2 fold cv is  18.81915
      * loss = target * 0.5 + target.diff.abs * 0.5, batch size=4, lr=1e-5, out=(x1+x2)/2.0
        * 1 fold cv is  19.56327, 2 fold cv is  19.48021
      * loss = target * 0.8 + target.diff.abs * 0.2, batch size=4, lr=1e-5, out=(x1 0.8 + x2 0. 2)
        * swin_base_patch4_window7_224
          * 1 fold cv is  18.91719, 2 fold cv is  18.1463
        * swin_large_patch4_window7_224
          * 3
    * init bias mean target value 38.0, mean target.diff.abs value xx.x
      * meta data=0, batchsize=4, lr=1e-5
        * loss = target * 0.8 + target.diff.abs * 0.2, batch size=4, lr=1e-5, out=(x1 0.8 + x2 0. 2)



* From MSE to BCE - Kaggle Notebook

  * 수정한 부분

    * ```python
      def __getitem__(self, item): # From class PawpularDataset
              image = cv2.imread(self.image_paths[item])
              image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
              
              if self.augmentations is not None:
                  augmented = self.augmentations(image=image)
                  image = augmented["image"]
              
              # (720, 405, 3) -> (3, 720, 405)
              image = np.transpose(image, (2, 0, 1)).astype(np.float32)
              
              features = self.dense_features[item, :]
              targets = self.targets[item] / 100.0 # 나누기 100 추가
              
              return {
                  "image": torch.tensor(image, dtype=torch.float),
                  "features": torch.tensor(features, dtype=torch.float),
                  "targets": torch.tensor(targets, dtype=torch.float),
              }
      ```

    * ```python
      if targets is not None: # From class PawpularModel(tez.Model)
                  # RMSE
                  # loss = nn.MSELoss()(x, targets.view(-1, 1))
                  # metrics = self.monitor_metrics(x, targets)
                  
                  # BCE loss 추가
                  loss = nn.BCEWithLogitsLoss()(x, targets.view(-1, 1))
                  metrics = self.monitor_metrics(torch.sigmoid(x) * 100, targets * 100)
                  return x, loss, metrics
      ```

  * model: swin_base_patch4_window7_224_in22k

    * init 38.0, meta data = 0, batch size=4, lr=1e-5, es 3
      * 1 fold cv is  21.97881, 2 fold cv is  19.18696, 3 fold cv is  18.90972
      * **BCE를 사용하는 이상 bias를 38.0으로 초기화하는 건 의미 없네**
    * init 0.0, meta data = 0, batch size=4, lr=1e-5, es 3
      * 1 fold cv is  18.74303, 2 fold cv is  19.27812
    * init 0.0, meta data = 0, RandomResizedCrop, batch size=4, lr=1e-5, es 3
      * 1 fold cv is  18.31289, 2 fold cv is  18.25749
    * init 0.0, meta data = 0, RandomResizedCrop, batch size=8, lr=1e-5, es 3
      * 1 fold cv is  18.073, 2 fold cv is  18.61555
    * init 0.0, meta data = 0, RandomResizedCrop, batch size=8, lr=2e-5, es 3
      * 1 fold cv is  18.63801, 2 fold cv is  18.69654
    * init 0.0, meta data = 0, RandomResizedCrop, batch size=16, lr=4e-5, es 5
      * 1 fold cv is  18.73122, 2 fold cv is  18.34747
  
  * model: swin_base_patch4_window7_224
  
    * init 0.0, meta data = 0, RandomResizedCrop, batch size=4, lr=1e-5, es 3
      * 1 fold cv is  18.46248, 2 fold cv is  18.41336
      * loss = target * 0.8 + target.diff.abs * 0.2, batch size=4, lr=1e-5, out=(x1 0.8 + x2 0.2)
        * 2
  
  * model: swin_large_patch4_window7_224
  
    * init 0.0, meta data = 0, RandomResizedCrop, batch size=4, lr=1e-5, es 3
      * 1 fold cv is  18.19514
  
  * model: swin_large_patch4_window7_224_in22k
  
    * init 0.0, meta data = 0, RandomResizedCrop, batch size=4, lr=1e-5, es 3
      * 1 fold cv is  18.19549, 2 fold cv is  17.96722



* albumentations RandomResizedCrop
  * Kaggle notebook에서는 문제 없음
    * version: 1.1.0
  * Google Colab notebook에서 AttributeError 발생
    * version: 0.1.12
    * 1.1.0 버전 설치해서 해결
    * !pip install albumentations==1.1.0



* 학습 데이터에 target 추가

  * add dog or cat label
  * add aux loss 



* 무료 라이선스로 공개된 강아지 & 고양이 품종 데이터셋을 구해서 backborn을 학습 후 사용 or GAN 사용
  * https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/278364

