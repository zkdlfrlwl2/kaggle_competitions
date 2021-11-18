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
      * remove dup and average target
        * 
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