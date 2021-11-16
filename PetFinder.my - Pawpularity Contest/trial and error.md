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
  
    * 추가 후 1fold CV
    * 1fold 소요 시간
  
  * 'swin_large_patch4_window7_224_in22k'
  
    * 추가 후 1fold CV
    * 1fold 소요 시간
  
  * 'swin_base_patch4_window12_384'
  
    * 추가 후 1fold CV
    * 1fold 소요 시간
  
  * 'swin_large_patch4_window12_384'
  
    * 추가 후 1fold CV
    * 1fold 소요 시간
  
  * 'swin_base_patch4_window12_384_in22k'
  
    * 추가 후 1fold CV
    * 1fold 소요 시간
  
  * 'swin_large_patch4_window12_384_in22k'
  
    * 추가 후 1fold CV
    * 1fold 소요 시간
  
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
    * 1 fold CV: 15.869 (best val) / 18.16462 (pb score) 
  
* fold split 방식을 바꿔보자

  * kfold
  * stkfold



* add rapids svr head - 5 
* add metadata input layer - 1 
* adjust epoch, batch size, lr, folds, lr scheduler - 3 
* add other augmentation - 2 
* use GANs for additinal data & aux loss - 6
* from rmse loss to bce loss - 4