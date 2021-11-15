* target 값 평균이 약 38.034 이므로 마지막 dense layer의 bias를 38.0으로 초기화해서  수렴 속도를 빠르게 하고 hockey stick loss curve를 없앤다

  * ```python
    self.dense2 = nn.Linear(64, 1)
    self.dense2.bias.data = torch.nn.Parameter(
        torch.Tensor([38.0])
    )
    ```

  * 'swin_base_patch4_window7_224'

    * 추가 전 1fold CV 16.28 (val)
    * 추가 후 1fold CV 16.18 (val)
    
  * 'swin_large_patch4_window7_224'
  
    * 추가 후 1fold CV 
  
  * 'swin_small_patch4_window7_224'
  
    * 추가 후 1fold CV
  
  * 'swin_tiny_patch4_window7_224'
  
    * 추가 후 1fold CV
  
* meta data 유무

  * 적용 1fold CV
  * 미적용 1fold CV


* add rapids svr head
* add metadata input layer
* try other lr scheduler
* adjust epoch, batch size, lr, folds
* add other augmentation
* use pet-centric cropped dataset 
* use GANs for additinal data
* from rmse loss to bce loss