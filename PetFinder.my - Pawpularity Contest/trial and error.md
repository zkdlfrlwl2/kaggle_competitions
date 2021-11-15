* target 값 평균이 약 38.034 이므로 마지막 dense layer의 bias를 38.0으로 초기화해서  수렴 속도를 빠르게 하고 hockey stick loss curve를 없앤다

  * ```python
    self.dense2 = nn.Linear(64, 1)
    self.dense2.bias.data = torch.nn.Parameter(
        torch.Tensor([38.0])
    )
    ```

  * 'swin_base_patch4_window7_224'

    * 추가 전 1fold CV 16.28 (best val)
    * 추가 후 1fold CV 16.17x (best val)
    
  * 'swin_large_patch4_window7_224'
  
    * 추가 후 1fold CV  7.4 (last train) / 16.184 (best val)
  
  * 'swin_small_patch4_window7_224'
  
    * 추가 후 1fold CV  9.52 (last train) / 16.082 (best val)
  
  * 'swin_tiny_patch4_window7_224'
  
    * 추가 후 1fold CV 11.3 (last train) / 16.76 (best val)
  
* meta data 유무

  * 적용 1fold CV
    * Conv1d 적용 1fold CV
    * Conv1d 미적용 1fold CV

  * 미적용 1fold CV


* add rapids svr head - 5 
* add metadata input layer - 1 
* adjust epoch, batch size, lr, folds, lr scheduler - 3 
* add other augmentation - 2 
* use GANs for additinal data & aux loss - 6
* from rmse loss to bce loss - 4