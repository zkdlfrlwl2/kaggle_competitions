## Kaggle Tips

### Google Brain - Ventilator Pressure Prediction

* Time line: September 22, 2021 ~ November 3, 2021

* Based on time series data, predict target pressure  

* **Chris Deotte Tips list-up**

  * https://www.kaggle.com/cdeotte

  * Chris Deotte의 notebook & discussion 정리한 내용

  * Ensemble Fold Models with **Median**

    * RSME 같이 square를 하면 error가 제곱이 되서 각 prediction error가 나빠지므로 prediction outlier를 final prediction 쪽으로 모우기 위해 **Mean**을 쓰지만 MAE는 제곱을 안하므로 **Median**이 더 낫다.
  
  * Batch size 줄이고 initial lr 값 조정, cosine learning schedule 사용
  
    * Batch size 512 is large and batch size 32 is small. I have noticed that sometimes large is better and sometimes small is better. We need to try both.
    * Batch size를 조정할 떄, 새로운 batch size에 대한 올바른 결론(제대로 동작하지 않는다는 착각을 피하기 위해서)을 얻기 위해 lr도 같이 조정해야한다.
    * 기본적으로 X만큼 batch size를 줄이면 lr도 X만큼 줄여야한다. 
      * lr를 완전 다르게 시도해봐도 된다
      * If the public bs is 512 with lr 5e-3 (we assume that the author found the best lr for bs 512), the first thing i do is reduce batch size to 32 with is 16x smaller. Therefore we try `5e-3 / 16 = 3.1e-4`. So, first experiment is bs 32 with lr 0.31e-4. Next try bs 32 with lr 2.5e-4 and lr 3.5e-4. To see if either is better than 3.1e-4
    * plateau scheduler로 얼마만큼의 epoch가 필요한지 찾는다
      * 찾은 epoch로 cosine with restarts를 돌려보고 성능이 괜찮은지 테스트 해본다
      * epoch를 10% 증가하거나 줄여서 테스트 해본다
      * warm up은 pretrained model 사용할 때 도움이 된다

  * Weighted median

  * 데이터도 많고 median을 사용하고 예측값의 분산도 높아서 많은 fold를 사용하면 성능이 향상됨

  * 다양한 모델 ensemble이 중요 

    * public notebook 중 제일 성능이 좋으면서 모델이 다른 것 2개 가져와서 최적화하고 ensemble 했으면 Top 50으로 마무리 했을 것

  * Top 50을 넘어서고 싶으면 개인 모델의 성능을 향상 시키고 **transformer**같은 다양한 모델을 생성

    * LSTM 전에 CNN을 넣어서 그 자신의 feature를 학습하게 만들 수도 있다

      * rearrange column order for cnn

    * pressure, pressure.diff(), pressure.cumsum() 같은 aux loss를 추가

      * I think this trick will work in any task where we are predicting a sequence.

    * u_out=0만 loss 계산

    * feature 추가 & 삭제 = FE

    * mixup, cutmix, pseudo labeling test data 같은 data augmentations

    * the transformer got stuck in some local minimum but with the restarts the loss boosted a lot.

    * 마지막으로 드롭아웃을 제거하고 CV 점수를 해치지 않고 하이퍼파라미터를 가능한 한 작게 조정했습니다. 임베딩 크기 및 피드포워드 차원 등을 작게 유지합니다.

    * First i optimized with just one cosine cycle. That was 125 epochs with 0 warm up epochs and start LR = 6e-4. Note that 0 warm up is often better for networks when we are not using pretrained start weights.

      After finding optimal single cycle. I just changed it to 3 cycles by using 125//2, 125, 125*2 and the same LR

  * batch size를 512에서 32로 줄이면 epoch 당 16배 많은 gradient updates가 일어나므로 epoch 수를 줄일 수 있다. batch size 512 + epoch 300++ == batch size 32 epoch 70 속도 유사. plateau보다 cosine schedule가 더 효율적이라 epoch 수를 더 적게 가져가도 된다.
  
    
  
    
  

