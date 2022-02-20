## Question about CV

### Cross Validation

* Train N Fold models and ensemble (i.e. average) the N test predictions
  * let's say you have 5-Fold models. Then you split the train data into 5 folds and train 5 models. Each model trains with 80% data and infers the other 20% data. When we combine all the 5x 20% train predictions, we have 1 prediction for each train row (i.e. movie frame). We compute our CV score on this (called OOF). During inference each of the 5 fold models predicts the test data. Then we have 5 predictions for each test row (movie frame). We ensemble these with WBFT.
* After discovering the best hyper-parameters from CV, train 1 model using 100% data and infer test with 1 model
  * we start with 5-fold models. We learn that learning rate 0.01 is best with batch size 8 and epochs 7. Afterward we ignore our 5-fold models. Now we train another model (a 6th model), that uses 100% train data. We train it with learning rate 0.01, batch size 8, and 7 epochs. Afterward we use this 1 model to predict the test data. We have 1 prediction for each test frame. This is our submission.
* Extra
  * A third approach is to use "approach one", and then just infer 3 of the 5 fold models because of time limitation or something. Then we have 3 predictions for each test frame and use WBFT. And a fourth approach is to train multiple models with 100% train and then average them
* QnA
  * Would you agree that using multiple folds during inference should be more robust than the model trained on 100% data with parameters of 5-fold models ?
    * Using 100% data usually works great and boosts LB in all Kaggle competitions. From the K-Fold CV, we know what epoch to stop at (and what hyperparameters to use), so it won't overfit because CV didn't overfit.
    * But, sometimes i don't use 100% **when the local fold validation score fluctuate a lot from epoch to epoch**. For example in this comp and CommonLit comp, my local validation score changes a lot from epoch to epoch. In this case, i don't trust using 100% (because i don't know if the stopping epoch will be the right epoch).
    * Instead I pick a large fold number like 10 or 20. Then you just train 5 folds. Each fold will be trained on 90% or 95% (respectively) of train data. (Note you can use 20-Fold, but only train 5 of the 20 folds and submit 5 folds to LB. This is what our team did in CommonLit comp).
    * Also note when you train with 100% data, you can still do this 5 times and submit an ensemble of 5 models. The LB score for NN's will usually also increase by training the same NN repeatedly and using different seeds to initialize the layer weights. Also training involves random augmentations and random batches which guarantee that two trained NN will never be the same.
  * How do you choose the best splitting strategy ? Do you use a fixed baseline model, use it to compute the OOF CV score over different splitting strategies (by video, creating sequences, etc) and choose the one that maximizes this CV score ?
    * We pick a strategy that mimics the relationship between train and test data. If train and test are just random subsets of some larger population then random K-Fold works. However, if there is a difference between train and test, like for example test uses different videos than train, then we should set up our CV to mimic this. We should put different videos in different CV folds.
    * Another example might be predicting customer credit scores. If train data and test data have different customers, then we want to keep all rows pertaining(유지, 참조) to one customer within a single fold. To keep things in one fold we use group K Fold.
    * If the positive target is rare like 1% (or we have multi label and/or multi class and there are rare targets <1%), then we should use stratified K Fold to make sure that each validation fold includes some positive targets.
    * After choosing our CV, the final test is to submit to LB. If our CV score is approximately equal to our LB score. And if our LB scores goes up when our CV score goes up, then we did a good job. Note that sometimes, the test data has some mystery so the best we can do is have LB go up and down when CV goes up and down.





### Image Problem

* Image Classification
  * 사진을 특정 알고리즘을 이용하여 분류하는 것
* Image Classification with localization
  * Image classification과 bounding box를 통하여 object를 표시하는 것
  * 특정 이미지에서 대상을 분류하고 위치도 표시
* Object detection
  * 위 Classification과 Localization은 1개의 Object를 대상으로 하는 알고리즘
  * 반면 Detection은 1개 이상의 Object를 Classification 및 Localization 하는 것



### F2 Score

* In the analysis of binary classification, the **F-score measures the accuracy of a test using precision and recall**. **Precision is the ratio of true positives (tp) to all predicted positives (tp + fp)**. **Recall is the ratio of true positives to all actual positives (tp + fn)**.

  The general formula for the F-score is the following:
  $$
  F_\beta = (1 + \beta^2)\cdot {precision\cdot recall \over (\beta^2 \cdot precision) + recall}
  $$
  where β is a positive real. **For the F2 score, you just set β equal to 2**. The intuition behind the F2 score is that it weights recall higher than precision. This makes the F2 score more suitable in certain applications where it’s more important to classify correctly as many positive samples as possible, rather than maximizing the number of correct classifications.

* F-Measure는 Precision과 Recall의 trade-off를 잘 통합하여 정확성을 한 번에 나타내는 지표이며 보통 가중치를 가진 조화 평균 (weighted harmonic mean)이라고도 한다.

  * 조화 평균

    * 일반적으로 계산하는 평균은 산술 평균 (Arithmetic Mean) 이라 하며 조화 평균 (Harmonic Mean)은 주어진 수들의 역수의 산술 평균을 구한 값의 역수를 말한다.
      $$
      H = {n \over { 1 \over a_1 } + { 1 \over a_2 } + \cdots { 1 \over a_n }}
      $$

    * 두 수 a1, a2 사이의 조화 평균은 다음과 같다
      $$
      H = { 2a_1a_2 \over a_1 + a_2}
      $$

  * F-Measure 수식

    * F-Measure를 구하기 위해 Precision과 Recall에 대한 조화 평균에 가중치 알파를 적용하면 다음과 같다
      $$
      F = {1 \over \alpha {1 \over P} + (1-\alpha){1\over R}} = {(\beta^2+1)PR \over \beta^2P+R}
      $$

    * 여기에서 Precision과 Recall에 적용한 가중치를 0.5로 동일하게 부여하면 우측 식의 베타 값은 1이 된다. 이것을 F1 Measure라 하고 자주 사용하는 F-Measure 값이기도 하다. 아래 수식을 보면 앞의 두 수 사이의 조화 평균과 동일한 것을 알 수 있다.
      $$
      F_1 = 2 \cdot {precision \cdot recall \over precision + recall}
      $$

  * F-Measure 활용

    * F1 Measure는 Precision과 Recall의 중요성을 동일하게 보고 있다. 만약 Recall이 Precision보다 더 중요하다면 F2 Measure를, Precision이 Recall 보다 더 중요하다면 F0.5 Measure를 사용할 수 있다. 여기서 2나 0.5는 베타 값이다.

  

  

  ## Reference

  * https://www.quora.com/What-is-the-F2-score-in-machine-learning
  * https://blog.acronym.co.kr/557
  * https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/308027







