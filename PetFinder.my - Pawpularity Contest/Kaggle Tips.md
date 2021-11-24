#### Chris Deotte Tips list-up

* https://www.kaggle.com/cdeotte
* Chris Deotte의 notebook & discussion 정리한 내용
* When using NN with regression RSME, we may need to set dropout=0.





#### What may be the reasons  for the CV-LB gap?

* https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/284172
* Maybe the LB dataset(hope not the final dataset) has a really different distribution with the training set.
* CV 계산법이 잘못됐을 가능성 존재
  * RMSE의 경우, batch 별 계산 대신 fold 별 계산을 해야하고 전체 CV 계산도 fold 별 값을 평균내는 것이 아니고 전 oof를 대상으로 해야한다.
  * https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/289790
  * If you're using public notebook many notebooks compute CV incorrectly. Since the metric does `mean` then `square root`. We cannot compute RSME per batch and then average the batches. We must compute CV per fold on the entire fold. (And then computing the entire OOF on all train, we must compute again and not average the folds).



#### Why does swin transformer model perform well in this competition ?

* swin transformer does a better job at using global image features like pet's body posture, the background, etc (because it uses self attention on the entire photo). Whereas CNN focuses more on local features like faces or fur color (because it uses localized convolutions).
* If we were just classifying breed, we would only need local features from the face (or fur) and body posture wouldn't be that important. If we were just classifying breed, CNN would work well. But in this competition, the entire photo influences the click rate.



#### BCE Loss

* https://nuguziii.github.io/dev/dev-002/



### Regression

* Regression toward the mean
  * https://en.wikipedia.org/wiki/Regression_toward_the_mean
  * https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/289790
  * To summarize, if the pet photos had **no signal**, then our models' best guess would be a constant prediction of `38` which is the mean of the train targets (and the blue bars would all be at 38 and have **no spread**). If the pet photos contain **more signal**, then our models will make predictions with more variation (and the blue bars will begin to **spread out**).
  * Regardless of whether you use BCE loss or MSE loss, in the end we convert to continuous predictions between 0 and 100. So in the end we are computing a "regression line" between features and targets. So the theory of "regression to the mean" applies.
  * Better (more accurate) models will have more spread (blue bars will spread), but all teams will face a limit to maximum spread due to the limit of the signal present (versus randomness present).
  * None-the-less, the fact that the blue bars spread to `2/3` (i.e. r = 0.66) of the red bars is a good sign that there is plenty of signal present.
    * Correlation coefficient 1 = A perfect positive relationship
    * Correlation coefficient 0.8 = A fairly strong positive relationship.
    * Correlation coefficient 0.6 = A moderate positive relationship.
  * Furthermore your plot implies a correlation coefficient of approximately `r = 0.66` which means that `r^2 = 0.44`. In statistics, this `r^2` value says that `44%` of the [variance is explained](https://en.wikipedia.org/wiki/Explained_variation) by our models. So our models are explaining nearly one half of why the target value is different from photo to photo.
  * There is nothing special about this competition's regression problem. The point of my comment above is that **all** regression problems have this property. Because no regression model is perfect.
  * So everytime we do regression, if we plot histogram of ground truth and histogram of prediction together, we will see this. (And when we see this, the ratio of prediction spread divided by ground truth spread approximately indicates the correlation coefficient of our models).
* regression coefficient
  * https://support.minitab.com/en-us/minitab-express/1/help-and-how-to/modeling-statistics/regression/supporting-topics/regression-models/what-is-a-regression-coefficient/
  * https://statisticsbyjim.com/glossary/regression-coefficient/
* pearson correlation coefficient
  * https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
  * z score
* correlation coefficient
  * https://www.statisticshowto.com/probability-and-statistics/correlation-coefficient-formula/
  * the ratio of prediction spread divided by ground truth spread approximately indicates the correlation coefficient of our models
* Explained variation
  * https://en.wikipedia.org/wiki/Explained_variation
