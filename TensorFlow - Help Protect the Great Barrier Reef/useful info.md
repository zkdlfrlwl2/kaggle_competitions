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

* Reference
  * https://www.quora.com/What-is-the-F2-score-in-machine-learning
  * https://blog.acronym.co.kr/557







