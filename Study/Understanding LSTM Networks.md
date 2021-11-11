## Understanding LSTM Networks

### Recurrent Neural Networks

* 전통적인 인공 신경망은 정보의 지속성이 없으나 반면에 RNN은 자신 내부에 정보를 유지시키는 순환을 만들어 이 결점을 해결했다.

  ![image-20211104094023772](C:\Users\swlee\AppData\Roaming\Typora\typora-user-images\image-20211104094023772.png)

* loop는 신경망의 한 단계에서 다음 단계로 정보를 전달 해준다. loop를 펼쳐보면 RNN을 동일한 신경망의 여러 복사본으로 생각해볼 수 있다.

  ![image-20211104095152040](C:\Users\swlee\AppData\Roaming\Typora\typora-user-images\image-20211104095152040.png)

* 이러한 체인과 같은 특성은 순환 신경망이 sequences 및 lists와 밀접하게 관련되어 있음을 보여줍니다. 이러한 데이터에 사용할 자연스러운 아키텍처입니다.



### The Problem of Long-Term Dependencies

* RNN의 경우, 다음 단어를 예측할 때 참고해야하는 문맥이 비교적 최근이라면 문제가 없지만 먼 과거의 문맥을 참조해야한다면, 그 격차가 더 커진다면 RNN은 정보의 연결을 학습할 수 없다. LSTM은 이런 문제가 없다.
* 왜 RNN만 long-term dependencies 문제가 생길까 ?
  * **관련된 요소가 멀리 떨어져 있는 경우** 시퀀스에 **장기 의존성**이 존재
  * tanh가 여러번 곱해져서 (-1 ~ 1) 즉, 1보다 작은 값이 반복적으로 곱해지기 때문에, **feed-forward 관점에서는 뒷단으로 갈 수록 앞의 정보를 충분히 전달할 수 없고 back-prop의 관점에서는 tanh 함수의 기울기가 0에 가깝거나 굉장히 큰 값이 발생할 수 있어 기울기 소실 (학습이 더이상 진행되지 않는다)혹은 폭발(loss로 NAN값이 나옴)의 문제**를 일으킵니다.





### LSTM Networks

* LSTM은 long-term dependencies를 학습할 수 있는 역량을 가진 특별한 형태의 RNN이다. 모든 RNN은 신경망 모듈이 반복되는 체인 형태를 가지고 있다. 기본적인 RNNs의 모듈은 하나의 tanh layer같은 매우 단순한 구조를 가진다.

  <img src="C:\Users\swlee\AppData\Roaming\Typora\typora-user-images\image-20211104102052123.png" alt="image-20211104102052123" style="zoom:80%;" />

* LSTMs 또한 이러한 구조의 체인을 가지고 있으나 반복되는 모듈의 구조는 다르다. 하나의 단순한 신경망 레이어를 가지는 대신 매우 특별한 방법으로 상호작용하는 4개의 신경망 레이어 구조를 가진다.

  <img src="C:\Users\swlee\AppData\Roaming\Typora\typora-user-images\image-20211104102219606.png" alt="image-20211104102219606" style="zoom:80%;" />

  <img src="C:\Users\swlee\AppData\Roaming\Typora\typora-user-images\image-20211104102835344.png" alt="image-20211104102835344" style="zoom:80%;" />





### The Core Idea Behind LSTMs

* LSTMs의 주요한 특징으로서 다이어그램의 상단에서 수평 라인의 cell state가 있다. cell state는 일종의 컨베이어 벨트이며 몇가지 사소한 linear interactions을 통해 체인 전체를 지난다. 정보 변형 없이 단순히 흐르게 해준다.

  ![image-20211104103858405](C:\Users\swlee\AppData\Roaming\Typora\typora-user-images\image-20211104103858405.png)

* LSTM은 게이트라 불리는 구조에 의해서 조심스럽게 조절되어 cell state에 정보를 제거하거나 추가하는 능력을 가지고 있다. 게이트는 선택적으로 정보를 통과시키는 방법이며 sigmoid layer와 pointwise multiplication operation으로 구성되어있다. sigmoid layer는 각 구성 요소가 얼마만큼 통과할지를 묘사한다. 값이 0 이면 통과시키지 않고 값이 1이면 모든 것을 통과시킨다. LSTM은 cell state를 보호하고 제어하기 위해 3개의 게이트를 가진다.

  ![image-20211104110047030](C:\Users\swlee\AppData\Roaming\Typora\typora-user-images\image-20211104110047030.png)





### Step-by-Step LSTM Walk Through

* LSTM의 첫 단계는 어떤 정보를 cell state에서 버릴지 결정하는 것 이다. 이는 'forget gate layer' 라 불리는 sigmoid layer에 의해 결정된다. forget gate layer의 출력값이 1이면 이전 cell state 값인 Ct-1을 완벽하게 유지하고 0이면 삭제한다.

  ![image-20211104111518477](C:\Users\swlee\AppData\Roaming\Typora\typora-user-images\image-20211104111518477.png)

* 다음 단계는 cell state에 어떤 새로운 정보를 저장할지 결정한다. 두 파트로 나뉘는데 첫째로 'input gate layer'라 불리는 sigmoid layer가 어떤 값을 업데이트할지 결정한다. 그리고 tanh layer는 state에 추가할 새로운 vector 값을 만든다. 그 후, state를 업데이트하기 위해 두 값을 결합한다.  오래된 gender 값을 제거하고 새로운 gender 값을 추가하는 것

  ![image-20211104134051554](C:\Users\swlee\AppData\Roaming\Typora\typora-user-images\image-20211104134051554.png)

* 이제 오래된 cell state인 Ct-1를 새로운 cell state인 Ct로 업데이트하면 된다. 

  ![image-20211104134339355](C:\Users\swlee\AppData\Roaming\Typora\typora-user-images\image-20211104134339355.png)

* 마지막으로 무엇을 출력할지 결정해야 합니다. 이 출력은 cell state를 기반으로 하지만 필터링된 버전이 됩니다. 먼저 cell state의 어떤 부분을 출력할지 결정하는 sigmoid layer를 실행합니다. 그런 다음 tanh를 통해 cell state를 입력하고(값을 -1과 1 사이가 되도록 밀어넣기 위해) Sigmoid 게이트의 출력을 곱하여 결정한 부분만 출력합니다. 언어 모델 예제의 경우, 주제를 보았으므로 다음에 올 경우를 대비하여 동사와 관련된 정보를 출력하려고 할 수 있습니다. 예를 들어 주어가 단수인지 복수인지 출력할 수 있으므로 다음에 오는 경우 동사를 활용해야 하는 형태를 알 수 있습니다.

  ![image-20211104135418579](C:\Users\swlee\AppData\Roaming\Typora\typora-user-images\image-20211104135418579.png)



### Reference

* http://colah.github.io/posts/2015-08-Understanding-LSTMs/
