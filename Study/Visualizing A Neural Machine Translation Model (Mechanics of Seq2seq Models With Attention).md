## Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention) 



### 신경망 기계 번역 모델의 시각화 (Seq2seq + Attention 모델의 메커니즘)

Sequence-to-sequence (Seq2seq) 모델은 기계 번역, 문서 요약, 그리고 이미지 캡셔닝 등의 문제에서 아주 큰 성공을 거둔 딥러닝 모델입니다. 구글 번역기도 2016년 말부터 이 모델을 실제 서비스에 이용하고 있습니다. 이 seq2seq 모델은 두 개의 선구자적인 논문에 의해 처음 소개되었습니다.

그러나 이 모델을 구현을 할 수 있을 정도까지 잘 이해하기 위해서는 모델 자체뿐만 아니라 이 모델의 기초에 이용된 수많은 기본 개념들을 이해해야합니다. 저는 이런 여러 개념들을 시각화하여 한 번에 볼 수 있다면 많은 사람들이 더 쉽게 이해할 수 있을 거라고 생각했습니다. 그것이 바로 제가 이번 포스트에서 목표하는 바입니다. 

Seq2seq 모델은 글자, 단어, 이미지의 feature 등의 아이템 시퀸스를 입력으로 받아 또 다른 아이템 시퀸스를 출력합니다. 학습된 모델은 다음과 같이 작동합니다.



![seq2seq_1](https://user-images.githubusercontent.com/92927837/141736795-7ccb1bf1-b959-4223-b7d7-c488b20177ba.gif)

신경망 기계 번역의 경우, 입력은 일련의 단어로 이루어진 sequence 이며 맨 앞 단어부터 차례대로 모델에서 처리됩니다. 그리고 출력으로 비슷한 형태이나 다른 언어의 sequence가 나오게 됩니다.



![seq2seq_2](https://user-images.githubusercontent.com/92927837/141737472-05647c7f-753e-43e9-bc5a-bbf95d530fcd.gif)



### 모델 안을 들여보기

이제 모델 안을 자세히 들여다보겠습니다. Seq2seq 모델은 하나의 **encoder**와 하나의 **decoder**로 이루어져 있습니다. 

**encoder**는 입력의 각 아이템을 처리하여 거기서 **정보를 추출한 후 그것을 하나의 벡터로** 만들어냅니다 (흔히 **context**라고 불립니다). 입력의 모든 단어에 대한 처리가 끝난 후 **encoder**는 **context**를 **decoder**에게 보내고 출력할 아이템이 하나씩 선택되기 시작합니다.



![seq2seq_3](https://user-images.githubusercontent.com/92927837/141740959-f50a81c3-ea47-4817-99dd-af6abbfa471b.gif)



물론 seq2seq 모델의 한 예시인 신경망 기계 번역도 동일한 구조를 가지고 있습니다.



![seq2seq_4](https://user-images.githubusercontent.com/92927837/141741134-02be16a8-0ca7-4ef6-874a-76485b7726d9.gif)

기계 번역의 경우, **context**가 하나의 벡터 형태로 전달됩니다. **encoder**와 **decoder**는 둘 다 RNN을 이용하는 경우가 많습니다. 



![seq2seq_context](https://user-images.githubusercontent.com/92927837/141741337-5bbaa969-de46-4efd-b868-59987762e857.png)



**context**는 float로 이루어진 하나의 **벡터**입니다. 우리의 시각화 예시에서는 더 높은 값을 가지는 소수를 더 밝게 표시할 예정입니다.

이 **context** 벡터의 크기는 모델을 처음 설정할 때, 원하는 값으로 설정할 수 있습니다. 하지만 보통 **encoder** RNN의 hidden unit 개수로 정합니다. 이 글의 시각화 예시에서는 크기 4의 **context** 벡터를 이용하는데요, 실제 연구에서는 256, 512, 1024와 같은 숫자를 이용합니다.

Seq2seq 모델 디자인을 보게 되면 하나의 RNN은 한 time step마다 두 개의 입력을 받습니다. 하나는 sequence의 한 아이템이고 다른 하나는 그전 step에서의 RNN의 hidden state 입니다. 이 두 입력들은 RNN에 들어가기 전에 꼭 vector로 변환되어야 합니다. 하나의 단어를 벡터로 바꾸기 위해서 우리는 "word embedding" 이라는 기법을 이용합니다. 이 기법을 통해 단어들은 벡터 공간에 투영되고 그 공간에서 우리는 단어 간 다양한 의미와 관련된 정보들을 알아낼 수 있습니다.



![seq2seq_embedding](https://user-images.githubusercontent.com/92927837/141742367-6a7bd8be-42a6-4570-a997-c57c69b27cbb.png)



앞서 설명한 대로 encoder에서 단어들을 처리하기 전에 먼저 벡터들로 변환해주어야 합니다. 우리는 word embedding 알고리즘을 이용해 변환합니다. 또한 우리는 pre-trained embeddings을 이용하거나 우리가 가진 데이터 셋을 이용해 직접 학습시킬 수 있습니다. 보통 크기 200 or 300의 embedding 벡터를 이용하지만 이 포스트에서는 예시로서 크기 4의 벡터를 이용합니다.



여기까지 모델에서 등장하는 주요 벡터들을 소개해보았는데요. 이제 RNN의 원리에 대해서 간단히 다시 돌아보고 시각화에서 우리가 쓸 기호들을 설명하겠습니다.



![seq2seq_RNN_1](https://user-images.githubusercontent.com/92927837/141743703-5d87a08b-0744-4272-96ae-ba502a908d1d.gif)



이와 같이 time step #2에서는 두 번째 단어와 첫 번째 hidden state를 이용하여 두 번째 출력을 만듭니다. 본 포스트의 뒷부분에서는 이와 유사한 애니메이션을 이용해 신경망 기계 번역 모델을 설명하겠습니다.



밑의 영상을 보시면 **encoder** 혹은 **decoder**에서 일어나는 각 진동은 한 번의 step 동안 출력을 만들어내는 과정을 의미합니다. **encoder**와 **decoder**는 모두 RNN 이며, RNN은 한 번 아이템을 처리할 때 마다 새로 들어온 아이템을 이용해 그의 hidden state를 업데이트 합니다. 이 hidden state는 그에 따라 **encoder**가 보는 입력 시퀸스 내의 모든 단어에 대한 정보를 담게 됩니다. 

그러면 시각화된 **encoder**의 **hidden states**를 볼까요 ? 여기서 한 가지 짚고 넘어갈 점은 마지막 단어의 **hidden state**가 바로 **decoder**에게 넘겨주는 **context**라는 겁니다.



![seq2seq_5](https://user-images.githubusercontent.com/92927837/141744454-41940da9-40cc-4c71-8125-113420cba3ff.gif)



**decoder**도 그만의 **hidden state**를 가지고 있으며 step 마다 업데이트를 하게 됩니다. 우리는 아직 모델의 큰 그림을 그리고 있기에 위의 영상에서는 그것을 표시하지 않았습니다.

그렇다면 이제 이 seq2seq 모델을 다른 방법으로 시각화 해보겠습니다. 아래의 영상은 **하나로 합쳐진 RNN이 아닌 각 step 마다 RNN을 표시하는 방법**입니다. 이렇게 하면 각 step 마다 입력과 출력을 정확히 볼 수 있습니다. 



![seq2seq_6](https://user-images.githubusercontent.com/92927837/141744779-40e2bdc3-f8ab-4ed3-93e8-09057f6832f1.gif)





### 이제 Attention을 해봅시다

연구를 통해 **context** 벡터가 이런 seq2seq 모델의 가장 큰 걸림돌인 것으로 밝혀졌습니다. 이렇게 **하나의 고정된 벡터로 전체의 맥락을 나타내는 방법은 특히 긴 문장들을 처리하기 어렵게** 만들었습니다. 이에 대한 해결 방법으로 제시된 것이 바로 "Attention" 입니다. [Bahdanau et al., 2014](https://arxiv.org/abs/1409.0473) and [Luong et al., 2015](https://arxiv.org/abs/1508.04025). 이 두 논문이 소개한 attention 메커니즘은 seq2seq 모델이 **디코딩 과정에서 현재 스텝에서 가장 관련된 입력 파트에 집중**할 수 있도록 해줌으로써 기계 번역의 품질을 매우 향상 시켰습니다.



![seq2seq_attention](https://user-images.githubusercontent.com/92927837/141746407-f4d04b5b-b309-4b13-b7dc-291afafab420.png)



step 7에서 attention 메커니즘은 영어 번역을 생성하려 할 때, **decoder**가 단어 “étudiant”에 집중하게 합니다. 이렇게 **step 마다 관련된 부분에 더 집중할 수 있게** 해주는 attention model은 attention이 없는 모델보다 훨씬 더 좋은 결과를 생성합니다. 



계속해서 개략적인 차원에서 attention 모델을 살펴보도록 하겠습니다. attention 모델과 기존 seq2seq 모델은 2 가지 차이점을 가집니다. 

첫 번째로 **encoder**가 **decoder**에게 넘겨주는 데이터의 양이 attention 모델에서 훨씬 더 많다는 점입니다. 기존 seq2seq 모델에서는 그저 마지막 아이템의 hidden state 벡터를 넘겼던 반면 attention 모델에서는 *모든* step의 **hidden state**를 **decoder**에게 넘겨줍니다.



![seq2seq_7](https://user-images.githubusercontent.com/92927837/141747379-c17f713d-3e8d-472e-ad00-ed5e4f31b48e.gif)



두 번째로는 attention 모델의 **decoder**가 출력을 생성할 때, 하나의 추가 과정이 필요합니다. **decoder**는 현재 step에서 관련 있는 입력을 찾아내기 위해 다음 과정을 실행합니다.

1. **encoder**에서 받은 전체 **hidden state**를 봅니다. 각 step에서의 encoder hidden states는 이전 맥락에 대한 정보도 포함하고 있지만 그 step에서의 입력 단어와 가장 관련이 있습니다.
2. 각 step의 hidden state 마다 점수를 매깁니다. 
3. 매겨진 점수들에 softmax를 취하고 이것을 각 time step의 hidden states에 곱해서 더합니다. 이를 통해 높은 점수를 가진 hidden states는 더 큰 부분을 차지하게 되고 낮은 점수를 가진 hidden states는 작은 부분을 가져가게 됩니다.



![seq2seq_attention_process](https://user-images.githubusercontent.com/92927837/141749729-7b847ddb-f94c-4fc9-9359-f73eaa930df1.gif)



이러한 점수를 매기는 과정은 decoder가 단어를 생성하는 매 step 마다 반복됩니다.



이제 지금까지 나온 모든 과정들을 합친 다음 영상을 보고 어떻게 attention이 작동하는지 정리해보겠습니다.

1. attention 모델에서의 decoder RNN은 <END> 와 추가로 initial decoder hidden state를 입력받습니다.
2. decoder RNN은 두 개의 입력을 가지고 새로운 hidden state 벡터를 출력합니다 (h4). RNN의 출력 자체는 사용되지 않고 버려집니다.
3. Attention 과정: encoder의 hidden state 모음과 decoder의 hidden state h4 벡터를 이용하여 그 step에 해당하는 context 벡터 C4를 계산합니다.
4. h4와 C4를 하나의 벡터로 concatenate 합니다.
5. 이 벡터를 feedforward 신경망 (seq2seq 모델 내에서 함께 학습되는 layer 입니다)에 통과 시킵니다.
6. feedforward 신경망에서 나오는 출력은 현재 time step의 출력 단어를 나타냅니다.
7. 이 과정을 다음 time step에서도 반복합니다.



![seq2seq_attention_tensor_dance](https://user-images.githubusercontent.com/92927837/141750486-810cf11d-a8a1-4b9f-b4a9-3782a48cac17.gif)



이 attention을 이용하면 각 decoding step에서 입력 문장의 어떤 부분을 집중하고 있는지에 대해 볼 수 있습니다.



![seq2seq_9](https://user-images.githubusercontent.com/92927837/141872868-a22d9011-e429-48d3-a2be-d030e790c0b8.gif)

여기서 한가지 짚고 넘어갈 것은 현재 모델이 아무 이유 없이 출력의 첫 번째 단어를 입력의 첫 번째 단어와 맞추는 (align) 것이 아니란 것입니다. 학습 과정에서 입력되는 두 개의 언어를 어떻게 맞출지는 학습이 됩니다 (우리의 예시에는 불어와 영어입니다). 얼마나 이것이 정확하게 학습 되는지를 알아보기 위해서 앞서 언급했던 attention 논문들에서는 다음과 같은 예시를 보여줍니다.



![seq2seq_attention_sentence](https://user-images.githubusercontent.com/92927837/141872901-0e1e7401-f688-48fc-a500-3418926a687f.png)



모델이 “European Economic Area”를 제대로 출력할 때 모델이 얼마나 잘 주의를 하고 있는지를 볼 수 있습니다. 영어와는 달리 불어에서는 이 단어들의 순서가 반대입니다 (“européenne économique zone”). 문장 속의 다른 단어들은 다 비슷한 순서를 가지고 있습니다.



### Reference

* https://nlpinkorean.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/
* https://proceedings.neurips.cc/paper/2014/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf
* https://www.youtube.com/watch?v=UNmqTiOnRfg -> RNN
* https://shuuki4.wordpress.com/2016/01/27/word2vec-%EA%B4%80%EB%A0%A8-%EC%9D%B4%EB%A1%A0-%EC%A0%95%EB%A6%AC/
* https://machinelearningmastery.com/what-are-word-embeddings/
* https://arxiv.org/abs/1409.0473
* https://arxiv.org/abs/1508.04025
* https://github.com/tensorflow/nmt
* https://www.udacity.com/course/natural-language-processing-nanodegree--nd892
* https://arxiv.org/abs/1706.03762

