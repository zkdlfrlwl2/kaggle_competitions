**AlexNet** - ILSVRC 2012년 1등 

* Conv -> ReLU -> Pooling으로 이어지는 CNN 기법 확립

* Architecture

  ![1](https://user-images.githubusercontent.com/92927837/147621304-2b1bcdf2-347d-4649-b9ee-395783fc3f60.PNG)

  * Input layer - Conv1 - MaxPool1 - Norm1 - Conv2 - MaxPool2 - Norm2 - Conv3 - Conv4 - Conv5 - MaxPool3 - FC1 - FC2 - Output layer
  * 11x11, 5x5와 같은 큰 size의 Kernel 적용

* 개요 및 특징

  * Activation 함수로 ReLU 첫 사용
    * sigmoid or tanh 대비 수렴 속도가 약 6배 빨라짐
  * MaxPooling, Overlapping Pooling 적용
  * Local Response Normalization (LRN) 사용
    * Generalization 목적
    * 이후 Batch Normalization 적용
  * Overfitting 개선하기 위해 Dropout Layer & Weight Decay & Data Augmentation 적용

* 추가 검색 필요

  * LRN 란 ? 
  * Weight Decay 란 ?





**VGGNet** - ILSVRC 2014년 2등 

* AlexNet 대비 더 좋은 정확도를 얻기 위해 AlexNet의 기본 구조를 향상 시키기 위한 많은 시도가 이루어 졌는데 이 논문에서는 **ConvNet 구조의 깊이에 집중** - 3x3 convolution filters 사용 
* AlexNet과 차별점
  * LRN 미적용
    * 단순 성능 하락 이유
  * 11x11, 5x5와 같은 큰 사이즈의 filter 대신 3x3 filter를 사용하여 성능 향상
    * 5x5 convolutional filter = 2개의 3x3 convolutional filter
    * 7x7 convolutional filter = 3개의 3x3 convolutional filter
    * 3x3 filter를 여러 겹 이용하게 되면 하나의 relu 대신 2개, 3개의 relu를 이용할 수 있고 parameter 수를 감소시킬 수 있다. 
    * C개의 channel을 가진 3개의 3x3 filter를 이용하면 연산량은 3(3^2C^2) = 27C^2가 되고 C개의 channel을 가진 1개의 7x7 filter를 이용하면 연산량은 7^2C^2=49C^2가 된다.
    * filter size 3x3, channel 3, 개수 64 의 parameter 수 = 3x3x3x64 = 1728 + bias 64 = 1792
    * filter size 5x5, channel 3, 개수 64 의 parameter 수 = 5x5x3x64 = 4800 + bias 64 = 4864
    * 3x3 size의 filter 2개 사용하는 게 5x5 size filter 1개 사용하는 것 보다 parameter 수가 적고 성능 향상에 도움을 줌 - 실험적으로 확인
    * 7x7 filter를 3개의 3x3 filter로 분해하면 parameter 수도 감소시키고 더 많은 relu 함수를 이용할 수 있게 된다.
    * 1x1 conv layer는 비선형성을 부여하기 위한 용도이다. 입력과 출력의 channels을 동일하게 하고 1x1 conv layer를 이용하면 relu 함수를 거쳐 추가적인 비선형성을 부여
  * Pre-initialization
    * 수렴 속도 빨라져 필요 epoch 수가 줄어듬 
    * 가중치 초기화 값은 평균 0 분산 0.01인 정규 분포에서 무작위 추출
    * 초기화가 잘못되면 gradient 불안정함 때문에 학습을 지연시키므로 이를 해결 하기 위해 VGGNet 논문 본문 중 가장 간단한 모델 구조인 A를 학습시킨 뒤 학습된 첫 번째, 네 번째 conv layer와 3개의 FC layer 가중치를 이용하여 다른 깊은 모델을 학습 
  * Data augmentation
    * Crop된 이미지를 무작위로 수평 뒤집기
    * 무작위로 RGB 값 변경
    * Image rescaling
      * 256x256 고정
      * 356x356 고정
      * [256, 512] 범위로 랜덤하게 resize - 성능 향상에 도움
  * Inference
    * Test dataset에도 augmentation 적용하여 성능 개선 





**GoogLeNet** - ILSVRC 2014년 1등 with Inception block

* Network의 깊이와 넓이가 증가할 수록 높은 정확도를 얻지만 parameter 수가 기하급수적으로 늘어나고 overfitting되기 쉬우나 GoogLeNet은 깊이가 22층이면서도 AlexNet에 비해 parameter 수가 12배 적다.

* parameter 수 증가와 overfitting 문제를 해결하기 위해 sparse하게 연결된 구조가 필요 

  * 이를 위해 NIN Network in network 논문을 인용

    * NIN은 높은 상관관계에 있는 뉴런들을 군집화 시키고 마지막 계층에서 활성화 함수들의 상관관계를 분석함으로써 최적의 network topology를 구축할 수 있다고 한다. Multilayer perceptron network를 Convolution 시 추가로 적용하여 feature map을 생성한다.

    <img src="https://user-images.githubusercontent.com/92927837/147631851-24bfaa47-5192-4098-9fbe-e30c855ad211.PNG" alt="2" style="zoom:80%;" />

    * 이를 통해 fully connected layer와 convolutional layer를 dense 구조에서 sparse 구조로 바꿀 수 있다. Inception module을 적용해서 달성. Dense 구조에서 Sparse 구조로 바꾸어 효율적인 데이터 분포로 만들어 더 깊고 넓은 network를 만들어 정확도를 높일 수 있다.

* **Inception Module**

  * Inception module의 주요 아이디어는 convolutional network에서 sparse 구조를 손쉽게 dense 요소드로 근사화하고 다룰 수 있는 방법을 찾는 것에 근거한다.

  * feature map을 효과적으로 추출하기 위해 1x1, 3x3, 5x5의 convolution 연산을 각각 수행하며 matrix의 height, width가 같아야 하므로 pooling 추가

    ![4](https://user-images.githubusercontent.com/92927837/147633757-f6284d2d-7d47-4d4f-bac3-b982d6e6a7be.PNG)

  * 이 module의 핵심인 1x1 convolution의 목적은 dimension reduction 적용하여 input filter 수를 조절하여 연산량을 감소시키는 것이다. 이전 layer에서 512개의 channel을 가진 output이 생성됐으면 256개의 1x1 convolution filter를 이용해서 256 channel로 줄일 수 있다. 이를 통해 다양한 크기의 filter (1x1, 3x3, 5x5)를 적용하여 여러 특징을 추출하면서도 연산량을 낮출 수 있게 된다. 

  * Inception module을 통해 다음과 같은 효과를 얻을 수 있다.

    * 1x1 conv dimension reduction을 통해 다음 계층의 input 수를 조절하여 연산량을 줄인다.
    * 1x1, 3x3, 5x5 conv 연산을 통해 동시에 서로 다른 규모에서 특징을 추출할 수 있다.
    * channel size를 유지하더라도 relu를 통해 non-linearity 성질을 추가할 수 있다.
  
* Model

  ![6](https://user-images.githubusercontent.com/92927837/147634412-f8a147cf-ff56-4fc5-bc95-9d8efb20f2fe.PNG)

  * ReLU 적용, 0.9 momentum SGD, 8 epoch 마다 4% lr 감소 
  * 모델의 깊이가 깊으면 gradient가 0으로 수렴하는 gradient vanishing 문제가 발생할 수 있다. 따라서 모든 layer에 효과적으로 기울기를 뒤로 전달하기 위해 auxiliary classifier를 중간 layer에 추가했다. 
    * 중간 layer에 classifier를 추가함으로써 역전파하는 gradient 신호를 늘리고 추가적인 regularization을 제공한다. 
    * 학습 도중 auxiliary classifier의 loss에 0.3을 곱해 전체 loss에 더해주고 inference 시 이용하지 않는다.





**ResNet** - ILSVRC 2015년 1등

* VGG 이후 더 깊은 Network에 대한 연구가 늘어났으나 Network 깊이가 깊어질 수록 성능이 저하됨

  * Kernel size 조절, Dropout, Weight Decay 등으로 Overfitting 방지하면서 Network 깊이를 늘려가는 한계점 봉착

* skip/shortcut connection을 사용하지 않는 일반적인 CNN (AlexNet, VGGNet)인 Plain Network는 깊이가 깊어질수록 Gradient Vanishing & Exploding 문제가 발생한다.

  * Gradient Vanishing & Exploding

    * 신경망이 깊을 때, 작은 미분값이 여러 번 곱해지면 0에 가까워 지고 이를 기울기 소실이라 하며 반대로 큰 미분값이 여러 번 곱해지면 값이 매우 커지게 되는 데 이를 기울기 폭발이라 한다.

    * 신경망이 깊어질 수록 더 정확한 예측을 할 것이라 생각할 수 있는 데 아래 그림을 보면 20-layer plain network가 50-layer plain network보다 성능이 더 좋게 나타난다. 논문에서는 이를 degradation 문제라고 하고 기울기 소실에 의해 발생한다고 한다.

      <img src="https://user-images.githubusercontent.com/92927837/147638906-216c71cd-b177-4186-9345-37a429a46028.png" alt="7" style="zoom:80%;" />

* Model

  ![image](https://user-images.githubusercontent.com/92927837/147714817-38edd9d0-f1ef-4995-aaec-ccf2533cd49e.png)

* **Concept**

  * ResNet은 Residual Network의 약자로 **잔차**의 개념을 도입한 방법이다. 이를 이해하기 위해서 Block과 Identity Mapping을 알아야 한다.

  * **Block**

    * Layer의 묶음이다. ResNet에서는 2개의 Conv layer를 하나의 block으로 묶어서 Residual Block이라고 부른다. 이런 Residual Block를 여러 개 쌓아간 것이 ResNet의 구조이다. 하지만 Block 수가 늘어날 수록 Parameter 수도 급격하게 늘어나  이를 해결하기 위해 **Bottleneck Block** 라는 것을 제안 한다.

      ![image](https://user-images.githubusercontent.com/92927837/147714660-bccbee8a-6903-4c14-b37d-e8168e950584.png)

    * 기존 Block과 비교하여 3x3 Conv Layer 앞 뒤로 1x1 Conv Layer를 추가하여 Channel의 수를 조절하면서 차원을 줄였다 늘리는게 병목 같다 하여 Bottleneck Block이라 불리게 되었다. 총 parameter 수가 약 6배 감소되었다. 1x1 Conv Layer는 신경망의 성능을 감소시키지 않고 parameter 수를 감소 시킨다.

  * **Identity Mapping (Shortcut, Skip Connection)**

    * 위 그림의 + 기호가 Identity Mapping이다. Identity Mapping이란 입력으로 들어간 값 x가 어떠한 함수를 통과하더라도 다시 x가 나와야 한다. 항등함수 개념이다.

    * ```python
      class Bottleneck(nn.Module):
          ### torchvision의 resnet 코드입니다.
          def forward(self, x):
              identity = x
      
              out = self.conv1(x)
              out = self.bn1(out)
              out = self.relu(out)
      
              out = self.conv2(out)
              out = self.bn2(out)
              out = self.relu(out)
      
              out = self.conv3(out)
              out = self.bn3(out)
      
              if self.downsample is not None:
                  identity = self.downsample(x)
      
              out += identity
              out = self.relu(out)
      
              return out
      ```

    * forward의 첫 line에서 x를 identity라는 변수에 따로 저장하고 마지막 out에 identity를 더한다. 예를 들어 x가 (28, 28, 64)인 feature map이라 해보자. identity의 shape은 그대로이며 마지막 out의 shape도 identity와 같아야 Element-wise sum이 되어 최종 out shape도 (28, 28, 64)가 된다. 아마 첫 입력의 shape와 출력의 shape가 같아서 Identity Mapping이라 하는 것 같다.

    * H(x)=F(x)+x식을 보자. 진정한 의미의 Identity Mapping은 H(x)=F(x)+x=x가 된다. Identity의 가정이 성립되기 위해서는 F(x)가 0이 되어야 한다. x를 좌변으로 이항하면 F(x)=H(x)−x가 되고 이것은 마치 잔차처럼 보이게된다. (마치 선형회귀모형에서 많이 보던 y=Xβ+ϵ의 모습) 그래서 이 알고리즘이 Residual (잔차) Network가 된 것이다.

  * **DownSample**

    * 모델 구조 그림에서 실선은 identity mapping이며 점선은 Feature map의 shape가 축소되는 지점에서 일어나는 Pooling이 되는 부분이다. Pooling은 stride 2 & 1x1 filter 사용

    * 맨 위 그림의 보라색 영역을 살펴보면 첫 번째 block에서 Feature map shape가 (28, 28, 64) 였다면 세 번째 block의 마지막 Conv Layer를 통과하고 Identity Mapping 까지 완료된 Feature map shape도 (28, 28, 64) 이다.

    * 녹색 영역의 시작 지점에서 채널의 수가 128로 늘어났고 /2 라는 표시가 있는 것으로 보아 첫 번째 block 대비 Conv layer의 stride가 2로 늘어나 (14, 14, 128)로 바뀐다는 것을 알 수 있다. Identity shape는 여전히 (28, 28, 64) 이므로 새로운 출력의 (14, 14, 128)로 맞춰주지 않으면 Identity Mapping을 할 수 없게 된다. 그래서 이 Identity에 대하여 down sample 해주는 것이 필요하다.

    * 방법은 stride 2를 가진 1x1 conv layer를 하나 연결해주기만 하면 된다. 이렇게 down sample 하여 연결되는 방법을 projection shortcut이라고 하고 해당 block 영역을 Convolution Block이라 한다. 일반적인 shortcut 형태는 Identity Block이라 한다.

      <img src="https://user-images.githubusercontent.com/92927837/147716786-0c615c47-0c5f-40d2-8b4e-cc2f83457fcb.png" alt="image" style="zoom:80%;" />

  * **Pre-Activation**

    * ResNet 모델에서는 가중치 초기화 방법으로 He, Xaiver와 같이 잘 알려진 초기화 방법을 사용하지 않았다. 대신 Conv Layer 전에 Batch Normalization를 적용했다. 
  
* **Extra**

  * 레이어의 수가 10개 넘어가는 네트워크에서 Gradient Vanishing & Exploding 문제는 normalization initialization and intermediate normalization layers로 크게 개선되었으나 더 layer를 깊게 쌓으니 수렴하기 시작할 때, training error & test error가 20-layer보다 56-layer가 증가하는 degradation 문제가 발견되었고 이는 ResNet의 Identity mapping with shortcuts으로 해결되었다.
  * ResNet의 BasicBlock도 101 이렇게 깊게 쌓으면 성능이 향상되었으나 BottleNeck로 바꾼 이유는 BottleNeck도 동일하게 degradation 문제를 해결해주고 더 경제적이었기때문에







**DenseNet** - 2017년

* DenseNet은 ResNet보다 적은 parameter 수로 더 높은 성능을 가진 모델이다. **DenseNet은 모든 레이어의 FeatureMap을 연결한다. 이전 Layer의 FeatureMap을 그 이후의 모든 Layer의 FeatureMap에 연결한다.** 연결할 때, ResNet과 달리 덧셈이 아니라 concatenate를 수행한다. 연결하기 위해 각 FeatureMap 크기가 동일 해야하고 FeatureMap을 계속해서 연결하면 Channel 수가 많아질 수 있기 때문에 각 Layer의 FeatureMap Channel 수는 굉장히 작은 값을 사용 한다.

  <img src="https://user-images.githubusercontent.com/92927837/147723799-6b4a731f-9de6-43ec-a9d6-195adfc79564.png" alt="image" style="zoom: 67%;" />

  * Concatenation을 통한 이점
    * **strong gradient & information flow를 통해 기울기 소실 문제를 완화하고 Feature reuse 효과도 있다.** 기존 CNN 모델의 경우, 많은 Layer를 통과하여 신경망의 끝에 다다르면 첫 Layer의 Feature map 정보가 사라질 수 있는데 이를 feature reuse 문제라고 한다. **DenseNet의 경우, 첫 Layer의 Feature map 정보를 마지막 Layer 까지 연결하므로 정보 및 기울기 소실 문제를 완화 시키고 다양한 Layer의 Feature map을 연결해서 학습하기 때문에 정규화 효과도 있다.**
    * 각 Layer의 Feature map을 연결하기 때문에 일반 CNN보다 적은 수의 channel을 사용하여 parameter 수가 적다.

* 주요 특징

  * ResNet 

    * ResNet의 $$l$$ 번째 Layer의 출력값은 $$x_l=H(x_{l-1})+x_{l-1}$$이 된다. $$H()$$는 conv, bn, relu 연산을 의미 한다. 그리고 $$+ x_{l-1}$$은 skip connection에 의한 덧셈이다. **Identity function을 통해 이전 Layer에서 최근 Layer로 grandient가 직접적으로 전달될 수 있지만 덧셈으로 결합되기 때문에 신경망에서 정보 흐름 (information flow)이 지연될 수 있다.** 
    
  * Dense connectivity
  
    * Information flow를 더욱 향상 시키기 위해 한 Layer를 모든 후속 Layer에 직접 연결하는 새로운 연결 패턴을 제안 한다.  따라서 $$l^{th}$$ layer는 모든 선행 Layer의 Feature-map을 입력으로 받는다. $$x_0, ..., x_{l-1}$$ 를 입력으로 $$x_l=H_l([x_0, x_1,...,x_{l-1}])$$ 이 되고 $$[x_0,x_1,...,x_{l-1}]$$ 은 layer $$0,1,...,l-1$$ 에서 생성된 Feature-map 들의 concatenation 이다. $$H()$$는 BN, ReLU, 3x3 Conv의 조합이다.
  
  * Dense Block
  
    * 연결(concatenation) 연산을 수행하기 위해서는 피쳐맵의 크기가 동일해야 합니다. 하지만 피쳐맵 크기를 감소시키는 pooling 연산은 conv net의 필수적인 요소입니다. **pooling 연산을 위해 Dense Block 개념을 도입합니다.** Dense Block은 여러 레이어로 구성되어 있습니다. **Dense Block 사이에 pooling 연산을 수행합니다.** pooling 연산은 BN, 1x1conv, 2x2 avg_pool로 수행합니다. 그리고 이를 transition layer이라고 부릅니다. - Pooling layers
  
       transition layer에는 theta 하이퍼 파라미터가 존재합니다. theta는 transition layer가 출력하는 채널 수를 조절합니다. transition layer의 입력값 채널 수가 m이면 theta * m 개의 채널수를 출력합니다. 1x1 conv에서 채널 수를 조절하는 것입니다. 논문에서는 theta=0.5를 사용하여 transition layer의 출력 채널 수를 0.5m으로 합니다. **즉, transition layer는 피쳐 맵의 크기와 채널 수를 감소시킵니다.** - Compression
  
       아래 그림을 보면 DenseNet은 3개의 Dense Block과 2개의 transition layer로 이루어져 있습니다.
  
      ![image](https://user-images.githubusercontent.com/92927837/147728575-c0e1dda6-04bc-4fb4-8cd7-81e8fab03c81.png)
  
  * Growth rate
  
    * Dense Block 내의 레이어는 k개의 피쳐 맵을 생성합니다. 그리고 이 k를 Growth rate라고 부르는 하이퍼파라미터 입니다. 논문에서는 k=12를 사용합니다. l번째 레이어는 k0 + k * (l-1) 개의 입력값을 갖습니다. k0은 입력 레이어의 채널 수 입니다. **이 Growth rate는 각 레이어가 전체에 어느 정도 기여를 할지 결정합니다.**
  
  * Bottleneck layers
  
    * DenseNet은 Bottleneck layer를 사용합니다. 보틀넥 레이어는 3x3 conv의 입력값 채널을 조절하여 연산량에 이점을 얻기 위해 사용됩니다. 1x1 conv는 3x3 conv의 입력값을 4k로 조절합니다. 그리고 3x3 conv는 k개의 피쳐맵을 생성하고 이전 레이어와 연결됩니다. 
  
      <img src="https://user-images.githubusercontent.com/92927837/147729719-163d4315-9b4e-49df-ae5d-713a67902707.png" alt="image" style="zoom:67%;" />







**MobileNet** - 2017년

* MobileNet은 이름처럼 모바일 환경에 최적화된 모델이며 모델 아키텍처가 결정된 이후 경량화 하는 기법인 Quantization, Prunning, Knowleage Distilation과 같은 방법론이 아니라 신경망 자체를 경량화한 방법론으로써 이 설계기법은 이후 연구에 많은 영향을 주었다.

* MobileNet은 **Depthwise separable convolution**을 활용하여 모델을 경량화 했다. 경량화에 집중한 이유는 핸드폰이나 임베디드 시스템 같이 저용량 메모리 환경에 딥러닝을 적용하기 위해서이다. **메모리가 제한된 환경에서 MobileNet을 최적으로 맞추기 위해 두 개의 파라미터를 소개 한다. 두 파라미터는 Latency와 Accuracy의 균형을 조절한다.** 

* ResNet 이후 출시된 모델들은 신경망이 깊어지면서 Layer 당 차지하는 Channel 수가 급증하게 되었다. 이에 따라 **곱하기와 더하기 (Mult-Add) 로 이루어진 Convolution Operation은 점점 더 많은 연산 비용을 요구하게 되었고 이는 학습/추론 시간의 증가, Memory I/O 증가, 전력 낭비 등 필연적인 문제**를 가지고 있다. **MobileNet-V1에서는 이러한 연산의 비효율성에 초점을 두고 Convolution Operation의 구조적 변경을 제안** 했다. 

* Depthwise separable convolution

  <img src="https://user-images.githubusercontent.com/92927837/147867383-d5a16c20-628a-40b4-9a21-7ff13217845a.png" alt="image" style="zoom:80%;" />

  * Depthwise convolution
  
    * **Depthwise convolution은 각 입력 채널에 대하여 3x3 conv 하나의 필터가 연산을 수행하여 하나의 feature map을 생성 한다.** 입력 채널 수가 M개면 M개의 feature map을 생성 한다. 각 채널마다 독립적으로 연산을 수행하여 **spatial correlation**을 계산하는 역할을 한다. 예를 들어 5 채널의 입력값이 입력되었으면 5개의 3x3 conv filter가 각 채널에 대하여 연산을 수행하고 5개의 feature map을 생성 한다. 연산량은 하기와 같다. 
      $$
      D_K \cdot D_K \cdot M \cdot D_F \cdot D_F
      $$
      D_k는 입력 값의 크기, M은 입력 채널의 수, D_F는 feature map의 크기
  
  * Pointwise convolution
  
    * **Pointwise convolution은 Depthwise convolution이 생성한 feature map들을 1x1 conv filter로 채널 수를 조정 한다.** 1x1 conv filter는 모든 채널에 대하여 연산하므로 **cross-channel correlation**을 계산하는 역할을 한다. 연산량은 다음과 같다.
      $$
      M \cdot N \cdot D_F \cdot D_F 
      $$
      M은 입력 채널의 수, N은 출력 채널의 수, D_F는 feature map의 크기
  
  * Depthwise separable convolution은 Depthwise convolution 이후에 Pointwise convolution을 적용한 것 이다. 아래 그림은 **MobileNet에서 사용하는 Depthwise separable convolution 구조**이다.
  
    <img src="https://user-images.githubusercontent.com/92927837/147867698-fd685266-d51b-40ee-944f-465831ccca80.png" alt="image" style="zoom: 80%;" />
  
    * 연산량은 하기와 같다
      $$
      D_K \cdot D_K \cdot M \cdot D_F \cdot D_F \cdot + M \cdot N \cdot D_F \cdot D_F
      $$
      둘의 연산량을 더해준 것과 같고 기존 conv 연산량 대비 **8 ~ 9배** 더 적다. 기존 conv 연산량은 하기와 같다
      $$
      D_k \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F
      $$
      MoblieNet은 이 Depthwise separable convolution을 기반으로 구축된 모델이다.
  
* MobileNet architecture

  <img src="https://user-images.githubusercontent.com/92927837/147867816-26b323ff-c3fe-4000-a9b5-617dae5226e8.png" alt="image" style="zoom:80%;" />

  * 첫 번째 conv를 제외하고 depthwise separabel convolution을 사용합니다. 마지막 FC 레이어를 제외하고 모든 레이어에 BN, ReLU를 사용합니다. Down-sampling은 depthwise convolution과 첫 번째 conv layer에서 수행합니다. 총 28 레이어를 갖습니다.

* Hyper parameter

  * latency와 accuracy를 조절하는 두 개의 하이퍼 파라미터가 존재 한다.

  * Width Multiplier: Thinner Models

    * 첫 번째 $$\alpha$$는 MoblineNet의 **두께를 결정**한다. Conv Net에서 **두께는 각 Layer의 filter 수를 의미** 한다. 이 $$\alpha$$는 더 얇은 모델이 필요할 때 사용 한다. 연산량은 다음과 같다.
      $$
      D_K \cdot D_K \cdot \alpha M \cdot D_F \cdot D_F + \alpha M \cdot \alpha N \cdot D_F \cdot D_F
      $$
      $$\alpha$$는 0 ~ 1 범위이며 기본적으로 1을 사용 한다. $$\alpha$$를 낮추면 **모델의 파라미터 수가 감소** 한다.

      ![image](https://user-images.githubusercontent.com/92927837/147868086-4f672e5a-e083-4786-ba69-9618a852b4ac.png)

  * Resolution Multiplier: Reduced Representation

    * 두 번째 $$\rho$$는 **모델의 연산량을 감소시키기 위해 사용**한다. $$\rho$$는 **입력 이미지에 적용하여 해상도를 낮춘다**. 범위는 0 ~ 1 이고 기본은 1을 사용 한다.

      ![image](https://user-images.githubusercontent.com/92927837/147868742-d49f606c-a337-4660-b688-9a56b826e3b3.png)






**EfficientNet** - 2019년

* 기존 CNN 모델의 성능을 향상 시키는 방법은 Depth, Width, Resolution 셋 중 하나의 Dimension을 조정하는 것 이었다. 이 중 두 가지 이상을 조정하는 방법도 고려될 수 있지만 이는 매우 사소하게 조정해줘야하는 작업들이 많이 필요하게 되며 그로 인해 최적의 결과를 찾아내지 못하는 경우도 생긴다. 따라서, **ConvNet이 더 좋은 성능을 발휘하게 만드는 이론에 입각한 scale up 방법이 존재**하는지에 대해 의문을 가졌고 사전 조사에 따르면 **Network의 Depth, Width, Resolution 사이의 균형을 맞추는 것**이 성능 향상에 있어서 매우 중요하며 놀랍게도 이들간의 균형은 간단한 상수비로 구해질 수 있다는 것이다. 이러한 관점에 기반하여 간단하면서도 효율적인 '**Compound scaling method'**를 제안한다. 이전의 임의 조정 방식들과 다르게 Network Depth, Width, Resolution를 균등하게 scaling 한다.

* Compound Model Scaling

  * Model scaling에 의한 성능 향상은 baseline network에 매우 의존적이기 때문에 baseline network를 설정하는 데 있어서 neural architecture search (NAS) - AutoML 를 사용하고 발견한 모델이 efficientnet b0이다. 이 b0 모델을 기점으로 세 가지 scaling factor를 동시에 고려하는 **Compound Scaling**을 적용하여 b1~7 모델을 생성했다.

  * 최적의 Layer architecture F_i 발견에 중점을 두는 일반적인 ConvNet 설계와 달리 Model scaling은 baseline network에서 미리 정의된 F_i을 수정 없이 network length (L_i), width (C_i), and/or resolution (H_i, W_i)을 확장 한다. F_i를 고정함으로써 model scaling은 새로운 resource 제약들을 위한 design problem을 간략화 한다. 그러나 여전히 각 layer의 각기 다른 L_i, C_i, H_i, W_i를 탐험할 큰 design space는 남는다. 더 design space를 줄이기 위해 모든 layer가 일정한 비율로 균일하게 확장되어야 한다고 제한한다. 목표는 최적화 문제로 공식화될 수 있는 주어진 resource 제약에 대한 모델 정확도를 최대화 하는 것 이다.

    ![image](https://user-images.githubusercontent.com/92927837/147875738-d117f91c-a427-4be5-a17c-8098bcbbd0b8.png)

    w, d, r은 network width, depth, and resolution scaling의 coefficients이다. F_i, L_i, H_i, W_i, C_i는 baseline network에서 미리 정의된 파라미터이다. 

* Compound Scaling

  * 네트워크 너비, 깊이 또는 해상도의 모든 차원을 확장하면 정확도가 향상되지만 더 큰 모델에서는 정확도 이득이 감소합니다. 더 나은 정확도와 효율성을 추구하려면 ConvNet 확장 중에 네트워크 너비, 깊이 및 해상도의 모든 차원의 균형을 유지하는 것이 중요합니다. 이전에도 이러한 시도가 있었으나 manual tuning  이었다. 이 논문에서는 복합 계수 φ를 사용하여 원칙적으로 네트워크 너비, 깊이 및 해상도를 균일하게 확장하는 새로운 복합 스케일링 방법을 제안합니다.

    ![image](https://user-images.githubusercontent.com/92927837/147876178-0b1b373f-e1b3-4ef6-a47c-4fbb7ab75106.png)

    여기서 α, β, γ는 small grid search로 결정할 수 있는 상수이다. 직관적으로 **φ는 model scaling에 사용할 수 있는 추가 resources를 제어하는 사용자 지정 계수이고 α, β, γ는 이러한 추가 resources를 network width, depth, and resolution에 각각 할당하는 방법을 지정 한다.** 특히, 일반 컨볼루션 연산의 FLOPS는 d, w^2, r^2에 비례 한다. 즉, 네트워크 깊이를 두 배로 늘리면 FLOPS가 두 배가 되지만 네트워크 너비 또는 해상도가 두 배가 되면 FLOPS가 4배 증가 한다.

    컨볼루션 연산은 일반적으로 ConvNet의 계산 비용을 지배하므로 등식 3으로 ConvNet을 확장하면 총 FLOPS가 대략 (α · β^2 · γ^2)^φ만큼 증가 한다. 이 논문에서 우리는 α · β2 · γ2 ≈ 2를 제한하여 새로운 φ에 대해 총 FLOPS가 대략 2^φ만큼 증가할 것이다.

* EfficientNet의 b0는 NAS를 통해 찾아낸 baseline이며 b1 ~ 7은 각각 φ 값이다. 각 φ 값에 따른 α, β, γ를 위 (2), (3) 식을 통해 찾아내어서 최적화한다.

* 추가 검색 필요
  * NAS - Neural architecture search 란 ?
  * receptive field 란 ?











### Reference

* https://deep-learning-study.tistory.com/ 
* [ImageNet Classification with Deep Convolutional Neural Networks (neurips.cc)](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) - AlexNet paper
* [1409.4842.pdf (arxiv.org)](https://arxiv.org/pdf/1409.4842.pdf) - GoogLeNet paper
* [1409.1556.pdf (arxiv.org)](https://arxiv.org/pdf/1409.1556.pdf) - VGGNet paper
* https://arxiv.org/pdf/1512.03385.pdf - ResNet paper
* https://computistics.tistory.com/3?category=866107 
* https://arxiv.org/pdf/1608.06993.pdf - DenseNet paper
* https://arxiv.org/pdf/1704.04861.pdf - MobileNet paper
* [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks (arxiv.org)](https://arxiv.org/pdf/1905.11946.pdf)
* [Depth-wise Convolution and Depth-wise Separable Convolution | by Atul Pandey | Medium](https://medium.com/@zurister/depth-wise-convolution-and-depth-wise-separable-convolution-37346565d4ec)
* https://arxiv.org/pdf/1905.11946.pdf - EfficientNet
* https://hoya012.github.io/blog/EfficientNet-review/

