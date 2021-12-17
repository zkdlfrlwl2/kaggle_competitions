## YOLOX: Exceeding YOLO Series in 2021



![Figure1](https://user-images.githubusercontent.com/92927837/146490762-960ca92b-cffd-4f40-95a0-6ab2b04bf8f5.PNG)



### Abstract

기존 YOLO 비해 성능이 좋다. YOLOX-L single model을 사용해서 Streaming Perception Challenge (Workshop on Autonomous Driving at CVPR 2021)에서 1등을 차지했다. ONNX, TensorRT, NCNN, Openvino의 deploy 버전을 github repo에서 제공한다.



### 1. Introduction

YOLO 시리즈는 real-time applications을 위해 최적의 속도와 정확도 사이의 trade-off를 추구해왔다. 현재, YOLOv5는 COCO에서 48.2% AP와 13.7ms 성능을 가지고 있다. 지난 2년 동안 객체 탐지 학회는 **anchor-free detectors, advanced label assignment strategies, and end-to-end (NMS-free) detectors**에 집중해왔으나 YOLO 시리즈에 통합하지 못했고 YOLOv4 & v5는 여전히 **anchor-base detectors with hand-crafted assigning rules**를 사용한다. 

연산 자원의 제한과 불충분한 소프트웨어 지원때문에 아직도 현업에서 YOLOv3를 많이 사용 한다. 그리고 YOLOv4 & v5는 **anchor-based pipeline**에 과도하게 최적화될 우려가 있어서 시작점으로 VOLOv3 (YOLOv3-SPP as the default YOLOv3) 를 선택했다.



### 2. YOLOX

#### 2.1 YOLOX-DarkNet53

YOLOv3 with DarkNet53을 우리의 baseline으로 선택했다.



#### Implementation details

* 300 epochs with 5 epochs warm-up
* SGD
  * weight decay = 0.0005
  * momentum = 0.9
* learning rate = lr x BatchSize/64 (linear scaling) 
  * with a initial lr=0.01 and the cosine lr schedule.
* BatchSize = 128



<img src="https://user-images.githubusercontent.com/92927837/146494710-ac23db81-a714-47dc-bbf4-45a1e58544bf.PNG" alt="Table1" style="zoom:80%;" />



#### YOLOv3 baseline

* 기본 모델 구조
  * DarkNet53 backbone and an SPP layer, referred to YOLOv3-SPP
* 학습 전략 변경점
  * adding EMA weights updating
  * cosine lr schedule
  * IoU loss and IoU-aware branch
  * use BCE loss for training *cls* and *obj* branch
  * IoU loss for training *reg* branch
  * Augmentation
    * RandomHorizontalFlip, ColorJitter, multi-scale
    * discard the RandomResizedCrop because found that is kind of overlapped with the planned mosaic augmentation
* 결과
  * 38.5% AP on COCO val 달성



#### Decoupled head

객체 탐지에서 분류와 회귀의 충돌은 잘 알려진 문제이다. 그러므로 대부분의 one-stage and two-stage detectors에서 classification과 localization를 위해 decoupled (분리된) head를 사용 한다. 하지만 YOLO 시리즈의 backbones 및 feature pyramids (e.g. FPN, PAN)가 계속해서 진화하면서 그들의 head는 Fig. 2에서 볼 수 있듯이 coupled head로 남아있다. 

두 개의 분석 실험은 coupled direction head가 성능을 해칠 수도 있다고 나타낸다. 1). YOLO head를 decoupled head로 교체하니 Fig.3에서 보는 바와 같이 수렴 속도가 크게 개선되었다. 2).  Tab.1에서 decoupled를 적용하면 0.8% AP가 줄어들지만 coupled head는 4.2% AP가 줄어드는 걸 확인 할 수 있으므로 decoupled head는 end-to-end YOLO에 필수적이다.



![Figure2](https://user-images.githubusercontent.com/92927837/146499346-28110e9d-b8c9-41c3-aaf1-c02bf923cef1.PNG)



<img src="https://user-images.githubusercontent.com/92927837/146500030-60426579-7712-4483-84b4-1f5b17bae8db.PNG" alt="Figure3" style="zoom:67%;" />



#### Strong data augmentation

* Add Mosaic and MixUp into our augmentation strategies to boost YOLOX's performance
* Mosaic is an efficient augmentation strategy proposed by ultralytics-YOLOv3. It is widely used in YOLOv4 & v5 and other detectors.
* Mixup is originally designed for image classification task but modified in BoF for object detection training.
* close data augmentation for the last 15 epochs, achieving 42.0% AP in Tab.2
* After using strong data augmentation, found that ImageNet pre-training is no mode beneficial thus train all the following models from scratch.



#### Anchor-free

YOLOv3 & v4 & v5 모두 anchor-based pipeline을 따른다. 하지만 anchor mechanism에는 알려진 많은 문제가 있다. 첫째, 최적의 탐지 성능을 얻기 위해 학습 전, 최적의 anchors set을 결정하기 위해 clustering analysis를 수행할 필요가 있다. 이 clustered anchors는 domain specific이며 less generalized 이다. 

둘째, anchor mechanism은 각 이미지에 대한 number of predictions 뿐만 아니라 detection head의 복잡성도 증가 시킨다. 

Anchor-free mechanism은 우수한 성능을 위해 heuristic tuning과 관련된 많은 트릭 (e.g. Anchor Clustering, Grid Sensitive) 이 필요한 설계 매개변수의 수를 크게 줄여 detector, 특히 training 및 decoding phase를 상당히 단순하게 만든다.



![Table2](https://user-images.githubusercontent.com/92927837/146501136-9849e565-19c3-4be2-b554-eb6096a4f5bc.PNG)



#### Multi positives

YOLOv3의 할당 규칙과 일치하기 위해 위의 anchor free version은 각 개체에 대해 하나의 긍정적인 샘플(중앙 위치)만 선택하는 반면 다른 고품질 예측은 무시한다. 그러나 이러한 고품질 예측을 최적화하면 유익한 기울기를 가져올 수 있으며, 이는 훈련 중 양수/음수 샘플링의 극심한 불균형을 완화할 수 있다. FCOS에서 "센터 샘플링"이라고도 하는 센터 3×3 영역을 양수로 지정하기만 하면 된다.



#### SimOTA

Advanced label assignment는 최근 몇 년 동안 object detection의 중요한 발전이다. 자체 연구 OTA를 기반으로 advanced label assignment에 대한 4 가지 주요 insights를 결론지었다.

* loss/quality aware
* center prior
* dynamic number of positive anchors for each ground-truth (abbreviated as dynamic top k)
* global view

OTA는 위의 네 가지 규칙을 모두 충족하므로 이를 후보 레이블 할당 전략으로 선택 한다. 특히, OTA는 글로벌 관점에서 레이블 할당을 분석하고 할당 절차를 OT(Optimal Transport) 문제로 공식화하여 현재 할당 전략 중 SOTA 성능을 생성 한다. 그러나 실제로 Sinkhorn-Knopp 알고리즘을 통해 OT 문제를 해결하면 25%의 추가 훈련 시간이 발생하며 이는 300개의 Epoch를 훈련하는 데 상당히 비싸다. 따라서 SimOTA라는 동적 top-k 전략으로 단순화하여 대략적인 솔루션을 얻는다.

 

![temp](https://user-images.githubusercontent.com/92927837/146511601-23caeec1-5e6c-4ac3-a93d-f4e38719dcf1.PNG)



### List

* Anchor Box
  * 각각의 grid는 오직 한 개의 object만을 detect 할 수 있는 한계를 극복하기 위해 나온 개념
  * Anchor box를 여러 개 적용한 경우, 각각의 object는 grid 중 Anchor box와 가장 큰 IoU를 가지는 grid에 object가 할당
  * 단점
    * 한 gird 내에 Anchor box의 갯수보다 많은 object가 있으면 모든 object를 검출하기 어렵다
    * 같은 Anchor box에 2개 이상의 object가 있으면 모든 object를 검출하기 어렵다  
* advanced label assignment strategies
* and end-to-end (NMS-free) detector
* YOLOv3
* feature pyramids (e.g. FPN, PAN)
* Multi positives ? 
  * positive anchor
* OTA





### Reference

* https://github.com/Megvii-BaseDetection/YOLOX
* https://arxiv.org/abs/2107.08430
* https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=infoefficien&logNo=221229808775