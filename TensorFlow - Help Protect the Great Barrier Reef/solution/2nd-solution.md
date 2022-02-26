## 2nd place solution

### Summary

There are three keys of this competition, **train/val split**, **resolution**, and **luck**.



**split - by video_id**

In the early of this competition, we saw that there is a very solid way to split the dataset called "subsequence". After some submissions, the LB is not stable enough, so in the end we use video_id 3 fold split, because in that time, we guess that the LB and private maybe totally different video sequences, besides, score of the CV is now almost positive correlation with the LB.



**preprocessing - rotation is the key**

augmentations that works, inherited from YOLOv5 default augmentation, we added:

* mosaic prob@0.25
* mixup prob@0.25
* random rotate90

the random rotate90 will boost ~ 0.02 with both CV and LB, and it also benefits ensembling with different rotation.



**training**

* use only YOLOv5l6 model for speed and performance balancing
* training with images that has ground truth only
* multiscaling +- 50%
  * The YOLOv5 has builtin multiscaling switch, from what i know, scaling matters a lot in detection especially for anchor based detection methods, so we just enable it.



**postprocessing/tracking**

postprocessing is focusing on tracking. Instead of using existing tracking methods (which is not working), we use a more simple method, which I called "attention area". It is working as follows:

1. the model has predicted some boxes B at N frame
2. select the boxes from B which has a high confidence (larger than a threshold T), these boxes are marked as "attention area"
3. in the N + 1 frame, all the predicted boxes (not actually all, but confidence threshold > 0.01), if it has an IoU with the "attention area" larger than 0.5, boost the score of these boxes with S, then do the confidence filtering. The T=0.15 and S=0.1 is a solid choice, this method boost both CV and LB by ~ 0.01.





**ensembling**

we know that one model inference with different resolution will have a huge difference, from our experiments, single model trained with 2400 resolution and inference with 4800 resolution will get the highest LB ~ 0.73, we should be careful here, cause the CV is not true.

all the evidences show that resolution matters a lot, high resolution benefit LB, but not CV, we are not sure if the private dataset has the same distribution as the LB, or just part of it, so chosing the selections is hard, luckly this time we got 4 chances. Another observation is that ensembling model numbers will benefit WBF, so we decide to take the total 9 hours inference quota as much as we can. Our winner solution, private 0.737 public 0.648, is ensembled of the following models/resolutions, 4 models inference with 11 resolutions





## Reference

* https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/307760
* https://github.com/louis-she/reef-solution
* https://www.kaggle.com/snaker/yolo-ensemle?scriptVersionId=87785527







