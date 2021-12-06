##  SIIM-FISABIO-RSNA COVID-19 Detection
* https://www.kaggle.com/c/siim-covid19-detection

### Time line

* May 17, 2021 - Start Date.
* August 9, 2021 - Final Submission Deadline.



### Model

* Efficientnetb7

  * Data augmentation
    * tf.image.random_flip_up_down
    * tf.image.random_flip_left_right
    * Mixup
    * Cutmix
  * Pseudo labeling

  * lr schedule
    * LR ramp up

* yolov5

* Weighted ensemble

  

### Result

![image](https://user-images.githubusercontent.com/92927837/141050711-4105cfa2-28ac-4a0f-912f-fac5bc30e187.png)





* Reference

  * https://www.kaggle.com/allunia/pulmonary-dicom-preprocessing
  * https://www.kaggle.com/xhlulu/siim-covid-19-convert-to-jpg-256px
  * https://www.kaggle.com/h053473666/siim-covid19-efnb7-train-study
  * https://www.kaggle.com/h053473666/siim-cov19-yolov5-train
  * https://www.kaggle.com/h053473666/siim-covid19-efnb7-train-fold0-5-2class
  * https://www.kaggle.com/h053473666/siim-cov19-efnb7-yolov5-infer

