##  PetFinder.my - Pawpularity Contest
* https://www.kaggle.com/c/petfinder-pawpularity-score/overview



### Time line

* **September 23, 2021** - Start Date.
* **January 6, 2022** - Entry Deadline. You must accept the competition rules before this date in order to compete.
* **January 6, 2022** - Team Merger Deadline. This is the last day participants may join or merge teams.
* **January 13, 2022** - Final Submission Deadline.





### Description

* In this competition, you’ll analyze raw images and metadata to **predict the “Pawpularity” of pet photos**. You'll train and test your model on PetFinder.my's thousands of pet profiles. Winning versions will offer accurate recommendations that will improve animal welfare.

  If successful, your solution will be adapted into AI tools that will guide shelters and rescuers around the world to improve the appeal of their pet profiles, automatically enhancing photo quality and recommending composition improvements. As a result, stray dogs and cats can find their "furever" homes much faster. With a little assistance from the Kaggle community, many precious lives could be saved and more happy families created..

* Evaluation: root mean squared error
* Data overview
  * Training Data
    * train/ - Folder containing training set photos of the form {id}.jpg, where {id} is a unique Pet Profile ID.
    * train.csv - Metadata (described below) for each photo in the training set as well as the target, the photo's Pawpularity score. The Id column gives the photo's unique Pet Profile ID corresponding the photo's file name.
  * Example Test Data
    * In addition to the training data, we include some randomly generated example test data to help you author submission code. When your submitted notebook is scored, this example data will be replaced by the actual test data (including the sample submission).
    * test/ - Folder containing randomly generated images in a format similar to the training set photos. The actual test data comprises about 6800 pet photos similar to the training set photos.
    * test.csv - Randomly generated metadata similar to the training set metadata.
    * sample_submission.csv - A sample submission file in the correct format.
  * Photo Metadata
    * The train.csv and test.csv files contain metadata for photos in the training set and test set, respectively. Each pet photo is labeled with the value of 1 (Yes) or 0 (No) for each of the following features:
    * Focus - Pet stands out against uncluttered background, not too close / far.
    * Eyes - Both eyes are facing front or near-front, with at least 1 eye / pupil decently clear.
    * Face - Decently clear face, facing front or near-front. Near - Single pet taking up significant portion of photo (roughly over 50% of photo width or height).
    * Action - Pet in the middle of an action (e.g., jumping).
    * Accessory - Accompanying physical or digital accessory / prop (i.e. toy, digital sticker), excluding collar and leash.
    * Group - More than 1 pet in the photo.
    * Collage - Digitally-retouched photo (i.e. with digital photo frame, combination of multiple photos).
    * Human - Human in the photo.
    * Occlusion - Specific undesirable objects blocking part of the pet (i.e. human, cage or fence). Note that not all blocking objects are considered occlusion.
    * Info - Custom-added text or labels (i.e. pet name, description).
    * Blur - Noticeably out of focus or noisy, especially for the pet’s eyes and face. For Blur entries, “Eyes” column is always set to 0.



### Model

![model](https://user-images.githubusercontent.com/92927837/150665796-3d24af1a-534b-4181-ab72-a0f4379984c3.png)



### Result

![result](https://user-images.githubusercontent.com/92927837/150665770-0b397e26-b4bb-4192-9394-b13fbfc04b15.png)



### Reference

* https://www.kaggle.com/ishandutta/petfinder-data-augmentations-master-notebook
* https://www.kaggle.com/subinium/petfinder-i-am-featurefinder-eda-notebook
* https://www.kaggle.com/cdeotte/rapids-svr-boost-17-8

