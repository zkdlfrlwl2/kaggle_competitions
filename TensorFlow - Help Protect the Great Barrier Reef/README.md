##  TensorFlow - Help Protect the Great Barrier Reef


### Time line

* **November 22, 2021** - Start Date.
* **February 14, 2022** - Final Submission Deadline.



### Description

* **Goal of the Competition**

  * The goal of this competitions is to accurately identify starfish in real-time by building ad object detection model trained on underwater videos of coral reefs. Your work will help researchers identify species that are threatening Australia's Great Barrier Reef and take well-informed action to protect the reef for future generations.

* **Context**

  * To know where the COTS are, a traditional reef survey method, called "Manta Tow", is performed by a snorkel diver. While towed by a boat, they visually assess the reef, stopping to record variables observed every 200m. While generally effective, this method faces clear limitations, including operational scalability, data resolution, reliability, and traceability.

    The Great Barrier Reef Foundation established an [innovation program](https://www.barrierreef.org/what-we-do/reef-trust-partnership/crown-of-thorns-starfish-control) to develop new survey and intervention methods to provide a step change in COTS Control. Underwater cameras will collect thousands of reef images and AI technology could drastically improve the efficiency and scale at which reef managers detect and control COTS outbreaks.

    To scale up video-based surveying systems, Australia’s national science agency, CSIRO has teamed up with Google to develop innovative machine learning technology that can analyse large image datasets accurately, efficiently, and in near real-time.

* **Evaluation** 

  * F2 Score at different intersection over union (IoU) thresholds

* **Data Description**

  * In this competition, you will **predict the presence and position of crown-of-thorns starfish** in sequences of underwater images taken at various times and locations around the Great Barrier Reef. **Predictions take the form of a bounding box together with a confidence score for each identified starfish**. An image **may contain zero or more starfish**.

    This competition uses a hidden test set that will be served by an API to ensure you evaluate the images in the same order they were recorded within each video. When your submitted notebook is scored, the actual test data (including a sample submission) will be availabe to your notebook.

  * **File**

    * **train/** - Folder containing training set photos of the form **video_{video_id}/{video_frame_number}.jpg**.
    * **[train/test].csv** - Metadata for the images. As with other test files, most of the test metadata data is only available to your notebook upon submission. Just the first few rows available for download.
      * `video_id` - ID number of the video the image was part of. The video ids are not meaningfully ordered.
      * `video_frame` - The frame number of the image within the video. Expect to see occasional gaps in the frame number from when the diver surfaced.
      * `sequence` - ID of a gap-free subset of a given video. The sequence ids are not meaningfully ordered.
      * `sequence_frame` - The frame number within a given sequence.
      * `image_id` - ID code for the image, in the format '{video_id}-{video_frame}'
      * `annotations` - The bounding boxes of any starfish detections in a string format that can be evaluated directly with Python. Does not use the same format as the predictions you will submit. Not available in **test.csv.** A bounding box is described by the pixel coordinate (x_min, y_min) of its upper left corner within the image together with its width and height in pixels.
    * **example_sample_submission.csv** - A sample submission file in the correct format. The actual sample submission will be provided by the API; this is only provided to illustrate how to properly format predictions. The submission format is further described on the [Evaluation page](https://www.kaggle.com/c/tensorflow-great-barrier-reef/overview/evaluation).
    * **example_test.npy** - Sample data that will be served by the example API.
    * **greatbarrierreef** - The image delivery API that will serve the test set pixel arrays. You may need Python 3.7 and a Linux environment to run the example offline without errors.

  * **Time-series API Details**

    * The API serves the images one by one, in order by video and frame number, as pixel arrays.
    * Expect to see roughly 13,000 images in the test set.
    * The API will require roughly two GB of memory after initialization. The initialization step (`env.iter_test()`) will require meaningfully more memory than that; we recommend you do not load your model until after making that call. The API will also consume less than ten minutes of runtime for loading and serving the data.



### Result

![image](https://user-images.githubusercontent.com/92927837/154830795-c95e78bb-f21c-4673-919d-7f32be54ed82.png)





### Reference

* https://www.kaggle.com/c/tensorflow-great-barrier-reef
