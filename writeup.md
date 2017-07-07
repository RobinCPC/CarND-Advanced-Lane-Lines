# **Advanced Lane Finding Project**

## The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistorted_images.png   " Undistorted      "
[image2]: ./output_images/undistorted_example1.png " Road Transformed "
[image3]: ./output_images/binary_example1.png      " Binary Example   "
[image4]: ./output_images/warped_images.png        " Warp Example     "
[image5]: ./output_images/visualized_result.png    " Output           "
[image6]: ./output_images/threshold_gui.png        " Threshold GUI    "
[image7]: ./output_images/detect_example.png       " Detect Visual    "
[image8]: ./output_images/fail_example1.png        " Fail Detection1  "
[image9]: ./output_images/fail_example2.png        " Fail Detection2  "
[video1]: ./output_images/project_output.mp4       " Video            "

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
## [Source Code](https://github.com/RobinCPC/CarND-Advanced-Lane-Lines)

How to use:

    For Calibration, please run :
        python main.py -c 1
    For testing pipeline, please run:
        python main.py
    For detecting lane lines in video, please run:
        python main.py -v 1
    Pleae type `python main.py --help`, for more detail.

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines #34 through #92 of the file called `main.py`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. Then, I use python module `pickle` to save these parameters in `camera_cal/wide_dist_pickle.p`.

I then used the output `objpoints` and `imgpoints` to compute the camera intrinsic matrix and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I use load `./camera_cal/wide_dist_pickle.p` to get the parameters of calibration and use `cv2.undistort()` function to get the distortion-corrected image (as below):

![alt text][image2]

From above image, we can tell that the bottom of images have obvious difference. The dashboard in the left image (distorted) get smaller in the right image (undistorted).

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines #95 through #126 in `main.py`).  Here's an example of my output for this step.

![alt text][image3]

In addition, in order to tune thresholds of color & gradient more easily, I create a GUI (a file called `threshold_tuning.py`) to tune thresholds with slide bars and I could see the result immediately (as below figure). The GUI help me find the suitable thresholds (For Color `S` channel: 60 to 250; for Gradient `sobel_x`: 30 to 60).

![alt text][image6]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `get_warped()`, which appears in lines 130 through 139 in the file `main.py` (./main.py).  The `get_warped()` function takes source (`src`) and destination (`dst`) points as inputs to compute the perspective matrix for getting "bird-eyes view".  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([[584, 455], [700, 455], [1052, 675], [268, 675]])
dst = np.float32([[320, 0], [960, 0], [960, 720], [320, 720]])
```

This resulted in the following source and destination points:

| Source          | Destination     |
| :-------------: | :-------------: |
| 584  , 455      | 320 , 0         |
| 700  , 455      | 960 , 0         |
| 1052 , 675      | 960 , 720       |
| 268  , 675      | 320 , 720       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

After I create the threshold binary image and transform it to bird-eyes (top) view, I use a function called `sliding_window_search()` and `filter_search()`,which appears in lines 34 through 106 in the file `detect_lanes.py` (./detect_lanes.py), to detect the lane lines in each image (as show in bottom right of the below figure).

![alt text][image7]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

After detecting lane lines in each image, I use `measure_curvature()` function in lines #146 through #180 in `main.py` to computer the curvature of left and right lanes. To computer the position of the vehicle with respect to center of lane line, I use following codes:

```python
mid_car = 640.0         # middle point of car dashboard
xm_per_pix = 3.7/700    # meters per pixel in x dimension
base_pos = ((right_fitx[-1] + left_fitx[-1])/2. - mid_car) * xm_per_pix     # unit: m
```

`right_fitx[-1]` is the closest point to dashboard in the polynomial line of right lane, and `left_fitx[-1]` is the closest point of the left lane. Therefore, in the above code, I compute the middle point of two lane line and comapre with center of camera, at 640 pixel, and times `3.7/700`, the scale to map pixel back to meter.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

After getting the polynomial equation of both lane line, I implemented the visualization in lines #205 through #255 in my code in `main.py` in the function `visualize_output()`.  Here is an example of my result on a test image:

![alt text][image5]

---

### Pipeline (video)

#### 1. Building pipeline of video for processing each frame.
For this step, I use `VideoFileClip()` function from moviepy library, in lines #400 to lines #417 in `main.py`, to execute my pipeline on each frame.

#### 2. Sanity Check.
Because of quality of each frame or evironment changes in each frame, `sliding_window_search` function may detect the wrong lane line (such as two line are not roughly parallel). Therefore, I use `check_sanity` function, in lines #268 to #286 in `main.py`, to check if two line are roughly parallel or have suitable curvature. If `check_sanity` return the result is not correct, and I will use `filter_search()` functioni, to detect lane lines again. `filter_search()` use previous detected result, from previous frame, to only search possible area in the image. The whole flow of sanity-checked is in lines #339 to lines #373 in `main.py`.

#### 3. Smoothing
For robust and avoiding the visualization of detect lines jump around from frame to frame, I use objects of `Line` class, in `detect_line.py`, to storage curvatures each frame of both lines, and use average curvature to draw lane line in output images. `Line` object is also useful to record other information for checking sanity.

#### 4. Provide a link to your final video output.
Here's a [link to my video result](https://nbviewer.jupyter.org/github/RobinCPC/CarND-Advanced-Lane-Lines/blob/master/project_output.ipynb)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Before I check sanity of the result of lane lines, my pipeline of processing video will fail when passing through the shadow of trees (21-25 sec, and 39-42 sec) and a black car passing (27-32 sec). Threfore, I use `check_sanity()` to get rid of bad detection.

#### 2. What could be improved about the algorithm/pipeline?

I believe `sliding_window_search()` still could be imporved. When sliding_window_search do convolute with mask and can not detect anything (which means whole search section are black), it will output the center area of search section. From below figure, we could see that the forth section of right detected lane output the center position, but there is no bright pixel overlap in that section.

![alt text][image8]

To imporve sliding window search, from below code segment, we could add a check function before appending l_center/r_center into window_centroids. If there is no bright pixel overlap in ouput detected area, we should append previous l_center/r_center (not the center area of search section).

```python
# Sum quarter bottom of image to get slice, could use a different ratio
l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)

# Add what we found for the first layer
window_centroids.append((l_center,r_center))
```

#### 3. What hypothetical cases would cause the pipeline to fail?

When there is a sharp turned road and very bright environment as below figure (such as 41 second in harder_challenge_video.mp4), the recording image are very hard for pipeline to process. Also, from the above code snippet, slide window search will split image in half and do searching. but the sharp-turned lane cross the middle point of image.

![alt text][image9]

