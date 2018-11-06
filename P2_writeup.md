## Write up for Advanced Lane Finding Project



### Written by Bo Li on Oct 21st, 2018


---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/comparison_original_undistort.png "Undistorted"
[image2]: ./output_images/undistorted_image.png "Road Transformed"
[image3]: ./output_images/comparison_different_color_transforms.png "Color_transform Example"
[image4]: ./output_images/comparison_raw_combined_threshold.png "raw_binary_comparison"
[image5]: ./output_images/comparison_undistorted_unwarped.png "undistorted_unwarped_comparison"
[image6]: ./output_images/polynomial_fitted_lanes.png "Polynomial_fit"
[image7]: ./output_images/draw_back_radius_distance.png "Radius_Distance"


[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

<!-- #### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it! -->

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

There are two main steps to this process: use chessboard images to obtain image points and object points, and then use the OpenCV functions cv2.calibrateCamera() and cv2.undistort() to compute the calibration and undistortion.


The code for this step is contained in the 3rd and the 8th code cell of the IPython notebook located in "./P2.ipynb". The keypoints in the code are listed as the folowing:

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

* 3rd Cell
  

```python
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')


# Step through the list and search for chessboard corners
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)

```

* 8th cell
  
```python
# We know that the number of chessboard image is 20, 
# so we can give an 5*4 array of pictures enlisting all the pictures,
# some of these images do not appear because 
# the specified number of chessboard corners were not found.
fig, axs = plt.subplots(5, 4, figsize = (16, 11))
fig.subplots_adjust(hspace = .2, wspace = .001)
axs = axs.ravel()

# Step through the list and search for the chessboard corners

for i, fname in enumerate(filenames):
    img = cv2.imread(path+cali_input+fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
    
    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        # Once we find the corners, we can increase their accuracy using cv2.cornerSubPix().
        # http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),criteria)
        imgpoints.append(corners2)
        
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        cv2.imwrite(path + output_chessboard +fname, img)
        axs[i].axis('off')
        axs[i].imshow(img)

```

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

The inplementation code of the result above is written as the following.
```python
# Test undistortion on an image
img = cv2.imread(path + cali_input +filenames[0])
img_size = (img.shape[1], img.shape[0])

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
dst = cv2.undistort(img, mtx, dist, None, mtx)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "calibration.p", "wb" ) )
#dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
f.subplots_adjust(hspace = .2, wspace=.05)
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)
print('Comparison between the original image and the undistorted image')


```




### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

I make use of the camera distortion matrix (`mtx`) and dist(`dist`) to undistort the raw lane pictures. By filling the parameters `mtx` and `dist`  in the `cv2.undistort()` function as follows, we can tranfer the raw image into distortion-corrected image.

```python
def undistort(img):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

exampleImg = cv2.imread(path+tst_input+filenames[4])
exampleImg_undistort = undistort(exampleImg)


```

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

In this section, I take three steps to implement the binary image.

For the first step, I make use RGB channel, HSV channel as well as the LAB channel to give the different imapct on the color transforms, in order to get some more useful intuition. The comparison of the transformed images is given as the following array of images.

![alt text][image3]

The visualization code for this step is in the following manner

```python
# Visualize multiple color space channels
exampleImg_unwarp_R = exampleImg_unwarp[:,:,0]
exampleImg_unwarp_G = exampleImg_unwarp[:,:,1]
exampleImg_unwarp_B = exampleImg_unwarp[:,:,2]
exampleImg_unwarp_HSV = cv2.cvtColor(exampleImg_unwarp, cv2.COLOR_RGB2HSV)
exampleImg_unwarp_H = exampleImg_unwarp_HSV[:,:,0]
exampleImg_unwarp_S = exampleImg_unwarp_HSV[:,:,1]
exampleImg_unwarp_V = exampleImg_unwarp_HSV[:,:,2]
exampleImg_unwarp_LAB = cv2.cvtColor(exampleImg_unwarp, cv2.COLOR_RGB2Lab)
exampleImg_unwarp_L = exampleImg_unwarp_LAB[:,:,0]
exampleImg_unwarp_A = exampleImg_unwarp_LAB[:,:,1]
exampleImg_unwarp_B2 = exampleImg_unwarp_LAB[:,:,2]
fig, axs = plt.subplots(3,3, figsize=(16, 12))
fig.subplots_adjust(hspace = .2, wspace=.001)
axs = axs.ravel()
axs[0].imshow(exampleImg_unwarp_R, cmap='gray')
cv2.imwrite(path+output_color+'unwarp_R.jpg', exampleImg_unwarp_R)
axs[0].set_title('RGB R-channel', fontsize=30)
axs[1].imshow(exampleImg_unwarp_G, cmap='gray')
cv2.imwrite(path+output_color+'unwarp_G.jpg', exampleImg_unwarp_G)
axs[1].set_title('RGB G-Channel', fontsize=30)
axs[2].imshow(exampleImg_unwarp_B, cmap='gray')
cv2.imwrite(path+output_color+'unwarp_B.jpg', exampleImg_unwarp_B)
axs[2].set_title('RGB B-channel', fontsize=30)
axs[3].imshow(exampleImg_unwarp_H, cmap='gray')
cv2.imwrite(path+output_color+'unwarp_H.jpg', exampleImg_unwarp_H)
axs[3].set_title('HSV H-Channel', fontsize=30)
axs[4].imshow(exampleImg_unwarp_S, cmap='gray')
cv2.imwrite(path+output_color+'unwarp_S.jpg', exampleImg_unwarp_S)
axs[4].set_title('HSV S-channel', fontsize=30)
axs[5].imshow(exampleImg_unwarp_V, cmap='gray')
cv2.imwrite(path+output_color+'unwarp_V.jpg', exampleImg_unwarp_V)
axs[5].set_title('HSV V-Channel', fontsize=30)
axs[6].imshow(exampleImg_unwarp_L, cmap='gray')
cv2.imwrite(path+output_color+'unwarp_L.jpg', exampleImg_unwarp_L)
axs[6].set_title('LAB L-channel', fontsize=30)
axs[7].imshow(exampleImg_unwarp_A, cmap='gray')
cv2.imwrite(path+output_color+'unwarp_A.jpg', exampleImg_unwarp_A)
axs[7].set_title('LAB A-Channel', fontsize=30)
axs[8].imshow(exampleImg_unwarp_B2, cmap='gray')
cv2.imwrite(path+output_color+'unwarp_B2.jpg', exampleImg_unwarp_B2)
axs[8].set_title('LAB B-Channel', fontsize=30)
print('Comparison between the binary pictures under different color transform method')

```

For the second step, the following gradient methods are attemped sequentially, namely, the Sobel Manitude Threshold (the 20th code cell of the IPython notebook located in "./P2.ipynb"), the Sobel Direction Threshold (the 21st code cell of the IPython notebook located in "./P2.ipynb"), HLS-Colorspace(the 22nd and 23rd code cell of the IPython notebook located in "./P2.ipynb"),
Lab-Colorspace(the 24th code cell of  "./P2.ipynb").

I used a combination of HLS-L-Channel and LAB-B-Channel to generate a binary image. The code is located at the 24th code cell of  "./P2.ipynb", as the following:

```python
def combined_thresh(img, thresh1 = (220, 255), thresh2 = (190, 255)):
    img_LThresh = hls_lthresh(img, thresh = thresh1)
#     exampleImg_SThresh = hls_sthresh(exampleImg_unwarp, thresh = thresh1)
    img_BThresh = lab_bthresh(img, thresh = thresh2)
    combined = np.zeros_like(exampleImg_LThresh)
    combined[(( img_LThresh == 1) | (img_BThresh == 1))] = 1
    return combined
#   & (exampleImg_sobelDir2 == 1) 
exampleImg_combined = combined_thresh(exampleImg_unwarp)
# cv2.imwrite(path+output_combined+'combined.jpg', exampleImg_combined)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
f.subplots_adjust(hspace = .2, wspace=.05)
ax1.imshow(exampleImg_unwarp)
ax1.set_title('Unwarped Image', fontsize=30)
ax2.imshow(exampleImg_combined, cmap='gray')
ax2.set_title('HLS-L-Channel + Lab-B-Channel', fontsize=30)
```

Here's an example of my output for this step. 

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `unwarp()`, which appears in lines 1 through 8 in the file in the 16th code cell of the IPython notebook).  The `unwarp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
h,w = exampleImg_undistort.shape[:2]

# define source and destination points for transform
src = np.float32([(575,464),
                  (707,464), 
                  (258,682), 
                  (1049,682)])
dst = np.float32([(450,0),
                  (w-450,0),
                  (450,h),
                  (w-450,h)])

```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 575, 464      | 450, 0        |
| 707, 464      | 830, 0      |
| 258, 682     | 450, 720      |
| 1049, 682      | 830, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I define method to fit polynomial to binary image with the lane lines detected by sliding windows and identify the number of nonzero pixels in the windows and recenter the window if the number is over some threshold. The left and right line pixel positions are extracted in this way which is what the second order polynomial is fitted to.The implementation code for the positions fitted is defined as sliding_window_polyfit() and polyfit_using_prev_fit() function located at the 29th cell and the the 32nd cell in the Jupyter Notebook "P2.ipynb", as the following:

* 29th cell
  
```python
# Define method to fit polynomial to binary image with lines extracted, using sliding window
def sliding_window_polyfit(img):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    quarter_point = np.int(midpoint//2)
    # Previously the left/right base was the max of the left/right half of the histogram
    # this changes it so that only a quarter of the histogram (directly to the left/right) is considered
    leftx_base = np.argmax(histogram[quarter_point:midpoint]) + quarter_point
    rightx_base = np.argmax(histogram[midpoint:(midpoint+quarter_point)]) + midpoint
    
    #print('base pts:', leftx_base, rightx_base)

    # Choose the number of sliding windows
    nwindows = 10
    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 80
    # Set minimum number of pixels found to recenter window
    minpix = 40
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Rectangle data for visualization
    rectangle_data = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        rectangle_data.append((win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high))
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    left_fit, right_fit = (None, None)
    # Fit a second order polynomial to each
    if len(leftx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2)
    
    visualization_data = (rectangle_data, histogram)
    
    return left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data

```

* 32nd cell

```python
def polyfit_using_prev_fit(binary_warped, left_fit_prev, right_fit_prev):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 80
    left_lane_inds = ((nonzerox > (left_fit_prev[0]*(nonzeroy**2) + left_fit_prev[1]*nonzeroy + left_fit_prev[2] - margin)) & 
                      (nonzerox < (left_fit_prev[0]*(nonzeroy**2) + left_fit_prev[1]*nonzeroy + left_fit_prev[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit_prev[0]*(nonzeroy**2) + right_fit_prev[1]*nonzeroy + right_fit_prev[2] - margin)) & 
                       (nonzerox < (right_fit_prev[0]*(nonzeroy**2) + right_fit_prev[1]*nonzeroy + right_fit_prev[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    left_fit_new, right_fit_new = (None, None)
    if len(leftx) != 0:
        # Fit a second order polynomial to each
        left_fit_new = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit_new = np.polyfit(righty, rightx, 2)
    return left_fit_new, right_fit_new, left_lane_inds, right_lane_inds


```


To take the test2.jpg as the example, the fitted polynomial line is described as the following two curves, one for the left lane, the other for the right lane.
![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this by writing a function called calc_curv_rad_and_center_dist() in the 33rd cell in my code file `P2.ipynb`, firstly, define the conversions in x and y from pixels space to meters, then define the y-value where we want radius of curvature. I chose the y-value at the bottom of the image, and then make use of the formula 

$R_{curve} = \frac{(1+(2Ay + B)^2)^{3/2}}{|2A|}$

to calculate the radius of the center and the distance from the lance center where the distance is defined as the image x midpoint - mean of l_fit and r_fit intercepts

The explicit code for this computation is the following:

```python

def calc_curv_rad_and_center_dist(bin_img, l_fit, r_fit, l_lane_inds, r_lane_inds):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 3.048/100 # meters per pixel in y dimension, lane line is 10 ft = 3.048 meters
    xm_per_pix = 3.7/378 # meters per pixel in x dimension, lane width is 12 ft = 3.7 meters
    left_curverad, right_curverad, center_dist = (0, 0, 0)
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    h = bin_img.shape[0]
    ploty = np.linspace(0, h-1, h)
    y_eval = np.max(ploty)
  
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = bin_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Again, extract left and right line pixel positions
    leftx = nonzerox[l_lane_inds]
    lefty = nonzeroy[l_lane_inds] 
    rightx = nonzerox[r_lane_inds]
    righty = nonzeroy[r_lane_inds]
    
    if len(leftx) != 0 and len(rightx) != 0:
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters
    
    # Distance from center is image x midpoint - mean of l_fit and r_fit intercepts 
    if r_fit is not None and l_fit is not None:
        car_position = bin_img.shape[1]/2
        l_fit_x_int = l_fit[0]*h**2 + l_fit[1]*h + l_fit[2]
        r_fit_x_int = r_fit[0]*h**2 + r_fit[1]*h + r_fit[2]
        lane_center_position = (r_fit_x_int + l_fit_x_int) /2
        center_dist = (car_position - lane_center_position) * xm_per_pix
    return left_curverad, right_curverad, center_dist

```
and we can calculate the curvature and distance from lane center for the test2.jpg
are 455.9491600309526 meters on the left lane, 1283.1788379439556 meters on the right lane, and the distance from the lane center is -0.39993419037886746 meters.



#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in 34th and 35th cell in Ipython Notebook `P2.ipynb` in the function `draw_lane()` using the inverse matrix to draw the unwarped image back to original warped image.  Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline implementation of the problem seems to work well on the project_video.mp4,however, the pipeline doesnot work so well when applied to the challenge_video.mp4(Here's a [link to my video result on challenge_video.mp4](./output_videos/challenge_video_output.mp4)), the right lane seem not so stable with some abrupt jump to some sigular curves twisted with the left curve. What is even more headache is the implementation of the harder_challenge_video.mp4  (Here's a [link to my video result on harder_challenge_video.mp4](./output_videos/harder_challenge_video_output.mp4)), wthe lane detection is seriously affected by the forests around as well as the sharp turn. When the car takes the U turn, the left lane and right lane seem to interfere with each other and I'm afraid my self-driving might fall over the the dark forest our hit the moutain to see my Dear God:(  
        
Would you be so kind to help me with my code and draw me back to life? 
