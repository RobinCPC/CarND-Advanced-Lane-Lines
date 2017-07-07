#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main script for Advanced Lanes Findings
"""
import argparse
import sys
import numpy as np
import glob
import pickle
import cv2
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from detect_lanes import sliding_window_search, window_mask, find_window_centroids
from detect_lanes import filter_search, Line
from collections import deque


# CLI parser
def parse_arg(argv):
    '''
    parsing cli arguments
    '''
    parser = argparse.ArgumentParser(description='Advanced Lane Finding module')
    parser.add_argument('-c', '--calibrate', default=0, help='Set 1 if do calibration')
    parser.add_argument('-fd','--folder', default='./camera_cal/', help='the folder that consist images for calibration.' )
    parser.add_argument('-v', '--video', default=0, help='Set 1 if process video file')
    parser.add_argument('-f', '--file', default='./project_video.mp4', help='the file for finding lane lines.')
    return parser.parse_args(argv[1:])


# Camera calibration
# Distortion correction
def calib(fold='./camera_cal/', nx=9, ny=6):
    '''
    Camera Calibration with OpenCV
    '''
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Array to store object points and image points from all the images.
    objpoints = []
    imgpoints = []

    # Make a list of calibration images
    images = glob.glob(fold+'*.jpg')

    # Step through the list and search for cheeseboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corner
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            #cv2.imshow('img', img)
            #cv2.waitKey(500)

    #cv2.destroyAllWindows()

    ## Test undistortion on an image
    img = cv2.imread('./camera_cal/calibration2.jpg')
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('./output_images/test_undist2.jpg', dst)

    # Save the camera calibration result for later use (we won't worry about rvecs/tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open(fold+"wide_dist_pickle.p", "wb") )
    #dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    # Visualize undistortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)
    plt.show()


# Color/gradient threshold
def thresh_pipeline(img, s_thresh=(180, 250), sx_thresh=(20, 100)):
    '''
    work on undistorted image
    '''
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    #sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary #color_binary


# Perspective transform
def get_warped():
    '''
    return Perspective matrix
    Currently, The points for perspective are chosed manually.
    '''
    src = np.float32([[584, 455], [700, 455], [1052, 675], [268, 675]])
    dst = np.float32([[320, 0], [960, 0], [960, 720], [320, 720]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv


# Detect lane lines
# seperate lane detecting functions into detect_lanes.py

# Determine the lane curvature
def measure_curvature(leftx, rightx, y_len=720):
    '''
    From detected pixel to compute polynomial function of lane line,
    and, then, computet the curvature and radius of lane line.
    '''
    # Fit a second order polynomial to pixel positions in each fake lane line
    ploty = np.linspace(0, y_len-1, num=y_len)# to cover same y-range as image
    left_fit = np.polyfit(leftx[:,0], leftx[:,1], 2)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fit = np.polyfit(rightx[:,0], rightx[:,1], 2)
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]


    # Calculate the radius of curvature
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    #print(left_curverad, right_curverad)


    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(leftx[:,0]*ym_per_pix, leftx[:,1]*xm_per_pix, 2)
    right_fit_cr = np.polyfit(rightx[:,0]*ym_per_pix, rightx[:,1]*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / (2*left_fit_cr[0]) #np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / (2*right_fit_cr[0]) #np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')

    return left_fit, right_fit, left_fitx, right_fitx, left_curverad, right_curverad


def get_curvature(left_fit, right_fit, y_len=720):
    '''
    Compute curvature and radius of lane lines from polynomial eaution of lane line
    '''
    ploty = np.linspace(0, y_len-1, num=y_len)# to cover same y-range as image
    left_fitx = line_l.best_fit[0]*ploty**2 + line_l.best_fit[1]*ploty + line_l.best_fit[2]
    right_fitx = line_r.best_fit[0]*ploty**2 + line_r.best_fit[1]*ploty + line_r.best_fit[2]

    # Calculate the radius of curvature
    y_eval = np.max(ploty)
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / (2*left_fit_cr[0]) #np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / (2*right_fit_cr[0]) #np.absolute(2*right_fit_cr[0])
    return left_fitx, right_fitx, left_curverad, right_curverad


def visualize_output(undist, warped, Minv, left_fitx, right_fitx, left_curverad, right_curverad, is_sanity=0, do_filter=0):
    '''
    visualize detected lanes on undistorted image
    '''
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    img_size = (undist.shape[1], undist.shape[0])
    ploty = np.linspace(0, img_size[1]-1, num=img_size[1])# to cover same y-range as image

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    img_size = (undist.shape[1], undist.shape[0])
    newwarp = cv2.warpPerspective(color_warp, Minv, img_size)
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    # put text on image and pick suitable curvature
    b_left, b_right = 0, 0
    suit_curv = 0
    suit_curv = pick_curv(left_curverad, right_curverad)            # unit: m
    base_pos = ((right_fitx[-1] + left_fitx[-1])/2. - 640) * (3.7/700)  # unit: m
    head_dist = (right_fitx[-1] - left_fitx[-1]) * (3.7/700)  # unit: m
    tail_dist = (right_fitx[0] - left_fitx[0]) * (3.7/700)  # unit: m


    left_str = 'Radius of left lane = {:.2f}(km)'.format(left_curverad/1000.)
    right_str = 'Radius of right lane = {:.2f}(km)'.format(right_curverad/1000.)
    #curv_str = 'Radius of Curvature = {:.2f}(km)'.format(np.abs(suit_curv/1000.))
    head_str = 'Lane width of head = {:.3f}(m)'.format(head_dist)
    tail_str = 'Lane width of tail = {:.3f}(m)'.format(tail_dist)
    san_str = 'filter= {}, sanity={}'.format(do_filter, is_sanity)
    pos_str = ''
    if base_pos <= 0:
        pos_str = 'Vehicle is {:.3f}m left of center'.format(abs(base_pos))
    else:
        pos_str = 'Vehicle is {:.3f}m right of center'.format(abs(base_pos))
    font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.putText(result, curv_str, (50,50), font, 2, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(result, left_str, (50,50), font, 1, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(result, right_str, (50,100), font, 1, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(result, pos_str, (50, 150), font, 1, (255,255,255), 2, cv2.LINE_AA)
    #cv2.putText(result, head_str, (50, 200), font, 1, (255,255,255), 2, cv2.LINE_AA)
    #cv2.putText(result, tail_str, (50, 250), font, 1, (255,255,255), 2, cv2.LINE_AA)
    #cv2.putText(result, san_str, (50, 350), font, 1, (255,255,255), 2, cv2.LINE_AA)
    return result


def pick_curv(left, right):
    '''
    Pick the curvature between left and right lane
    (not use in current pipeline)
    '''
    temp = np.array((left, right))
    ind = np.argmin(abs(temp - 1000.))
    return temp[ind]


def check_sanity(left_fitx, right_fitx, left_curverad, right_curverad):
    '''
    check sanity of lane detection
    '''
    fail_rate = 0
    base_pos = ((right_fitx[-1] + left_fitx[-1])/2. - 640) * (3.7/700)  # unit: m
    head_dist = (right_fitx[-1] - left_fitx[-1]) * (3.7/700)  # unit: m
    tail_dist = (right_fitx[0] - left_fitx[0]) * (3.7/700)  # unit: m

    # check if left and right lane are roughly parallel
    if head_dist < 2.6 or head_dist > 3.5:
        fail_rate += 1
    if tail_dist < 2.6 or tail_dist > 3.5:
        fail_rate += 1

    # check if curvatures are reasonable (not too small)
    if np.abs(left_curverad)/1000. < 0.4 or np.abs(right_curverad)/1000. < 0.4:
        fail_rate += 1
    return fail_rate


def do_processing(mtx, dist, M, Minv):
    '''
    upper_layer function to input relative parameter and set
    other global parameter to precessing image
    '''
    left_fit, right_fit = None, None
    line_l.current_fit = deque(maxlen=10)
    line_r.current_fit = deque(maxlen=10)
    # precossing image
    def process_image(image):
        '''
        main pipeline to process each frame of video
        '''
        # read image
        img = np.copy(image)
        img_size = (img.shape[1], img.shape[0])
        ploty = np.linspace(0, img_size[1]-1, num=img_size[1])# to cover same y-range as image

        # undistort image before any preprocessing
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        undist = cv2.GaussianBlur(undist, (3,3), 0)

        # color/grdient threshold
        thresh_bin = thresh_pipeline(undist, s_thresh=(160, 250), sx_thresh=(30, 60))

        # Perspective transform to get bird-eyes view
        warped = cv2.warpPerspective(thresh_bin, M, img_size, flags=cv2.INTER_LINEAR)

        # find lane by sliding window
        lane_img, leftx, rightx = sliding_window_search(warped*255)
        #if line_l.detected == False or line_r.detected == False:
        #    lane_img, leftx, rightx = sliding_window_search(warped*255)
        #    line_l.detected = True
        #    line_l.allx = leftx
        #    line_r.detected = True
        #    line_r.allx = rightx
        #else:
        #    lane_img, leftx, rightx = filter_search(warped*255, line_l.current_fit, line_r.current_fit)
        #    line_l.detected = True
        #    line_l.allx = leftx
        #    line_r.detected = True
        #    line_r.allx = rightx

        leftx = np.transpose(np.nonzero(leftx))
        rightx = np.transpose(np.nonzero(rightx))

        # Measure curvature (wrap into a funtion)
        out_group = measure_curvature(leftx, rightx, y_len=img_size[1])
        left_fit, right_fit, left_fitx, right_fitx, left_curverad, right_curverad = out_group

        # check sanity
        is_sanity = 0
        do_filter = check_sanity(left_fitx, right_fitx, left_curverad, right_curverad)

        if do_filter >= 1:
            lane_img, leftx, rightx = filter_search(warped*255, line_l.current_fit[-1], line_r.current_fit[-1])
            leftx = np.transpose(np.nonzero(leftx))
            rightx = np.transpose(np.nonzero(rightx))

            # Measure curvature (wrap into a funtion)
            out_group = measure_curvature(leftx, rightx, y_len=img_size[1])
            left_fit, right_fit, left_fitx, right_fitx, left_curverad, right_curverad = out_group

            # check sanity again if not pass use previous
            is_sanity = check_sanity(left_fitx, right_fitx, left_curverad, right_curverad)

        if is_sanity < 1:
            line_l.current_fit.append(left_fit)   # use append  and measured_curvature for ave curv
            line_r.current_fit.append(right_fit)
            line_l.best_fit = np.mean(line_l.current_fit, axis=0)
            line_r.best_fit = np.mean(line_r.current_fit, axis=0)
            out_group =  get_curvature(line_l.best_fit, line_r.best_fit, y_len=img_size[1])
            left_fitx, right_fitx, left_curverad, right_curverad = out_group

            line_l.allx = left_fitx
            line_l.radius_of_curvature = left_curverad
            line_r.allx = right_fitx
            line_r.radius_of_curvature = right_curverad
        else:
            left_fit = line_l.current_fit[-1]
            left_curverad = line_l.radius_of_curvature
            left_fitx = line_l.allx
            right_fit = line_r.current_fit[-1]
            right_curverad = line_r.radius_of_curvature
            right_fitx = line_r.allx

        # Create an image to draw the lines on
        result = visualize_output(undist, warped, Minv, left_fitx, right_fitx, left_curverad, right_curverad, is_sanity, do_filter)
        return result

    return process_image


# Declare a global Line class object to store useful parameter to check the
# sanity between each frame of images
global line_l, line_r
line_l = Line()
line_r = Line()
global left_fit
global right_fit


if __name__ == '__main__':
    args = parse_arg(sys.argv)
    if int(args.calibrate) == 1:
        '''
        Do calibration with images in camera_cal
        '''
        calib()

    elif int(args.video) == 1:
        '''
        Use pipeline to detect lane line on video
        '''
        # load camera parametes (mtx, dist)
        mtx, dist = [] ,[]
        with open('./camera_cal/wide_dist_pickle.p', 'rb') as f:
            data = pickle.load(f)
            mtx, dist = data['mtx'], data['dist']
        # add matrix for perspective transform
        M, Minv = get_warped()

        # read file name from cli parameters
        fn = args.file

        project_output = './output_images/project.mp4'
        clip1 = VideoFileClip(fn)
        white_clip = clip1.fl_image(do_processing(mtx, dist, M, Minv)) #NOTE: this function expects color images!!
        white_clip.write_videofile(project_output, audio=False)

    else:
        '''
        Test pipeline on images in test_images dir
        '''
        # load camera parametes (mtx, dist)
        mtx, dist = [] ,[]
        with open('./camera_cal/wide_dist_pickle.p', 'rb') as f:
            data = pickle.load(f)
            mtx, dist = data['mtx'], data['dist']
        # add matrix for perspective transform
        M, Minv = get_warped()

        left_fit, right_fit = None, None

        # read images and processing
        test_images = glob.glob('./test_images/*.jpg')
        for ind, fn in enumerate(test_images):
            img = cv2.imread(fn)
            img_size = (img.shape[1], img.shape[0])

            # undistort image before any preprocessing
            undist = cv2.undistort(img, mtx, dist, None, mtx)
            undist = cv2.GaussianBlur(undist, (3,3), 0)

            # color/grdient threshold
            thresh_bin = thresh_pipeline(undist, s_thresh=(160, 250), sx_thresh=(30, 60))

            # Perspective transform to get bird-eyes view
            warped = cv2.warpPerspective(thresh_bin, M, img_size, flags=cv2.INTER_LINEAR)

            # find lane by sliding window and filter search
            #lane_img, leftx, rightx = sliding_window_search(warped*255)
            if left_fit is None or right_fit is None:
                lane_img, leftx, rightx = sliding_window_search(warped*255)
                print(lane_img.shape)
            else:
                lane_img, leftx, rightx = filter_search(warped*255, left_fit, right_fit)
                print(lane_img.shape)

            plt.subplot(221), plt.imshow(undist[:,:,::-1])
            plt.subplot(222), plt.imshow(thresh_bin, cmap='gray')
            plt.subplot(223), plt.imshow(warped, cmap='gray')
            plt.subplot(224), plt.imshow(lane_img)
            plt.show()

            # transform to suitable data structure for computing curvature
            leftx = np.transpose(np.nonzero(leftx))
            rightx = np.transpose(np.nonzero(rightx))

            # Measure curvature (wrap into a funtion)
            out_group = measure_curvature(leftx, rightx, y_len=img_size[1])
            left_fit, right_fit, left_fitx, right_fitx, left_curverad, right_curverad = out_group

            # pick suitable curvature
            b_left, b_right = 0, 0
            suit_curv = 0
            suit_curv = pick_curv(left_curverad, right_curverad)                # unit: m
            base_pos = ((right_fitx[-1] + left_fitx[-1])/2. - 640) * (3.7/700)  # unit: m
            ploty = np.linspace(0, img_size[1]-1, num=img_size[1])# to cover same y-range as image


            # Plot up the data
            mark_size = 3
            plt.plot(leftx[:,1], leftx[:,0], 'o', color='red', markersize=mark_size)
            plt.plot(rightx[:,1], rightx[:,0], 'o', color='blue', markersize=mark_size)
            plt.xlim(0, 1280)
            plt.ylim(0, 720)
            plt.plot(left_fitx, ploty, color='green', linewidth=3)
            plt.plot(right_fitx, ploty, color='green', linewidth=3)
            plt.gca().invert_yaxis() # to visualize as we do the images
            plt.text(400, 100, 'left_cur:{:.3f}, right_cur:{:.3f}'.format(left_curverad, right_curverad))
            plt.show()


            # Create an image to draw the lines on undistorted image.
            result = visualize_output(undist, warped, Minv, left_fitx, right_fitx, left_curverad, right_curverad)
            cv2.imwrite('result.png', result)
            plt.imshow(result[:,:,::-1])
            plt.show()



