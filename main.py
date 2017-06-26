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


# CLI parser
def parse_arg(argv):
    '''
    parsing cli arguments
    '''
    parser = argparse.ArgumentParser(description='Advanced Lane Finding module')
    parser.add_argument('-c', '--calibrate', default=0, help='Set 1 if do calibration')
    parser.add_argument('-fd','--folder', default='./camera_cal/', help='the folder that consist images for calibration.' )
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
            #write_name = 'corners_found'+str(ind)+'.jpg'
            #cv2.imwrite(write_name, img)
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
    #cv2.imshow('undistortion', dst)
    #cv2.waitKey()

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
def thresh_pipeline(img, s_thresh=(140, 255), sx_thresh=(20, 100)):
    '''
    work on undistorted image
    '''
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    # Sobel x
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    #sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # TODO: why use l_channel Take the derivative in x
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
def warped():
    '''
    return Perspective matrix
    Currently, The points for perspective are chosed manually.
    '''


# Detect lane lines


# Determine the lane curvature



if __name__ == '__main__':
    args = parse_arg(sys.argv)
    if int(args.calibrate) == 1:
        calib()
    else:
        print("nothing now")
    pass
