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
    src = np.float32([[584, 455], [700, 455], [1052, 675], [268, 675]])
    dst = np.float32([[320, 0], [960, 0], [950, 720], [320, 720]])
    M = cv2.getPerspectiveTransform(src, dst)
    return M


# Detect lane lines
def sliding_window_search(img, window_width=50, window_height=80, margin=100):
    '''
    Apply convolution to do sliding window search
    '''
    # Use convolution to find window centroids
    window_centroids = find_window_centroids(img, window_width, window_height, margin)
    # If found any window centers
    if len(window_centroids) > 0:
        # Points use to draw all the left and right windows
        l_points = np.zeros_like(img)
        r_points = np.zeros_like(img)

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width, window_height, img, window_centroids[level][0], level)
            r_mask = window_mask(window_width, window_height, img, window_centroids[level][1], level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1) )] = 255
            r_points[(r_points == 255) | ((r_mask == 1) )] = 255

        # Draw the results
        template = np.array(r_points + l_points, np.uint8) # add both left and right window pixels togther
        #l_template = np.array(l_points, np.uint8) # add both left and right window pixels togther
        #r_template = np.array(r_points, np.uint8) # add both left and right window pixels togther
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
        #template = np.array(cv2.merge((l_template,zero_channel, r_template)),np.uint8) # make window pixels green
        warpage = np.array(cv2.merge((img,img,img)),np.uint8) # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
    return output, l_points, r_points


def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output


def find_window_centroids(image, window_width, window_height, margin):

    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)

    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))

    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(warped.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        # Add what we found for that layer
        window_centroids.append((l_center,r_center))

    return window_centroids


# Determine the lane curvature



if __name__ == '__main__':
    args = parse_arg(sys.argv)
    if int(args.calibrate) == 1:
        calib()
    else:
        # load camera parametes (mtx, dist)
        mtx, dist = [] ,[]
        with open('./camera_cal/wide_dist_pickle.p', 'rb') as f:
            data = pickle.load(f)
            mtx, dist = data['mtx'], data['dist']
        # add matrix for perspective transform
        M = warped()

        # read images and processing
        test_images = glob.glob('./test_images/test5.jpg')
        #print(test_images[-1])
        for ind, fn in enumerate(test_images):
            img = cv2.imread(fn)
            img_size = (img.shape[1], img.shape[0])
            # undistort image before any preprocessing
            undist = cv2.undistort(img, mtx, dist, None, mtx)
            undist = cv2.GaussianBlur(undist, (3,3), 0)
            # color/grdient threshold
            thresh_bin = thresh_pipeline(undist)
            # Perspective transform to get bird-eyes view
            warped = cv2.warpPerspective(thresh_bin, M, img_size, flags=cv2.INTER_LINEAR)
            # find lane by sliding window
            lane_img, leftx, rightx = sliding_window_search(warped*255)
            plt.subplot(221), plt.imshow(undist)
            plt.subplot(222), plt.imshow(thresh_bin, cmap='gray')
            plt.subplot(223), plt.imshow(warped, cmap='gray')
            plt.subplot(224), plt.imshow(lane_img)
            plt.show()
            #plt.imshow(warped, cmap='gray')
            #plt.show()
            #cv2.imwrite('./warp2.png', warped)

            ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
            #leftx = leftx[::-1]
            #rightx = rightx[::-1]
            print(type(leftx))
            print(leftx.shape)
            #print(np.transpose(np.nonzero(leftx)))

            leftx = np.transpose(np.nonzero(leftx))
            print(leftx[:,0].shape)
            print(leftx[:,1].shape)
            rightx = np.transpose(np.nonzero(rightx))


            # Fit a second order polynomial to pixel positions in each fake lane line
            #left_fit = np.polyfit(ploty, leftx, 2)
            left_fit = np.polyfit(leftx[:,0], leftx[:,1], 2)
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            #right_fit = np.polyfit(ploty, rightx, 2)
            right_fit = np.polyfit(rightx[:,0], rightx[:,1], 2)
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

            # Plot up the fake data
            mark_size = 3
            #plt.plot(leftx, ploty, 'o', color='red', markersize=mark_size)
            plt.plot(leftx[:,1], leftx[:,0], 'o', color='red', markersize=mark_size)
            #plt.plot(rightx, ploty, 'o', color='blue', markersize=mark_size)
            plt.plot(rightx[:,1], rightx[:,0], 'o', color='blue', markersize=mark_size)
            plt.xlim(0, 1280)
            plt.ylim(0, 720)
            plt.plot(left_fitx, ploty, color='green', linewidth=3)
            plt.plot(right_fitx, ploty, color='green', linewidth=3)
            plt.gca().invert_yaxis() # to visualize as we do the images
            plt.show()
            #left_fit = np.polyfit(ploty, leftx, 2)
            #right_fit = np.polyfit(ploty, rightx, 2)




        print("nothing now")
    pass



        #cap = cv2.VideoCapture()
        #cap.open(args.file)
        #if(not cap.isOpened()):
        #    print("could not open the file!")
        #while(1):
        #    ret, frame = cap.read()

        #    cv2.imshow('play',frame)
        #    k = cv2.waitKey(30) & 0xff
        #    if k == 27:
        #        break
