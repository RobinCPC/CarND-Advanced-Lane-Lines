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
from detect_lanes import sliding_window_search, window_mask, find_window_centroids


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
def thresh_pipeline(img, s_thresh=(180, 250), sx_thresh=(20, 100)):
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
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv


# Detect lane lines
# seprate lane detecting functions into detect_lanes.py

# Determine the lane curvature
def measure_curvature(leftx, rightx, y_len=720):
    # Fit a second order polynomial to pixel positions in each fake lane line
    ploty = np.linspace(0, y_len-1, num=y_len)# to cover same y-range as image
    #left_fit = np.polyfit(ploty, leftx, 2)
    left_fit = np.polyfit(leftx[:,0], leftx[:,1], 2)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    #right_fit = np.polyfit(ploty, rightx, 2)
    right_fit = np.polyfit(rightx[:,0], rightx[:,1], 2)
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]


    # Calculate the radius of curvature
    y_eval = np.max(ploty)
    print('y_eval:{}', y_eval)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    print(left_curverad, right_curverad)


    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(leftx[:,0]*ym_per_pix, leftx[:,1]*xm_per_pix, 2)
    right_fit_cr = np.polyfit(rightx[:,0]*ym_per_pix, rightx[:,1]*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    print(left_curverad, 'm', right_curverad, 'm')

    return left_fitx, right_fitx, left_curverad, right_curverad


def visualize_output(undist, warped, Minv, left_fitx, right_fitx, ploty, ave_curv, base_pos):
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, img_size)
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    # put text on image
    curv_str = 'Radius of Curvature = {:.3f}(m)'.format(ave_curv)
    pos_str = ''
    if base_pos <= 0:
        pos_str = 'Vehicle is {:.3f}m left of center'.format(abs(base_pos))
    else:
        pos_str = 'Vehicle is {:.3f}m right of center'.format(abs(base_pos))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, curv_str, (50,50), font, 2, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(result, pos_str, (50, 100), font, 2, (255,255,255), 2, cv2.LINE_AA)
    return result


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
        M, Minv = warped()

        # read images and processing
        test_images = glob.glob('./test_images/test5.jpg')
        for ind, fn in enumerate(test_images):
            img = cv2.imread(fn)
            img_size = (img.shape[1], img.shape[0])

            # undistort image before any preprocessing
            undist = cv2.undistort(img, mtx, dist, None, mtx)
            undist = cv2.GaussianBlur(undist, (3,3), 0)

            # color/grdient threshold
            thresh_bin = thresh_pipeline(undist, s_thresh=(190, 250), sx_thresh=(20, 100))

            # Perspective transform to get bird-eyes view
            warped = cv2.warpPerspective(thresh_bin, M, img_size, flags=cv2.INTER_LINEAR)

            # find lane by sliding window
            lane_img, leftx, rightx = sliding_window_search(warped*255)

            plt.subplot(221), plt.imshow(undist[:,:,::-1])
            plt.subplot(222), plt.imshow(thresh_bin, cmap='gray')
            plt.subplot(223), plt.imshow(warped, cmap='gray')
            plt.subplot(224), plt.imshow(lane_img)
            plt.show()

            #leftx = leftx[::-1]
            #rightx = rightx[::-1]
            print(type(leftx))
            print(leftx.shape)
            #print(np.transpose(np.nonzero(leftx)))

            leftx = np.transpose(np.nonzero(leftx))
            print(leftx[:,0].shape)
            print(leftx[:,1].shape)
            rightx = np.transpose(np.nonzero(rightx))


            # Measure curvature (wrap into a funtion)
            left_fitx, right_fitx, left_curverad, right_curverad = measure_curvature(leftx, rightx, y_len=img_size[1])

            ave_curv = (left_curverad + right_curverad) / 2.0                  # unit: m
            base_pos = ((right_fitx[-1] + left_fitx[-1])/2. - 640) * (3.7/700) # unit: m
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


            # Create an image to draw the lines on
            result = visualize_output(undist, warped, Minv, left_fitx, right_fitx, ploty, ave_curv, base_pos)
            #cv2.imshow('result', result)
            #cv2.waitKey(2000)
            cv2.imwrite('result.png', result)
            plt.imshow(result[:,:,::-1])
            plt.show()


        print("nothing now")
    pass


