'''
module for detecting lanes
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None


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


def filter_search(img, left_fit, right_fit):
    '''
    Assume you now have a new warped binary image
    from the next frame of video (also called "img")
    It's now much easier to find line pixels!
    '''
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Draw the results
    l_points = np.zeros_like(img)
    r_points = np.zeros_like(img)
    l_points[(lefty, leftx)] = 255
    r_points[(righty, rightx)] = 255
    template = np.array(r_points + l_points, np.uint8) # add both left and right window pixels togther
    #l_template = np.array(l_points, np.uint8) # add both left and right window pixels togther
    #r_template = np.array(r_points, np.uint8) # add both left and right window pixels togther
    zero_channel = np.zeros_like(template) # create a zero color channel
    template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
    #template = np.array(cv2.merge((l_template,zero_channel, r_template)),np.uint8) # make window pixels green
    warpage = np.array(cv2.merge((img,img,img)),np.uint8) # making the original road pixels 3 color channels
    output = cv2.addWeighted(warpage, 0.5, template, 0.5, 0.0) # overlay the orignal road image with window results
    return output, l_points, r_points


def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output


def find_window_centroids(warped, window_width, window_height, margin):
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
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

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
    mid_car = 640.0         # middle point of car dashboard
    xm_per_pix = 3.7/700    # meters per pixel in x dimension
    #suit_curv = pick_curv(left_curverad, right_curverad)                    # unit: m
    base_pos = ((right_fitx[-1] + left_fitx[-1])/2. - mid_car) * xm_per_pix # unit: m
    head_dist = (right_fitx[-1] - left_fitx[-1]) * xm_per_pix               # unit: m
    tail_dist = (right_fitx[0] - left_fitx[0]) * xm_per_pix                 # unit: m


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
    cv2.putText(result, left_str, (450,50), font, 0.75, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(result, right_str, (450,100), font, 0.75, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(result, pos_str, (450, 150), font, 0.75, (255,255,255), 2, cv2.LINE_AA)
    #cv2.putText(result, head_str, (50, 200), font, 1, (255,255,255), 2, cv2.LINE_AA)
    #cv2.putText(result, tail_str, (50, 250), font, 1, (255,255,255), 2, cv2.LINE_AA)
    #cv2.putText(result, san_str, (50, 350), font, 1, (255,255,255), 2, cv2.LINE_AA)
    return result


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


