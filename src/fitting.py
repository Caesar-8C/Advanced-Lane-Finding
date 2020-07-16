import numpy as np
import matplotlib.pyplot as plt
import cv2

def fitPoly(self, leftx, lefty, rightx, righty):
    leftw = lefty
    rightw = righty

    leftFit = np.polyfit(lefty, leftx, 2, w=leftw)
    rightFit = np.polyfit(righty, rightx, 2, w=rightw)

    return leftFit, rightFit

def findWindows(self, binary_warped):
    histogram = np.sum(binary_warped[int(0.65*binary_warped.shape[0]):,:], axis=0)
    midpoint = np.int(histogram.shape[0]//2)

    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 15
    minMargin = 50
    maxMargin = 150
    margin = np.linspace(minMargin, maxMargin, nwindows).astype(np.int)
    minpix = 50

    leftStop = False
    rightStop = False

    window_height = np.int(binary_warped.shape[0]//nwindows)

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height

        if not leftStop:
            win_xleft_low = leftx_current - margin[window]
            win_xleft_high = leftx_current + margin[window]
        if not rightStop:
            win_xright_low = rightx_current - margin[window]
            win_xright_high = rightx_current + margin[window]

        if self.verbalize:
            if not leftStop:
                cv2.rectangle(out_img,(win_xleft_low,win_y_low),
                (win_xleft_high,win_y_high),(0,255,0), 2)
            if not rightStop:
                cv2.rectangle(out_img,(win_xright_low,win_y_low),
                (win_xright_high,win_y_high),(0,255,0), 2)

        good_left_inds = []
        good_right_inds = []
        for i in range(len(nonzerox)):
            if win_y_low <= nonzeroy[i] < win_y_high:
                if not leftStop and win_xleft_low <= nonzerox[i] < win_xleft_high:
                    good_left_inds.append(i)
                if not rightStop and win_xright_low <= nonzerox[i] < win_xright_high:
                    good_right_inds.append(i)

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        def recenter(indices):
            if len(indices) <= minpix:
                return

            m = np.mean(nonzerox[indices])
            return int(m)

        newCurrent = recenter(good_left_inds)
        if newCurrent != None:
            leftx_current = newCurrent
            if binary_warped.shape[1]-margin[window] < leftx_current or leftx_current < margin[window]: leftStop = True

        newCurrent = recenter(good_right_inds)
        if newCurrent != None:
            rightx_current = newCurrent
            if binary_warped.shape[1]-margin[window] < rightx_current or rightx_current < margin[window]: rightStop = True

    left_lane_inds = np.concatenate(left_lane_inds).astype(np.int)
    right_lane_inds = np.concatenate(right_lane_inds).astype(np.int)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if len(leftx) == 0 or len(lefty) == 0 or len(rightx) == 0 or len(righty) == 0:
        return [], []
    leftFit, rightFit = self.fitPoly(leftx, lefty, rightx, righty)
    leftFitScaled, rightFitScaled = self.fitPoly(leftx*self.xPix2M, lefty*self.yPix2M, rightx*self.xPix2M, righty*self.yPix2M)

    if False:
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = leftFit[0]*ploty**2 + leftFit[1]*ploty + leftFit[2]
        right_fitx = rightFit[0]*ploty**2 + rightFit[1]*ploty + rightFit[2]

        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        # out_img[ploty.astype(np.int), left_fitx.astype(np.int), 1:3] = 255
        # out_img[ploty.astype(np.int), right_fitx.astype(np.int), 1:3] = 255

        cv2.imshow('binary detection', out_img)

    return leftFit, rightFit, leftFitScaled, rightFitScaled



def findAroundPoly(self, binary_warped):
    margin = 50

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    x_min_left_array = []
    x_max_left_array = []
    x_min_right_array = []
    x_max_right_array = []

    left_lane_inds = []
    right_lane_inds = []

    for i in range(binary_warped.shape[0]):
        x_fit_left = self.leftFit[-1][0]*i**2 + self.leftFit[-1][1]*i + self.leftFit[-1][2]
        x_fit_right = self.rightFit[-1][0]*i**2 + self.rightFit[-1][1]*i + self.rightFit[-1][2]

        x_min_left_array.append(x_fit_left - margin)
        x_max_left_array.append(x_fit_left + margin)
        x_min_right_array.append(x_fit_right - margin)
        x_max_right_array.append(x_fit_right + margin)

    for i in range(len(nonzeroy)):
        x_min_left = x_min_left_array[nonzeroy[i]]
        x_max_left = x_max_left_array[nonzeroy[i]]
        x_min_right = x_min_right_array[nonzeroy[i]]
        x_max_right = x_max_right_array[nonzeroy[i]]

        if x_min_left < nonzerox[i] < x_max_left:
            left_lane_inds.append(i)
        elif x_min_right < nonzerox[i] < x_max_right:
            right_lane_inds.append(i)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if len(leftx) == 0 or len(lefty) == 0 or len(rightx) == 0 or len(righty) == 0:
        return [], []

    leftFit, rightFit = self.fitPoly(leftx, lefty, rightx, righty)
    leftFitScaled, rightFitScaled = self.fitPoly(leftx*self.xPix2M, lefty*self.yPix2M, rightx*self.xPix2M, righty*self.yPix2M)



    if False:
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fitx = leftFit[0]*ploty**2 + leftFit[1]*ploty + leftFit[2]
        right_fitx = rightFit[0]*ploty**2 + rightFit[1]*ploty + rightFit[2]

        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin,
                                  ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin,
                                  ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        out_img[ploty.astype(np.int), left_fitx.astype(np.int), 1:3] = 255
        out_img[ploty.astype(np.int), right_fitx.astype(np.int), 1:3] = 255

        cv2.imshow('binary detection', out_img)

    return leftFit, rightFit, leftFitScaled, rightFitScaled