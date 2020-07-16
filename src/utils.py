import numpy as np
import cv2
import glob

def calibrateCamera(self):
    cal_images = glob.glob('camera_cal/calibration*.jpg')

    nx = 9
    ny = 6

    good_cal_images = 0
    imgp = []
    cal_images_shape = (720, 1280, 3)

    for name in cal_images:
        img = cv2.imread(name)
        if img.shape != cal_images_shape:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret == True:
            imgp.append(corners)
            good_cal_images += 1

    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx,0:ny].T.reshape(-1, 2)
    objp = [objp for _ in range(good_cal_images)]

    cal_shape = (cal_images_shape[1], cal_images_shape[0])

    return cv2.calibrateCamera(objp, imgp, cal_shape, None, None)


def thresh(self, img, s_thresh=(100, 255), l_thresh=(100, 255), sx_thresh=(20, 100)):

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    s_channel = hls[:,:,2]
    l_channel = hls[:,:,1]

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    binary = np.zeros(scaled_sobel.shape, dtype=scaled_sobel.dtype)
    binary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 255
    binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])\
        &  (l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 255

    return binary

def getTop(point1, point2, y):
    x = (y-point1[1])*(point2[0]-point1[0])/(point2[1]-point1[1]) + point1[0]
    return [x, y]


def warp(self, img):
    leftBottom = [260,  680]
    rightBottom = [1050, 680]

    origHeight = 430
    leftTopOrig = [625, origHeight]
    rightTopOrig = [650, origHeight]

    leftTop = getTop(leftBottom, leftTopOrig, self.ROI_height)
    rightTop = getTop(rightBottom, rightTopOrig, self.ROI_height)

    src = np.float32([leftBottom, leftTop, rightTop, rightBottom])

    dst = np.float32([[300, img.shape[0]],
                      [300, 0],
                      [980, 0],
                      [980, img.shape[0]]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    threshed = np.zeros(warped.shape)
    threshed[warped>0]=255

    return threshed, M, Minv