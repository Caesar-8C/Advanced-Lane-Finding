import numpy as np
import cv2


class Detect:
    from src.fitting import fitPoly, findWindows, findAroundPoly
    from src.utils import calibrateCamera, thresh, warp

    def __init__(self):
        self.leftFit = []
        self.rightFit = []
        self.curveList = []
        self.newRun = True
        self.ROI_height = 450
        self.yPix2M = 40/720
        self.xPix2M = 3.7/680
        _, self.mtx, self.dist, _, _ = self.calibrateCamera()

    def preprocessImg(self, img):
        img = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        img = self.thresh(img)
        img, _, Minv = self.warp(img)
        return img, Minv

    def updateFit(self, newLeftFit, newRightFit):
        self.leftFit.append(newLeftFit)
        self.rightFit.append(newRightFit)

        if len(self.leftFit) == 6:
            del self.leftFit[0]
        if len(self.rightFit) == 6:
            del self.rightFit[0]

    def getPlotPoints(self, shape):
        plotY = np.linspace(0, shape[0]-1, shape[0])

        plotLeftFit = np.mean(self.leftFit, axis=0)
        plotRightFit = np.mean(self.rightFit, axis=0)

        leftFitX = plotLeftFit[0]*plotY**2 + plotLeftFit[1]*plotY + plotLeftFit[2]
        rightFitX = plotRightFit[0]*plotY**2 + plotRightFit[1]*plotY + plotRightFit[2]

        return leftFitX.astype(np.int), rightFitX.astype(np.int), plotY.astype(np.int)

    def getCurvature(self, y_eval, leftScaled, rightScaled):

        leftCurvature = ((1+(2*leftScaled[0]*y_eval*self.yPix2M+leftScaled[1])**2)**(3./2.))/2./abs(leftScaled[0])
        rightCurvature = ((1+(2*rightScaled[0]*y_eval*self.yPix2M+rightScaled[1])**2)**(3./2.))/2./abs(rightScaled[0])
        curvature = np.array([leftCurvature, rightCurvature]).astype(np.int)
        return curvature

    def getShift(self, leftFit, rightFit, y, center):
        bottomLeft = leftFit[0]*y**2 + leftFit[1]*y + leftFit[2]
        bottomRight = rightFit[0]*y**2 + rightFit[1]*y + rightFit[2]

        shift = ((bottomRight + bottomLeft)/2 - center)*self.xPix2M
        return shift


    def setText(self, img, curvature, shift):
        self.curveList.append(curvature)
        if len(self.curveList) == 6:
            del self.curveList[0]

        curvature = np.array(self.curveList)
        meanCurvature = np.mean(curvature).astype(int)

        if 0 < meanCurvature < 2000:
            curvText = 'lane curvature radius: '+str(meanCurvature)
        elif meanCurvature > 0:
            curvText = 'lane is straight'
        else:
            curvText = 'lane curvature lost'


        shiftText = 'car is shifted ' + str(round(abs(shift), 1)) + 'm '
        shiftText += 'left' if np.sign(shift) else 'right'
        shiftText += ' of the center'

        cv2.putText(img, curvText, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, shiftText, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        return img

    def curvatureTest(self, curvature):
        if  curvature[0] < 200:
            return False
        if  curvature[1] < 200:
            return False
        if curvature[0] < 2000 and curvature[1] < 2000:
            if abs(curvature[0]-curvature[1]) > 500:
                return False
        return True


    def run(self, img_orig):
        img, Minv = self.preprocessImg(img_orig)

        if self.newRun:
            leftFit, rightFit, leftFitScaled, rightFitScaled = self.findWindows(img)
            if len(leftFit) == 0 or len(rightFit) == 0:
                return img_orig
            self.updateFit(leftFit, rightFit)
        else:
            leftFit, rightFit, leftFitScaled, rightFitScaled = self.findAroundPoly(img)
            if len(leftFit) == 0 or len(rightFit) == 0:
                leftFit, rightFit, leftFitScaled, rightFitScaled = self.findWindows(img)
            self.updateFit(leftFit, rightFit)


        leftFitX, rightFitX, plotY = self.getPlotPoints(img_orig.shape)
        leftLinePoints = np.array([np.transpose(np.vstack([leftFitX, plotY]))])
        rightLinePoints = np.array([np.flipud(np.transpose(np.vstack([rightFitX, plotY])))])
        lanePoints = np.hstack((leftLinePoints, rightLinePoints))

        lanes_img = np.zeros(img_orig.shape, np.uint8)
        cv2.fillPoly(lanes_img, np.int_([lanePoints]), (0, 255, 0))

        lanes_img = cv2.warpPerspective(lanes_img, Minv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)


        weighted = cv2.addWeighted(img_orig, 1, lanes_img, 0.3, 0)
        curvature = self.getCurvature(max(plotY), leftFitScaled, rightFitScaled)
        shift = self.getShift(leftFit, rightFit, max(plotY), img.shape[1]/2)

        if not self.newRun:
            test = self.curvatureTest(curvature)
            if test == False:
                self.newRun = True
                return self.run(img_orig)
        self.newRun = False

        result = self.setText(weighted, curvature, shift)
        return result




