import cv2
from src.detect import Detect
from moviepy.editor import VideoFileClip
import glob


imgsNames = glob.glob('test_videos/proj_vid_imgs/*.png')
det = Detect()


# videoOutputPath = 'output_videos/project_video_test.mp4'
# videoInput = VideoFileClip('test_videos/project_video.mp4')
# videoOutput = videoInput.fl_image(det.run)
# videoOutput.write_videofile(videoOutputPath, audio=False)


for i in range(0, len(imgsNames)):
    print('frame: ', i)
    img = cv2.imread(imgsNames[i])
    img = det.run(img)
    cv2.imshow('lane drawn', img)
    cv2.waitKey()
cv2.destroyAllWindows()