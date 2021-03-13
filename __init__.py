from skimage.metrics import structural_similarity as ssim
import os
import cv2

vid = cv2.VideoCapture(1)
base_frame = None
gray_base_frame = None
while (True):
    ret, frame = vid.read()
    if(gray_base_frame is None):
        gray_base_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    reference_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    (score, diff) = ssim(gray_base_frame, reference_gray, full=True)
    print("SSIM: {}".format(score))

    cv2.imshow('ref', reference_gray)

    if(score < 0.7):
        gray_base_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        os.system('afplay /Users/noel/Desktop/nela.mp3') 

vid.release()
cv2.destroyAllWindows()
