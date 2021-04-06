from skimage.metrics import structural_similarity as ssim
import dogTenzer as dt
import cv2

vid = cv2.VideoCapture(0)
base_frame = None
gray_base_frame = None
frame_path = "frames/frame.jpg"
tenzer = dt.DogTenzer()
while (True):
    ret, frame = vid.read()
    if(gray_base_frame is None):
        gray_base_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    reference_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    (score, diff) = ssim(gray_base_frame, reference_gray, full=True)
    #print("SSIM: {}".format(score))

    cv2.imshow('ref', reference_gray)


    gray_base_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prediction = round(tenzer.predict_if_dog(gray_base_frame)[0][0], 1)
    if prediction > 0.3 and prediction < 1.0:
        cv2.imshow("hund", gray_base_frame)
        cv2.waitKey(0)

vid.release()
cv2.destroyAllWindows()