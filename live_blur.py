import cv2
import cv2.data
import numpy as np
import scipy
import scipy.ndimage
from scipy import signal
from scipy.signal import windows

PATH_XML = 'haarcascade_frontalface_default.xml'

def generate_kernel(kernel_len=5, desvio_padrao=5):
    generate_kernelid = windows.gaussian(kernel_len, std=desvio_padrao).reshape(kernel_len, 1)
    geneate_kernel2d = np.outer(generate_kernelid, generate_kernelid)
    return geneate_kernel2d

video_capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + PATH_XML)

kernel = generate_kernel(kernel_len=31, desvio_padrao=30)
# kernel_tile = np.tile(kernel, (3, 1, 1))
kernel_sum = kernel.sum()
kernel = kernel / kernel_sum

while True:
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for x, y, w, h in faces:
        # frame[y:y+h,x:x+w] = scipy.ndimage.convolve(frame[y:y+h,x:x+w], np.atleast_3d(kernel), mode='nearest')
        frame[y:y+h, x:x+w] = cv2.GaussianBlur(frame[y:y+h, x:x+w], (63, 63), sigmaX=20, sigmaY=20)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
