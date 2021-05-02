import cv2
import pytesseract
import numpy as np

import matplotlib.pyplot as plt

file_path = './photos/IMG_5716.jpg'

MAX_PIX = 800

def extract_filename(file_path):
    list = file_path.split('/')[-1]
    file_name = list.split('.')[0]
    return file_name

img = cv2.imread("./photos/IMG_5716.jpg")

h, w, c = img.shape
print(f'{h}H x {w}W x {c}C')

def show_image(img, **kwargs):
    """Show RGB numpy array of image without any interpolaiton"""
    plt.subplot()
    plt.axis('off')
    plt.imshow(X=img, interpolation=None, **kwargs)
    plt.show()


show_image(img)

ymin, ymax = 20, 115
xmin, xmax = 25, 315
img = img[int(ymin):int(ymax), int(xmin):int(xmax)]
h, w, c = img.shape
print(f'image shape: {h}H x {w}W x{c}C')
show_image(img)





img = cv2.resize(img, None, fx=2.4, fy=2.4,
                 interpolation=cv2.INTER_CUBIC)
gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



thr = cv2.adaptiveThreshold(gry, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                            cv2.THRESH_BINARY_INV, 25, 22)

# thr = cv2.threshold(cv2.bilateralFilter(gry, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

bnt = cv2.bitwise_not(thr)
show_image(bnt)

kernel = np.ones((1, 1), np.uint8)
img = cv2.erode(bnt, kernel, iterations=1)
img = cv2.dilate(img, kernel, iterations=1)

show_image(img)

txt = pytesseract.image_to_string(img, config="--psm 6")
print(txt)

print(txt)
with open(f'{extract_filename(file_path)}.txt', mode='w') as f:
    f.write(txt)