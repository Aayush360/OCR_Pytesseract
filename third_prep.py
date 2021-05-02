## reading the image, converting to RGB color Channel instead of BGR and displayng the image withouh any interpolation

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pytesseract

file_path = './photos/IMG_5716.jpg'

def extract_filename(file_path):
    list = file_path.split('/')[-1]
    file_name = list.split('.')[0]
    return file_name

print(extract_filename(file_path))

img = cv.imread(file_path, flags=cv.IMREAD_COLOR)
h, w, c = img.shape
print(f'{h}H x {w}W x {c}C')

img = cv.cvtColor(src=img, code=cv.COLOR_BGR2RGB)


def show_image(img, **kwargs):
    """Show RGB numpy array of image without any interpolaiton"""
    plt.subplot()
    plt.axis('off')
    plt.imshow(X=img, interpolation=None, **kwargs)
    plt.show()


show_image(img)

# Crop the image (slice along the height and width)

ymin, ymax = 20, 115
xmin, xmax = 25, 315
img = img[int(ymin):int(ymax), int(xmin):int(xmax)]

h, w, c = img.shape
print(f'image shape: {h}H x {w}W x{c}C')
show_image(img)




def deskew(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    gray = cv.bitwise_not(gray)
    thresh = cv.threshold(gray, 0, 255,
                          cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(image, M, (w, h),
                            flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
    return rotated


im_deskewed = deskew(img)
show_image(im_deskewed)


img = cv.resize(im_deskewed, None, fx=2.5, fy=2.5,
                 interpolation=cv.INTER_CUBIC)

show_image(img)
gry = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

thr = cv.adaptiveThreshold(gry, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                            cv.THRESH_BINARY_INV, 25, 22)
show_image(thr)
bnt = cv.bitwise_not(thr)
show_image(bnt)

kernel = np.ones((1, 1), np.uint8)
img = cv.erode(bnt, kernel, iterations=1)
img = cv.dilate(img, kernel, iterations=1)

show_image(img)
txt = pytesseract.image_to_string(bnt, config="--psm 6")
print(txt)








# with open(f'{extract_filename(file_path)}1.txt', mode='w') as f:
#     f.write(txt)
