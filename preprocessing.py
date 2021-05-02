## reading the image, converting to RGB color Channel instead of BGR and displayng the image withouh any interpolation

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from time import sleep


file_path = './photos/IMG_5714.jpg'

MAX_PIX =800


img = cv.imread(file_path,flags=cv.IMREAD_COLOR)
h, w, c = img.shape
print(f'{h}H x {w}W x {c}C')
# cv.imshow('original image',img)

img = cv.cvtColor(src=img, code=cv.COLOR_BGR2RGB)


def show_image(img, **kwargs):
    """Show RGB numpy array of image without any interpolaiton"""
    plt.subplot()
    plt.axis('off')
    plt.imshow(X=img, interpolation=None, **kwargs)
    plt.show()
    # sleep(3)

show_image(img)
# cv2.imshow('RGB image ',img)

# Crop the image (slice along the height and width)

ymin,ymax = 63,435
xmin,xmax = 55,655
img = img[int(ymin):int(ymax), int(xmin):int(xmax)]

h,w,c = img.shape
print(f'image shape: {h}H x {w}W x{c}C')
show_image(img)


## Add a border -- padding (useful for API with OCR model trained on documents-- documents usually have white border)

img_border = cv.copyMakeBorder(src=img, top=10,bottom=10,left=10,right=10,borderType=cv.BORDER_CONSTANT, value=(255,255,255))

h,w,c = img_border.shape
print(f'Image with border shape: {h}Hx {w}Wx {c}C')
show_image(img_border)

# resize the image
'''APIs will have maximim dimension for input image  if the image need resizing the aspect ratio should be preserved'''

def resize_image(img,flag):
    """Resize the RGB numpy array of images either along the height or the width and keep its apsect ratio"""
    h,w,c = img.shape
    if flag=='h':
        dsize = (int((MAX_PIX*w)/h), int(MAX_PIX))
    else:
        dsize = (int(MAX_PIX),int((MAX_PIX*h)/w))

    im_resized = cv.resize(img,dsize=dsize, interpolation=cv.INTER_CUBIC)

    h,w,c = im_resized.shape

    print(f'resized image shape:{h}Hx{w}Wx{c}C ')
    show_image(im_resized)
    return im_resized


if h>MAX_PIX:
    im_resized = resize_image(img,'h')

if w>MAX_PIX:
    im_resized = resize_image(img,'w')
else:
    im_resized = img


print('resized image channel',im_resized.shape)


# Apply Morphological Operations
# closing - for closing small holes inside the foreground object suing 5x5 kernel
# openeing - for removing noise using 5x5 kernel

# def apply_morph(img,method):
#     """apply morphological operation either opening or closing"""
#     if method=='open':
#         op = cv.MORPH_OPEN
#     if method =='close':
#         op = cv.MORPH_CLOSE
#
#     img_morphology = cv.morphologyEx(src=img,op=op,kernel=np.ones((5,5),np.uint8))
#     show_image(img_morphology)
#     return img_morphology
#
# img = apply_morph(im_resized,'open')


# apply gaussain blurring -- low pass filer (remove high grequency noise)
# bigger the kernel more the blurring effect
#  If both are given as zeros, they are calculated from the kernel size.
#  Gaussian blurring is highly effective in removing Gaussian noise from an image.

img_gaussain = cv.GaussianBlur(src=im_resized, ksize=(5,5), sigmaX=0, sigmaY=0)

show_image(img_gaussain)


# Adaptive Thresholding
# transforms grayscale image into binary image
# adaptive thresholding is useful when the image has different lighting conditions in different areas
# as it calculates different thresholds for different regions.

def apply_adaptive_threshold(img,method):
    """Mean Threshold: caclulates the mean
    Gaussain thresholding caculates the weighted sum of neighborhoold values
    block_size= size of neighborhood area
    c = constant to be subtracted from mean / weghted mean"""
    img = cv.cvtColor(img,cv.COLOR_RGB2GRAY)

    if method=='gaussian':
        adaptive_method = cv.ADAPTIVE_THRESH_GAUSSIAN_C
    elif method=='mean':
        adaptive_method = cv.ADAPTIVE_THRESH_MEAN_C

    im_adaptive = cv.adaptiveThreshold(src=img,maxValue=255,adaptiveMethod=adaptive_method,thresholdType=cv.THRESH_BINARY
                            ,blockSize=3,C=2)
    show_image(im_adaptive, cmap='gray')
    return im_adaptive




apply_adaptive_threshold(img_gaussain,'gaussian')

# using ostu's thresholding after adaptive
ret2,th2 = cv.threshold(apply_adaptive_threshold(img_gaussain,'gaussian'),0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

show_image(th2)
# Apply Sobel Filter
# combines gaussian smoothing and diffrentiation (first order derivative along x and y direction)
# useful to detect horizontal and veritcal edges that are resitant to noise

def apply_sobel(img,direction):
    img = cv.cvtColor(src=img, code=cv.COLOR_RGB2GRAY)

    if direction=='h':
        dx,dy = 0,1
    if direction=='v':
        dx,dy = 1,0

    img_sobel = cv.Sobel(src=img,ddepth=cv.CV_64F, dx=dx, dy=dy, ksize=5)
    return img_sobel

show_image(apply_sobel(img_border,'h'))
show_image(apply_sobel(img_border,'v'))


# Apply Laplacian Filter
#they use the second derivative of the image along
# x and y (by internally adding up the second x and y derivatives calculated using the Sobel operator).
# Laplacian operators are useful to detect edges


def apply_laplacian(img):
    img = cv.cvtColor(src=img,code=cv.COLOR_RGB2GRAY)
    img_laplacian = np.uint8(np.absolute(cv.Laplacian(src=img, ddepth=cv.CV_64F)))
    show_image(img_laplacian,cmap='gray')
    return img_laplacian


show_image(apply_laplacian(img_border))

# Encoding
# when you do have a number of box coordinates for different regions in your image,, previous method of cropping is slow
# crop each region > encode cropped image into memory buffer > post request using the image in memory buffer.


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

show_image(deskew(im_resized))
cv.waitKey(0)
cv.destroyAllWindows()
