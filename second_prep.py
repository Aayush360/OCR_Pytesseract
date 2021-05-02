## reading the image, converting to RGB color Channel instead of BGR and displayng the image withouh any interpolation

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pytesseract

file_path = './photos/IMG_5715.jpg'

def extract_filename(file_path):
    list = file_path.split('/')[-1]
    file_name = list.split('.')[0]
    return file_name

MAX_PIX = 800

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

ymin, ymax = 58, 385
xmin, xmax = 80, 1080
img = img[int(ymin):int(ymax), int(xmin):int(xmax)]

h, w, c = img.shape
print(f'image shape: {h}H x {w}W x{c}C')
show_image(img)

## Add a border -- padding (useful for API with OCR model trained on documents-- documents usually have white border)

img_border = cv.copyMakeBorder(src=img, top=10, bottom=10, left=10, right=10, borderType=cv.BORDER_CONSTANT,
                               value=(255, 255, 255))

h, w, c = img_border.shape
print(f'Image with border shape: {h}Hx {w}Wx {c}C')
show_image(img_border)

# resize the image
'''APIs will have maximim dimension for input image  if the image need resizing the aspect ratio should be preserved'''


def resize_image(img, flag):
    """Resize the RGB numpy array of images either along the height or the width and keep its apsect ratio"""
    h, w, c = img.shape
    if flag == 'h':
        dsize = (int((MAX_PIX * w) / h), int(MAX_PIX))
    else:
        dsize = (int(MAX_PIX), int((MAX_PIX * h) / w))

    im_resized = cv.resize(img, dsize=dsize, interpolation=cv.INTER_CUBIC)

    h, w, c = im_resized.shape

    print(f'resized image shape:{h}Hx{w}Wx{c}C ')
    show_image(im_resized)
    return im_resized


if h > MAX_PIX:
    im_resized = resize_image(img, 'h')

if w > MAX_PIX:
    im_resized = resize_image(img, 'w')
else:
    im_resized = img

print('resized image channel', im_resized.shape)


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


im_deskewed = deskew(im_resized)
show_image(im_deskewed)




# apply gaussain blurring -- low pass filer (remove high grequency noise)
# bigger the kernel more the blurring effect
#  If both are given as zeros, they are calculated from the kernel size.
#  Gaussian blurring is highly effective in removing Gaussian noise from an image.

img_gaussain = cv.GaussianBlur(src=im_deskewed, ksize=(5,5), sigmaX=0, sigmaY=0)

show_image(img_gaussain)


# Adaptive Thresholding
# transforms grayscale image into binary image
# adaptive thresholding is useful when the image has different lighting conditions in different areas
# as it calculates different thresholds for different regions.

def apply_adaptive_threshold(img, method):
    """Mean Threshold: caclulates the mean
    Gaussain thresholding caculates the weighted sum of neighborhoold values
    block_size= size of neighborhood area
    c = constant to be subtracted from mean / weghted mean"""
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    if method == 'gaussian':
        adaptive_method = cv.ADAPTIVE_THRESH_GAUSSIAN_C
    elif method == 'mean':
        adaptive_method = cv.ADAPTIVE_THRESH_MEAN_C

    im_adaptive = cv.adaptiveThreshold(src=img, maxValue=255, adaptiveMethod=adaptive_method,
                                       thresholdType=cv.THRESH_BINARY
                                       , blockSize=3, C=3)
    show_image(im_adaptive, cmap='gray')
    return im_adaptive


im_thresh_adapt = apply_adaptive_threshold(img_gaussain, 'gaussian')

show_image(im_thresh_adapt)



# find contour from thresholded image

contours, _ = cv.findContours(im_thresh_adapt, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)


# text detection
def contours_text(orig, contours):
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)

        # Drawing a rectangle on copied image
        rect = cv.rectangle(orig, (x, y), (x + w, y + h), (0, 255, 255), 2)

        cv.imshow('cnt', rect)
        cv.waitKey()

        # Cropping the text block for giving input to OCR
        cropped = orig[y:y + h, x:x + w]

        # Apply OCR on the cropped image
        config = ('-l eng --oem 1 --psm 3')
        text = pytesseract.image_to_string(cropped, config=config)
    return text


# contours_text(im_deskewed,contours)
out = contours_text(im_deskewed, contours)

with open(f'{extract_filename(file_path)}.txt', mode='w') as f:
    f.write(out)
