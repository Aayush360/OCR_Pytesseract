import cv2
import numpy as np
import pytesseract

file_name ='photos/lic2.JPG'

# import the necessary packages
import numpy as np
import argparse
import cv2

image = '/Users/aayush/PycharmProjects/ComputerVision/ocr/photos/IMG_5714.jpg'


def deskew(image):
    # load the image from disk
    image = cv2.imread(image)

    # convert the image to grayscale and flip the foreground
    # and background to ensure foreground is now "white" and
    # the background is "black"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    # threshold the image, setting all foreground pixels to
    # 255 and all background pixels to 0
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # grab the (x, y) coordinates of all pixel values that
    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all
    # coordinates
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)
    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle

    # rotate the image to deskew it using affine transformation
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # draw the correction angle on the image so we can validate it
    # cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # show the output image
    # print("[INFO] angle: {:.3f}".format(angle))
    # cv2.imshow("Input", image)
    # cv2.imshow("Rotated", rotated)
    # cv2.waitKey(0)
    return rotated


def gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(r"./ocr/photos/img_gray.png", img)
    return img


# blur
def blur(img):
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite(r"./ocr/photos/img_blur.png", img)
    return img_blur


# threshold
def threshold(img):
    # pixels with value below 100 are turned black (0) and those with higher value are turned white (255)
    img = cv2.threshold(img, 100, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)[1]
    cv2.imwrite(r"./ocr/photos/img_threshold.png", img)
    return img


# im = cv2.imread(file_name)
im_deskewed = deskew(image)
# Finding contours
im_gray = gray(im_deskewed)
im_blur = blur(im_gray)
im_thresh = threshold(im_blur)

contours, _ = cv2.findContours(im_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


# text detection
def contours_text(orig, contours):
    for cnt in contours:

        x, y, w, h = cv2.boundingRect(cnt)

        # Drawing a rectangle on copied image
        rect = cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 255, 255), 2)

        cv2.imshow('cnt', rect)
        cv2.waitKey()

        # Cropping the text block for giving input to OCR
        cropped = orig[y:y + h, x:x + w]

        # Apply OCR on the cropped image
        config = ('-l eng --oem 1 --psm 3')
        text = pytesseract.image_to_string(cropped, config=config)

    return text

out=contours_text(im_deskewed,contours)





#
with open('file.txt', mode = 'w') as f:
    f.write(out)