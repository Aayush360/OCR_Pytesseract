from easyocr import Reader
import argparse
import cv2
import os

def cleanup_text(text):
    '''strip out non-ascii text so we can the detected text on the image using opencv'''
    return "".join([c if ord(c)<128 else "" for c in text]).strip()


def write_to_file(filename, text):
    if os.path.exists(filename):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not

    highscore = open(filename, append_write)
    highscore.write(text + '\n')
    highscore.close()


# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required=True, help="path to input image to be OCR'd")
ap.add_argument("-l","--langs",type=str,default="en", help="comma separated list of languages to OCR")
ap.add_argument("-g", "--gpu", type=int, default=-1,help="whether or not GPU should be used")
args = vars(ap.parse_args())

# break the input languages into comma separated list
langs = args["langs"].split(",")
print("[INFO] OCR'ing with the following languages: {}".format(langs))

# load the input image from the disk
image = cv2.imread(args["image"])

# OCR the input image using EasyOCR
print("[INFO] OCR'ing the input image ...  ")
reader = Reader(langs,gpu=args["gpu"] > 0)
results = reader.readtext(image)





# loop over the results
for(bbox,text,prob) in results:
    # display the ocr'd text and associated probability
    print("[INFO] {:.4f}: {}".format(prob,text))

    # unpack the bounding box
    (tl,tr,br,bl) = bbox
    tl = (int(tl[0]),int(tl[1]))
    tr = (int(tr[0]),int(tr[1]))
    bl = (int(bl[0]),int(bl[1]))
    br = (int(br[0]),int(br[1]))

    # cleanup the text and draw the box surrounding the text along with OCR'd text itself
    text = cleanup_text(text)
    cv2.rectangle(image,tl,br,(0,255,0),2)
    cv2.putText(image,text,(tl[0],tl[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0),2)
    write_to_file(filename="file_easy_ocr.txt",text=text)





# show the output image
cv2.imshow("Image",image)
cv2.waitKey(0)

