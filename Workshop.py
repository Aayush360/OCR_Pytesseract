import pytesseract
import cv2
import numpy as np
import os

path='licimg'
per =50
roi = [[(367, 309), (699, 339), 'DLNO'],
       [(304, 379), (702, 419), 'BG'],
       [(914, 322), (1422, 364), 'NAME'],
       [(984, 397), (1427, 437), 'ADDRESS'],
       [(924, 602), (1289, 639), 'DOB'],
       [(997, 682), (1379, 729), 'F/H NAME'],
       [(1102, 764), (1427, 804), 'CITIZENSHIP NO'],
       [(1689, 787), (1794, 819), 'CATEGORY'],
       [(1059, 827), (1389, 872), 'PASSPORT'],
       [(1004, 897), (1302, 949), 'PHONE'],
       [(324, 784), (639, 844), 'DOI'], [(349, 862), (629, 904), 'DOE']]



imgQ = cv2.imread('template/temp.jpeg')
h,w,c = imgQ.shape
orb = cv2.ORB_create(1000) # helps to find the transformation matrix for the image
kp1,des1 = orb.detectAndCompute(imgQ,None)

my_pic_list=os.listdir(path)
print(my_pic_list)

for j,y in enumerate(my_pic_list):
    img = cv2.imread(path+'/'+y)
    # cv2.imshow(y,img)
    kp2, des2 = orb.detectAndCompute(img, None) #ORB (Oriented FAST and Rotated BRIEF)
    # An efficient alternative to SIFT or SURF
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2,des1)
    matches.sort(key=lambda x: x.distance) # lower the distance better the match
    good_matches = matches[:int(len(matches)*(per/100))] #gives 25% of the best matches
    img_match = cv2.drawMatches(img,kp2,imgQ,kp1,good_matches[:20],None,flags=2)
    # cv2.imshow(y,img_match)

    src_points = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    dst_points = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    # finding the relationship is called homography
    M,_ = cv2.findHomography(src_points,dst_points,cv2.RANSAC,5.0)
    img_scan =cv2.warpPerspective(img,M,(w,h))
    # cv2.imshow(y,img_scan)

    img_show = img_scan.copy()
    img_mask = np.zeros_like(img_show)

    for x,r in enumerate(roi):
        cv2.rectangle(img_mask,(r[0][0],r[0][1]),(r[1][0],r[1][1]),(0,255,0),cv2.FILLED)
        img_show = cv2.addWeighted(img_show,0.99,img_mask,0.1,0)
    cv2.imshow(y,img_show)






# imkp1 = cv2.drawKeypoints(imgQ,kp1,None)
# cv2.imshow('Output',imgQ)

# cv2.imshow('Keypoint',imkp1)
cv2.waitKey(0)
