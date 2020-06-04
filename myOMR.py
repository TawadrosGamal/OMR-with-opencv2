import cv2
import imutils as ml
import numpy as np
from imutils import contours
from imutils.perspective import four_point_transform
pathes=[]
pathes.append('E:\\ML\\projects\\optical-mark-recognition\\myIMG\\1.png')
pathes.append('E:\\ML\\projects\\optical-mark-recognition\\myIMG\\2.png')
pathes.append('E:\\ML\\projects\\optical-mark-recognition\\myIMG\\3.png')
pathes.append('E:\\ML\\projects\\optical-mark-recognition\\myIMG\\4.png')
pathes.append('E:\\ML\\projects\\optical-mark-recognition\\myIMG\\5.png')
print("this system will calculate all exam papers ,write stop to stop at any time")
strs=input()
strs.lower()

for i in pathes :



    if strs=='stop':
        exit()


    Answers = {0: 0, 1: 4, 2: 0, 3: 3, 4: 1}
    # load the image, convert it to grayscale, blur it
    # slightly, then find edges
    image = cv2.imread(i)
    imgray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    imgblur = cv2.GaussianBlur(src=imgray, ksize=(5, 5), sigmaX=0)
    imgedged = cv2.Canny(image=imgblur, threshold1=75, threshold2=200)
    cv2.imshow('Edged IMG',imgedged)
    cv2.waitKey()
    # find contours in the edge map, then initialize
    # the contour that corresponds to the document
    Contours = cv2.findContours(image=imgedged.copy(), mode=cv2.RETR_EXTERNAL,
                                method=cv2.CHAIN_APPROX_SIMPLE)
    Contours = ml.grab_contours(cnts=Contours)
    docCnt = None

    # ensure that at least one contour was found
    if len(Contours) > 0:
        # sort the contours according to their size in
        # descending order
        Contours = sorted(Contours, key=cv2.contourArea, reverse=True)

        # loop over the sorted contours
        for c in Contours:
            # approximate the contour
            peri = cv2.arcLength(curve=c, closed=True)
            approx = cv2.approxPolyDP(curve=c, epsilon=0.02 * peri, closed=True)

            # if our approximated contour has four points,
            # then we can assume we have found the paper
            if len(approx) == 4:
                docCnt = approx
                break

    # apply a four point perspective transform to both the
    # original image and grayscale image to obtain a top-down
    # birds eye view of the paper
    paper = four_point_transform(image=image, pts=docCnt.reshape(4, 2))
    warped = four_point_transform(image=imgray, pts=docCnt.reshape(4, 2))
    cv2.imshow('four point view ',warped)
    cv2.waitKey()
    # apply Otsu's thresholding method to binarize the warped
    # piece of paper
    thresh = cv2.threshold(src=warped, thresh=0, maxval=255,
                           type=cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # find contours in the thresholded image, then initialize
    # the list of contours that correspond to questions
    Contours = cv2.findContours(image=thresh.copy(), mode=cv2.RETR_EXTERNAL,
                                method=cv2.CHAIN_APPROX_SIMPLE)
    Contours = ml.grab_contours(cnts=Contours)
    questionCnts = []

    # loop over the contours
    for c in Contours:
        # compute the bounding box of the contour, then use the
        # bounding box to derive the aspect ratio
        (x, y, width, heigh) = cv2.boundingRect(array=c)
        aspectRatio = width / float(heigh)

        # in order to label the contour as a question, region
        # should be sufficiently wide, sufficiently tall, and
        # have an aspect ratio approximately equal to 1
        if width >= 20 and heigh >= 20 and aspectRatio >= 0.9 and aspectRatio <= 1.1:
            questionCnts.append(c)

    # sort the question contours top-to-bottom, then initialize
    # the total number of correct answers
    questionCnts = contours.sort_contours(cnts=questionCnts,
                                          method="top-to-bottom")[0]
    correct = 0

    # each question has 5 possible answers, to loop over the
    # question in batches of 5
    numq = 5
    for (q, i) in enumerate(np.arange(0, len(questionCnts), numq)):
        # sort the contours for the current question from
        # left to right, then initialize the index of the
        # bubbled answer
        Contours = contours.sort_contours(cnts=questionCnts[i:i + numq])[0]
        bubbled = None

        # loop over the sorted contours
        for (j, c) in enumerate(Contours):
            # construct a mask that reveals only the current
            # "bubble" for the question
            mask = np.zeros(shape=thresh.shape, dtype="uint8")
            cv2.drawContours(image=mask, contours=[c], contourIdx=-1, color=255, thickness=-1)

            # apply the mask to the thresholded image, then
            # count the number of non-zero pixels in the
            # bubble area
            mask = cv2.bitwise_and(src1=thresh, src2=thresh, mask=mask)
            total = cv2.countNonZero(src=mask)

            # if the current total has a larger number of total
            # non-zero pixels, then we are examining the currently
            # bubbled-in answer
            if bubbled is None or total > bubbled[0]:
                bubbled = (total, j)

        # initialize the contour color and the index of the
        # *correct* answer
        color = (0, 0, 255)
        k = Answers[q]

        # check to see if the bubbled answer is correct
        if k == bubbled[1]:
            color = (0, 255, 0)
            correct += 1

        # draw the outline of the correct answer on the test
        cv2.drawContours(image=paper, contours=[Contours[k]], contourIdx=-1, color=color, thickness=3)

    # grab the test taker
    score = (correct / float(numq)) * 100
    print("[INFO] score:Test "+str(i)+" {:.2f}%".format(score))


    cv2.putText(img=paper, text="{:.2f}%".format(score), org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9, color=(0, 0, 255), thickness=2)
    cv2.imshow("Original", image)
    cv2.imshow("Exam", paper)
    cv2.waitKey(0)
    print("we have calculated exam number " + str(i) + "press any key to continue ")
    strs = str(input())
    strs.lower()
