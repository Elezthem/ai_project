import cv2
#import numpy
import numpy as np
import imutils
import easyocr
from matplotlib import pyplot as pl

img = cv2.imread('images/car2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_filter = cv2.bilateralFilter(gray, 11, 15, 15)
edges = cv2.Canny(img_filter, 30, 200)

cont = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cont = imutils.grab_contours(cont)
cont = sorted(cont, key=cv2.contourArea, reverse=True)

pos = None
for c in cont:
    approx = cv2.approxPolyDP(c, 10, True)
    if len(approx) == 4:
        pos = approx
        break

mask = np.zeros(gray.shape, np.uint8)
new_img = cv2.drawContours(mask, [pos], 0, 255, -1)
bitwise_img = cv2.bitwise_and(img, img, mask=mask)

#pl.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
#pl.show()

(x, y) = np.where(mask == 255)
(x1, y1) = np.min(x), np.min(y)
(x2, y2) = np.max(x), np.max(y)
crop = gray[x1:x2, y1:y2]

text = easyocr.Reader(['en'])
text = text.readtext(crop)

res = text[0] [-2]
final_image = cv2.putText(img, res, (x1, y2 + 60), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 2)
final_image = cv2.rectangle(img, (x1, x2), (y1, y2), (0, 255, 0), 2)

pl.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
pl.show()

#img = cv2.imread('images/test.jpg')
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#faces = cv2.CascadeClassifier('faces.xml')

#results = faces.detectMultiScale(gray, scaleFactor=2, minNeighbors=4)

#for (x, y, w, h) in results:
#    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)

#cv2.imshow("Result", img)
#cv2.waitKey(0)




#photo = cv2.imread("images/ai.jpg")
#img = numpy.zeros(photo.shape[:2], dtype='uint8')

#circle = cv2.circle(img.copy(), (200, 300), 120, 255, -1)
#square = cv2.rectangle(img.copy(), (25, 25), (250, 750), 255, -1)

#img = cv2.bitwise_and(photo, photo, mask=square)
#img = cv2.bitwise_or(circle, square)
#img = cv2.bitwise_xor(circle, square)
#img = cv2.bitwise_not(square)

#img = cv2.imread("images/ai.jpg")

#img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

#img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#r, g, b = cv2.split(img)

#img = cv2.merge([b, g, r])

#new_img = np.zeros(img.shape, dtype='uint8')

#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img = cv2.GaussianBlur(img, (5, 5), 0)

#img = cv2.Canny(img, 100, 140)

#con, hir, = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

#cv2.drawContours(new_img, con, -1, (230, 111, 148), 1)

#print(con)

#cv2.imshow("Result", img)
#cv2.imshow("Result", img)
#cv2.waitKey(0)

#img = cv2.flip(img, -1)

#def rotate(img_param, angle):
#    height, width = img_param.shape[:2]
#    point = (width // 2, height // 2)

#    mat = cv2.getRotationMatrix2D(point, angle, 1)
#    return cv2.warpAffine(img, mat, (width, height))


#img = rotate(img, -90)

#def tranform(img_param, x, y):
 #   mat = np.float32([[1, 0, x], [0, 1, y]])
#    return cv2.warpAffine(img_param, mat, (img_param.shape[1], img_param.shape[0]))

#img = tranform(img, 30, 200)


#cap = cv2.VideoCapture("videos/admin.mp4")

#while True:
#    success, img = cap.read()
#    img = cv2.GaussianBlur(img, (9, 9), 0)
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#    img = cv2.Canny(img, 25, 25)
#    kernel = np.ones((5, 5), np.uint8)
#    img = cv2.dilate(img, kernel, iterations=1)

#    img = cv2.erode(img, kernel, iterations=1)

#    cv2.imshow("Result", img)

#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break

#photo = np.zeros((450, 450, 3), dtype='uint8')

#photo[10:150, 200:280] = 119, 201, 105

#cv2.rectangle(photo, (50,70), (100, 100), (119, 201, 105), thickness=cv2.FILLED)

#cv2.line(photo, (0, photo.shape[0] // 2), (photo.shape[1], photo.shape[0] // 2), (119, 201, 105), thickness=3)

#cv2.circle(photo, (photo.shape[1] // 2, photo.shape[0] // 2), 100, (119, 201, 105), thickness=1)

#cv2.putText(photo, 'Elezthem', (100, 150), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 1)

#cv2.imshow('Photo', photo)
#cv2.waitKey(0)

#img = cv2.imread('images/ai.jpg')

#new_img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
#img = cv2.GaussianBlur(img, (15, 15), 0)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#img = cv2.Canny(img, 200, 200)

#kernel = np.ones((5, 5), np.uint8)
#img = cv2.dilate(img, kernel, iterations=1)

#img = cv2.erode(img, kernel, iterations=1)

#img[0:100, 0:150]
#cv2.imshow('Result', img)

#print(img.shape)

#cv2.waitKey(0)

#cap = cv2.VideoCapture(0)
#cap.set(3, 500)
#cap.set(4, 300)

#while True:
#    success, img = cap.read()
#    cv2.imshow('Result', img)

#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
