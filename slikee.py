import cv2
import numpy as np
import os

# Ova skripta pravi crno bele slike od svih pocetnih generisanih
rootdir = 'sve'

prosao = []
brojac = 0
poslednje = ""
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        podaci = [file[0]]
        if file[0] != poslednje:
            poslednje = file[0]
            brojac = 0
        putanjaSlike = os.path.join(subdir, file)
        img = cv2.imread(putanjaSlike)

        lower_thresh1 = 129
        upper_thresh1 = 255
        gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 5, 55])
        upper_red = np.array([170, 170, 255])

        mask = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(img, img, mask=mask)
        value = (35, 35)
        blurred = cv2.GaussianBlur(gr, value, 0)
        _, thresh1 = cv2.threshold(blurred, lower_thresh1, upper_thresh1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        a = "test/" + str(file[0]) + str(brojac)
        cv2.imwrite(a + ".png", thresh1)
        brojac += 1
