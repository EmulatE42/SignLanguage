import sys
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUi
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import numpy as np
from keras.models import model_from_json


class MyWindow(QDialog):
    def __init__(self):
        super(MyWindow, self).__init__()
        loadUi('a.ui', self)
        self.frame = 0
        self.image = None
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.loaded_model = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.prepoznavanje = False
        self.load()
        self.timer.start()
        self.label_2.setStyleSheet('color: red')
        self.label_3.setStyleSheet('color: red')
        self.label_4.setStyleSheet('color: red')
        self.label_5.setStyleSheet('color: red')
        self.trenutnoSlovo = ""
        self.word = ""
        self.pushButton.clicked.connect(self.delete)

    def delete(self):
        if len(self.word) > 0:
            self.word = self.word[:len(self.word) - 1]
            tekst = self.label_5.text()
            self.label_5.setText(tekst[:len(tekst) - 1])

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_5:
            self.prepoznavanje = not self.prepoznavanje
            if (self.prepoznavanje):
                self.label_3.setText("Recognizing - 5 ON")
            else:
                self.label_3.setText("Recognizing - 5 OFF")
                self.label_2.setText("The Letter is :")

        elif event.key() == Qt.Key_6:
            if self.prepoznavanje:
                self.label_5.setText(self.label_5.text() + self.trenutnoSlovo)
                self.word += self.trenutnoSlovo

    def rec(self):
        x = 50
        y = 50
        w = 300
        h = 300
        lower_thresh1 = 129
        upper_thresh1 = 255
        img = self.image[y:y + h, x:x + w]
        gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        value = (35, 35)
        blurred = cv2.GaussianBlur(gr, value, 0)
        _, thresh1 = cv2.threshold(blurred, lower_thresh1, upper_thresh1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh1 = cv2.resize(thresh1, (100, 100))

        noviElement = thresh1[np.newaxis, :]
        predvidjen = self.loaded_model.predict(np.array([noviElement]))
        indeks = np.argmax(predvidjen)
        SVI = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.trenutnoSlovo = str(SVI[indeks])
        self.label_2.setText("The Letter is :  " + str(SVI[indeks]))

    def here(self):

        x = 50
        y = 50
        w = 300
        h = 300
        lower_thresh1 = 129
        upper_thresh1 = 255
        img = self.image[y:y + h, x:x + w]
        gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        value = (35, 35)
        blurred = cv2.GaussianBlur(gr, value, 0)
        _, thresh1 = cv2.threshold(blurred, lower_thresh1, upper_thresh1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cv2.imshow('Binary Image', thresh1)  #

    def load(self):
        json_file = open('SIGN.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)
        self.loaded_model.load_weights("SIGN.h5")

    def update_frame(self):
        ret, self.image = self.capture.read()
        self.frame += 1
        self.image = cv2.flip(self.image, 1)

        #self.here()
        if (self.prepoznavanje):
            self.rec()

        x = 50
        y = 50
        w = 300
        h = 300
        cv2.rectangle(self.image, (x, y), (x + w, y + h), (255, 255, 00), 2)
        self.displayImage(self.image, 1)

    def displayImage(self, img, window=1):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        outImage = outImage.rgbSwapped()
        if window == 1:
            self.label.setPixmap(QPixmap.fromImage(outImage))
            self.label.setScaledContents(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()

    sys.exit(app.exec_())
