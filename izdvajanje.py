import os
import cv2

# Ova skripta sluzi za izdvajanje slika za test iz skupa train slika
rootdir = 'train'

prosao = []
brojac = 0
poslednje = ""
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        podaci = [file[0]]
        putanjaSlike = os.path.join(subdir, file)
        b = putanjaSlike.split(file[0])[1]
        b = b[:-4]
        print(b)
        if (int(b) <= 30):
            img = cv2.imread(putanjaSlike)
            cv2.imwrite("test/" + str(file), img)
        brojac += 1
