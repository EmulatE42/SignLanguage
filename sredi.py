from PIL import Image
import csv
import os

# Ova skripta generise csv fajlove za train i test crno bele slike. Potom se csv fajlovi u Cnn salju mrezi
rootdir = 'train'  # menjamo train/test

size = 100


def makeCSV():
    heder = ["label"]
    for i in range(1, size * size + 1):
        heder.append("pixel" + str(i))

    with open('train.csv', 'w', newline='') as f:  # menjamo train/test
        w = csv.writer(f)
        w.writerow(heder)

        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                podaci = [file[0]]
                putanjaSlike = os.path.join(subdir, file)

                img = Image.open(putanjaSlike).convert('L')
                img = img.resize((size, size))

                WIDTH, HEIGHT = img.size

                data = list(img.getdata())
                data = [data[offset:offset + WIDTH] for offset in range(0, WIDTH * HEIGHT, WIDTH)]
                # !python3
                for row in data:
                    for z in row:
                        podaci.append(z)
                w.writerow(podaci)


makeCSV()
