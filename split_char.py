import cv2
import numpy as np
import uuid
import os

MIN_CONTOUR_AREA = 100
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

PAD_IMAGE_SIZE = 128


# pad to keep the shape so that the character will not be stretched
def pad(img, w, h):
    tar = PAD_IMAGE_SIZE

    if w > h:
        tar_w = 0
        tar_h = (w-h) // 2
    else:
        tar_h = 0
        tar_w = (h-w) // 2

    img = np.pad(img, ((tar_h + h//8, tar_h+h//8), (tar_w+w//8, tar_w+w//8)),  'constant', constant_values=(255, 255))
    return cv2.resize(img, (tar, tar))


# split the character from the image
def split_char(path, save=False):
    ValidChars = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                  'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                  'U', 'V', 'W', 'X', 'Y', 'Z'}

    # convert the image to black and white and find the contour
    pic = cv2.imread(path)
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    pic = cv2.GaussianBlur(pic, (5, 5), 0)
    pic = cv2.adaptiveThreshold(pic, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    pic_r = cv2.bitwise_not(pic)

    cnt, contour, hierarchy = cv2.findContours(pic_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    chars = []
    for i in range(len(contour)):
        if cv2.contourArea(contour[i]) <= MIN_CONTOUR_AREA:
            continue
        x, y, w, h = cv2.boundingRect(contour[i])
        img_roi = pic[y:y + h, x:x + w]

        # TODO: we can sort the character by x and y if we need to
        # if save is True, it's a simple labeling tool. Just press the corresponding key on the keyboard.
        if save:
            img_roi = cv2.resize(img_roi, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
            cv2.imshow('1', img_roi)
            intChar = chr(cv2.waitKey())
            if intChar in ValidChars:
                dir_path = os.path.join('.', 'data', intChar)
                if not os.path.isdir(dir_path):
                    os.mkdir(dir_path)
                path = os.path.join(dir_path, str(uuid.uuid4()) + '.jpg')
                cv2.imwrite(path, img_roi)
            else:
                print('not valid')
        else:
            img_pad = pad(img_roi, w, h)
            chars.append(img_pad)
    return chars


def main():
    split_char("training_chars2.png", save=True)


if __name__ == "__main__":
    main()
