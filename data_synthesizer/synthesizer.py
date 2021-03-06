import random
from os import listdir
from os.path import join, isdir

import cv2
import numpy as np
import pandas as pd
from PIL import ImageFont, ImageDraw, Image

bg_path = 'data/backgrounds'  # you can modify the background images in this directory
kharbocha_path = 'data/khrbocha'
ha_path = 'data/ha'  # This is a special character that I didn't find how to write it

'''In the following two lists, put your fonts directories, you can download fonts in fonts.google.com'''

ARABIC_FONTS = ['data/Fonts/Amiri', 'data/fonts/Lateef', 'data/fonts/Scheherazade_New',
                'data/fonts/Aref_Ruqaa',
                'data/fonts/Harmattan',
                ]
LATIN_FONTS = ['data/fonts/Barlow', 'data/fonts/Be_Vietnam_Pro', 'data/fonts/Bebas_Neue',
               'data/fonts/Fira_Sans',
               'data/fonts/Montserrat', 'data/fonts/Open_Sans', 'data/fonts/Anton',
               'data/fonts/License-Plate',
               'data/fonts/Teko', 'data/fonts/Barlow_Condensed',
               ]

'''put here all your characters'''

CHARS = {'LETTERS': ['ق', 'س', 'ش', 'م', 'و', 'د', 'ج', 'ب', 'أ', 'المغرب', 'W', 'HA'],
         'DIGITS': [str(i) for i in range(0, 10)]}

# mapping characters with labels from 0 to num_labels
LABELS = {i: str(i) for i in range(0, 10)}
for i in range(len(CHARS['LETTERS'])):
    LABELS[i+10] = CHARS['LETTERS'][i]

'''extracting the true font's directories'''

ARABIC_FONTS_ttf = []
for path in ARABIC_FONTS:
    if len([f for f in listdir(path) if isdir(join(path, f))]) != 0:
        ARABIC_FONTS_ttf.append([join(*[path, 'static', f]) for f in listdir(join(path, 'static')) if 'ttf' in f])
    else:
        ARABIC_FONTS_ttf.append([join(path, f) for f in listdir(path) if 'ttf' in f])

LATIN_FONTS_ttf = []
for path in LATIN_FONTS:
    if len([f for f in listdir(path) if isdir(join(path, f))]) != 0:
        LATIN_FONTS_ttf.append([join(*[path, 'static', f]) for f in listdir(join(path, 'static')) if 'ttf' in f])
    else:
        LATIN_FONTS_ttf.append([join(path, f) for f in listdir(path) if 'ttf' in f])

'''transforming characters to real images, and organizing it in a dictionary'''

CHAR_IMGS = {}
for i in LABELS:
    CHAR_IMGS[i] = []
    if i <= 9 or LABELS[i] == 'W':
        for FONT in LATIN_FONTS_ttf:
            for style in FONT:
                frame = np.ones((160, 350, 3), dtype=np.uint8) * 255
                font = ImageFont.truetype(style, 150)
                img_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(img_pil)
                draw.text((175, 80), LABELS[i], font=font, fill=(0, 0, 0), anchor='mm')
                img = np.array(img_pil)
                img_thresh = cv2.threshold(img[:, :, 0], 20, 255, cv2.THRESH_BINARY)[1]
                img_binary = cv2.bitwise_not(img_thresh)
                x1, y1, w, h = cv2.boundingRect(img_binary)
                CHAR_IMGS[i].append(img[y1 - 3:y1 + h + 3, x1 - 3:x1 + w + 3])
    elif LABELS[i] == 'HA':
        for f in listdir(ha_path):
            im = cv2.imread(join(ha_path, f))
            CHAR_IMGS[i].append(im)
    else:
        for FONT in ARABIC_FONTS_ttf:
            for style in FONT:
                frame = np.ones((300, 600, 3), dtype=np.uint8) * 255
                font = ImageFont.truetype(style, 150)
                img_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(img_pil)
                draw.text((300, 150), LABELS[i], font=font, fill=(0, 0, 0), anchor='mm')
                img = np.array(img_pil)
                img_thresh = cv2.threshold(img[:, :, 0], 20, 255, cv2.THRESH_BINARY)[1]
                img_binary = cv2.bitwise_not(img_thresh)
                x1, y1, w, h = cv2.boundingRect(img_binary)
                CHAR_IMGS[i].append(img[y1 - 3:y1 + h + 3, x1 - 3:x1 + w + 3])

'''putting the special characters to be ignored in the images'''

FALSE_CHARS = []
for x in ['|', '|', '|', '|', '|', '|', '-', '*']:
    for fontpath in ['data/fonts/Barlow/Barlow-Regular.ttf',
                     'data/fonts/Barlow_Condensed/BarlowCondensed-Light.ttf',
                     'data/fonts/Teko/Teko-Bold.ttf']:
        frame = np.ones((200, 400, 3), dtype=np.uint8)*255
        font = ImageFont.truetype(fontpath, 150)
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        draw.text((175, 80), x, font=font, fill=(0,0,0), anchor='mm')
        FALSE_CHARS.append(np.array(img_pil)[25:150, 150:200])

'''Some artifacts to be applied on the characters'''

KHARBOCHA = [cv2.imread(join(kharbocha_path, p)) for p in listdir(kharbocha_path) if 'khrbocha' in p]

'''extracting the backgrounds in a list'''
BACKGROUNDS = []

for f in listdir(bg_path):
    im = cv2.imread(join(bg_path, f))#cv2.imread(bg_path + '/' +f)#
    im = cv2.resize(im, (450, 170))
    BACKGROUNDS.append(im)


class DataCreator:

    ''' this class that is responsible for generating plates by putting
     characchters in the backgrounds on random positions and random values '''

    def __init__(self, characters=CHAR_IMGS, labels=LABELS):
        self.background = None
        self.boxes = []
        self.chars = []
        self.alphas = []
        self.sigmas = []
        self.digits = [i for i in labels.keys() if LABELS[i].isnumeric()]
        self.letters = [i for i in labels.keys() if not LABELS[i].isnumeric()]
        self.CHARACTERS = characters

    def generate_background(self, backgrounds):
        self.background = np.copy(random.choice(backgrounds))

    def draw_shape(self, shape, xy, alpha, sigma):
        ''' draw a character in the background in a box given by it's coordinates
         in xy and with a value alpha and variance sigma'''
        ih, iw = shape.shape[0], shape.shape[1]  # image.shape
        h, w = xy[0][1] - xy[0][0], xy[1][1] - xy[1][0]
        # resize image
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        image = cv2.resize(shape, (nw, nh))
        b = np.ones((h, w, 3)) * 255
        b[dy:dy + nh, dx:dx + nw, :] = image
        if random.choices([True, False], [0.05, 0.95])[0]:
            kh = cv2.resize(random.choice(KHARBOCHA), (xy[1][1] - xy[1][0], xy[0][1] - xy[0][0]))
            b = 255 * (1 - b / 255) * (1 - kh / 255) + b
        crop = self.background[xy[0][0]:xy[0][1], xy[1][0]:xy[1][1]] * (b / 255) + \
               (1 - b / 255) * np.random.normal(alpha, sigma, (xy[0][1] - xy[0][0], xy[1][1] - xy[1][0], 3))
        self.background[xy[0][0]:xy[0][1], xy[1][0]:xy[1][1]] = crop

    def generate_aligned_boxes(self):
        '''generate boxes with eandom dimentions and positions, and associating labels to them'''
        Bh, Bw = self.background.shape[0], self.background.shape[1]
        h = random.randrange(44, 60)
        w = int(h * 73 / 111)
        dx = int((Bh - h) / 2)
        dy = int((Bw - 11 * w) / 2)
        for i in range(6):
            x_1, y_1 = dx, dy + i * w
            x_2, y_2 = dx + h, dy + (i + 1) * w
            self.boxes.append([(x_1, x_2), (y_1, y_2)])
            self.chars.append(random.choice(self.digits))
            self.alphas.append(random.randrange(0, 70))
            self.sigmas.append(random.randrange(5, 30))
        self.boxes.append([(dx, dx + h), (dy + 7 * w, dy + 8 * w)])
        self.boxes.append([(dx, dx + h), (dy + 9 * w, dy + 10 * w)])
        self.boxes.append([(dx, dx + h), (dy + 10 * w, dy + 11 * w)])
        false_char = random.choice(FALSE_CHARS)
        r1, r2 = random.randrange(dy + 6 * w, dy + 7 * w - int(5 * h / 12)), random.randrange(dy + 8 * w,
                                                                                              dy + 9 * w - int(
                                                                                                  5 * h / 12))
        self.draw_shape(false_char, [(dx, dx + h), (r1, r1 + int(5 * h / 12))], random.randrange(30, 60),
                        random.randrange(5, 30))
        self.draw_shape(false_char, [(dx, dx + h), (r2, r2 + int(5 * h / 12))], random.randrange(30, 60),
                        random.randrange(5, 30))
        self.chars += [random.choice(self.letters), random.choice(self.digits), random.choice(self.digits)]
        self.alphas += [random.randrange(0, 70), random.randrange(0, 70), random.randrange(0, 70)]
        self.sigmas += [random.randrange(5, 30), random.randrange(5, 30), random.randrange(5, 30)]

    def generate_random_boxes(self, n):
        '''generate boxes with eandom dimentions and positions, and associating labels to them'''
        Bh, Bw = self.background.shape[0], self.background.shape[1]
        for i in range(n):
            h, w = random.randrange(50, 150), random.randrange(20, 150)
            x_1, y_1 = random.randrange(20, Bh - 70), random.randrange(int(i * Bw / n), int((i + 1) * Bw / n) - 20)
            x_2, y_2 = min([Bh - 20, x_1 + h]), min([int((i + 1) * Bw / n), y_1 + w])
            self.boxes.append([(x_1, x_2), (y_1, y_2)])
            if random.choices([True, False], [0.1, 0.9])[0] and y_1 - 20 > 3:
                false_char = random.choice(FALSE_CHARS)
                self.draw_shape(false_char, [(x_1, x_2), (y_1 - 20, y_1 - 5)], random.randrange(30, 60),
                                random.randrange(5, 30))
            if random.choices([True, False], [0.9, 0.1])[0]:
                self.chars.append(random.choice(self.letters))
            else:
                self.chars.append(random.choice(self.digits))
            self.alphas.append(random.randrange(0, 70))
            self.sigmas.append(random.randrange(5, 30))

    def reset(self):
        self.background = None
        self.boxes = []
        self.chars = []
        self.alphas = []

    def create_image(self, BACKGROUNDS):
        self.reset()
        self.generate_background(BACKGROUNDS)
        if random.choices([True, False], [0.5, 0.5])[0]:
            self.generate_aligned_boxes()
        else:
            self.generate_random_boxes(random.randrange(5, 12))
        for i in range(len(self.boxes)):
            xy = self.boxes[i]
            alpha = self.alphas[i]
            sigma = self.sigmas[i]
            shape = random.choice(self.CHARACTERS[self.chars[i]])
            self.draw_shape(shape, xy, alpha, sigma)
        self.background = np.uint8(np.clip(self.background, 0.0, 255.0))
        return self.background


def generate(n, save=False):
    chars = []
    images = []
    y = {'id': [], 'annotations': []}
    for i in range(n):
        obj = DataCreator()
        image = obj.create_image(BACKGROUNDS)
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        y['id'].append(i)
        y['annotations'].append('')
        for j in range(len(obj.boxes)):
            [(x_1, x_2), (y_1, y_2)] = obj.boxes[j]
            class_id = obj.chars[j]
            chars.append(class_id)
            y['annotations'][-1] += (str(x_1) + ',' + str(y_1) + ',' + str(x_2) + ',' + str(y_2) + ',' + str(class_id))
            if j != len(obj.boxes) - 1:
                y['annotations'][-1] += ' '

    df = pd.DataFrame(data=y)
    images = np.array(images)

    if save:
        np.save('images_numpy_v2', np.array(images))
        df.to_csv('annotations_v2.csv')
    else:
        return images, df


if __name__ == '__main__':
    generate(20000, True)
