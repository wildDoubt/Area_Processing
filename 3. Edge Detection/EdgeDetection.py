import cv2
import numpy as np
import math
from matplotlib import pyplot as plt


def edge_convolution(_image, mask_x):
    mask_y = np.flip(mask_x.T, axis=0)
    image_x = cv2.filter2D(_image, -1, mask_x)
    image_y = cv2.filter2D(_image, -1, mask_y)

    # _result = np.sqrt(np.square(image_x) + np.square(image_y))
    # _result *= 255.0 / _result.max()
    _result = np.sqrt(image_x ** 2 + image_y ** 2)
    _result *= 255.0 / np.max(_result)
    return _result


def getFileName(s):
    return s.split('/')[-1].split('.')[0]


def getGaussianKernel(size, sigma=1):
    # 홀수 사이즈만 들어온다고 가정
    # sigma는 1이라고 가정하고 구현
    a = np.array([size - size // 2 - abs(i - size // 2 - 1) for i in range(1, size + 1)])
    return np.outer(a, a.T)


def LoG(x, y, sigma):
    # Formatted this way for readability
    nom = ((y ** 2) + (x ** 2) - 2 * (sigma ** 2))
    denom = ((2 * math.pi * (sigma ** 6)))
    expo = math.exp(-((x ** 2) + (y ** 2)) / (2 * (sigma ** 2)))
    return nom * expo / denom


def getLoGMask(mask_size):
    sigma = 1.4
    w = math.ceil(mask_size * sigma)
    if w % 2 == 0:
        w += 1
    new_mask = []
    for i in range(-w // 2, w // 2, 1):
        for j in range(-w // 2, w // 2, 1):
            # print(f'({i}, {j})')
            new_mask.append(LoG(i, j, 1))
    new_mask = np.array(new_mask)
    return new_mask.reshape(w, w)


if __name__ == "__main__":
    file_paths = ['../images/High_boost_filter_image/fig3_original.jpg',
                  '../images/High_boost_filter_image/fig4_original.jpg',
                  '../images/High_boost_filter_image/fig5_original.jpg',
                  '../images/High_boost_filter_image/fig6_original.png']

    images = []
    sobel_mask = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    prewitt_mask = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    LoG_mask = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    for file_path in file_paths:
        images.append(cv2.imread(file_path, 0))
    # for j in range(len(images)):
    #     result1 = cv2.GaussianBlur(images[j], (5, 5), 1)
    #     result1 = edge_convolution(result1, sobel_mask).astype(np.float)
    #     result1[result1 > 180] = 255
    #     result1[result1 < 180] = 0
    #
    #     result2 = cv2.GaussianBlur(images[j], (5, 5), 1)
    #     result2 = edge_convolution(result2, prewitt_mask).astype(np.float)
    #     result2[result2 > 200] = 255
    #     result2[result2 < 200] = 0
    #     # cv2.imwrite('test.jpg', result)
    #
    #
    #     cv2.imwrite('../result/Edge_detection/' + getFileName(file_paths[j]) + '_sobel' + '.jpg',
    #                 result1)
    #     cv2.imwrite('../result/Edge_detection/' + getFileName(file_paths[j]) + '_prewitt' + '.jpg',
    #                 result2)
    # result = edge_convolution(images[3], create_log(3)).astype(np.float)

    for i in range(3, 10, 2):
        for j in range(len(images)):
            result = cv2.filter2D(images[j], -1, getLoGMask(i)) * 255.0
            cv2.imwrite('../result/Edge_detection/' + getFileName(file_paths[j]) + '_LoG' + str(i) + '.jpg',
                        result)
    temp = getLoGMask(3)
    temp = np.round(temp, 2)
    print(temp)