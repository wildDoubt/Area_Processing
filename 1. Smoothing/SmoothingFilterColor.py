import cv2
import numpy as np
from matplotlib import pyplot as plt

# 가우시안 필터

# 미디언 필터

# 평균 필터

class Filter:
    def __init__(self):
        pass

    def getAbs(self, image, x, y, _x, _y):
        return np.sum(np.abs(image[x][y] - image[_x][_y]))

    def getMedianColorIndex(self, image, x, y, mask_size):
        arr = []
        dist_x = mask_size // 2
        dist_y = mask_size // 2
        for i in range(x - dist_x, x + dist_x + 1):
            for j in range(y - dist_y, y + dist_y + 1):
                arr.append([self.getAbs(image, i, j, x, y), i, j])
        arr = sorted(arr)
        # print(arr)
        return arr[mask_size * mask_size // 2]

    def mFilter(self, image, mask_size):
        x = image.shape[0]
        y = image.shape[1]
        start_row_index, start_column_index = mask_size // 2, mask_size // 2
        end_row_index, end_column_index = x - mask_size // 2, y - mask_size // 2
        new_image = np.zeros((image.shape[0] - (mask_size // 2) * 2, image.shape[1] - (mask_size // 2) * 2))
        for i in range(start_row_index, end_row_index):
            for j in range(start_column_index, end_column_index):
                new_image[i - start_row_index][j - start_column_index] = self.getMedian(image, i, j, mask_size)
        return new_image

    def mFilterColor(self, image, mask_size):
        x = image.shape[0]
        y = image.shape[1]
        start_row_index, start_column_index = mask_size // 2, mask_size // 2
        end_row_index, end_column_index = x - mask_size // 2, y - mask_size // 2
        new_image = np.zeros((image.shape[0] - (mask_size // 2) * 2, image.shape[1] - (mask_size // 2) * 2, 3))
        for i in range(start_row_index, end_row_index):
            for j in range(start_column_index, end_column_index):
                print(f'({i}, {j})')
                # image[i][j] = self.getMedian(image, i, j, mask_size)
                for k in range(3):
                    _x = self.getMedianColorIndex(image, i, j, mask_size)[1]
                    _y = self.getMedianColorIndex(image, i, j, mask_size)[2]
                    new_image[i - start_row_index][j - start_column_index][k] = image[_x][_y][k]
        return new_image


def getFileName(s):
    return s.split('/')[-1].split('.')[0]


if __name__ == "__main__":
    file_paths = ['../images/Color_noisy/Gaussian_noise.png',
                  '../images/Color_noisy/Lena_noise.png',
                  '../images/Color_noisy/Salt_pepper_noise.png']
    images = []
    s_images = []
    for file_path in file_paths:
        images.append(cv2.imread(file_path, cv2.IMREAD_COLOR))
    f = Filter()


    for i in range(3, 8, 2):
        for j in range(len(images)):
            print(f"{'../result/Color_noisy/' + getFileName(file_paths[j]) + '_median' + str(i) + '.jpg'} 진행 중")
            cv2.imwrite('../result/Color_noisy/' + getFileName(file_paths[j]) + '_median' + str(i) + '.jpg',
                        cv2.medianBlur(images[j], i))
            print(f"{'../result/Color_noisy/' + getFileName(file_paths[j]) + '_median' + str(i) + '.jpg'} 저장 완료")
            print(f"{'../result/Color_noisy/' + getFileName(file_paths[j]) + '_gaussian' + str(i) + '.jpg'} 진행 중")
            cv2.imwrite('../result/Color_noisy/' + getFileName(file_paths[j]) + '_gaussian' + str(i) + '.jpg',
                        cv2.GaussianBlur(images[j], (i, i), 0))
            print(f"{'../result/Color_noisy/' + getFileName(file_paths[j]) + '_gaussian' + str(i) + '.jpg'} 저장 완료")
