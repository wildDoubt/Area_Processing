import cv2
import numpy as np
from matplotlib import pyplot as plt


# images = [cv2.imread('../images/Grayscale_noisy/fig1_original.jpg', 0),
#           cv2.imread('../images/Grayscale_noisy/fig1_gaussian_noise.jpg', 0),
#           cv2.imread('../images/Grayscale_noisy/fig1_saltnpepper_noise.jpg', 0),
#           cv2.imread('../images/Grayscale_noisy/fig2_original.jpg', 0),
#           cv2.imread('../images/Grayscale_noisy/fig2_saltnpepper_noise.jpg', 0),
#           cv2.imread('../images/Grayscale_noisy/fig2_gaussian_noise.jpg', 0)]
#
# cv2.imshow('test', images[0])
# cv2.waitKey(5000)
# cv2.destroyAllWindows()


# 가우시안 필터

# 미디언 필터

# 평균 필터

class Filter:
    def __init__(self):
        pass

    def getGaussianKernel(self, size, sigma=1):
        # 홀수 사이즈만 들어온다고 가정
        # sigma는 1이라고 가정하고 구현
        a = np.array([size - size // 2 - abs(i - size // 2 - 1) for i in range(1, size + 1)])
        return np.outer(a, a.T)

    def getAverageKernel(self, size):
        return np.ones((size, size), dtype=np.int)

    def calculationPoint(self, image, x, y, mask):
        total_sum = 0
        dist_x = mask.shape[0] // 2
        dist_y = mask.shape[1] // 2
        for i in range(x - dist_x, x + dist_x + 1):
            for j in range(y - dist_y, y + dist_y + 1):
                total_sum += image[i][j] * mask[i - (x - dist_x)][j - (y - dist_y)]
        return total_sum / np.sum(mask)

    def getMedian(self, image, x, y, mask_size):
        arr = []
        dist_x = mask_size // 2
        dist_y = mask_size // 2
        for i in range(x - dist_x, x + dist_x + 1):
            for j in range(y - dist_y, y + dist_y + 1):
                arr.append(image[i][j])
        return np.median(np.array(arr))

    def convolution(self, image, mask, mask_size):
        x = image.shape[0]
        y = image.shape[1]
        start_row_index, start_column_index = mask_size // 2, mask_size // 2
        end_row_index, end_column_index = x - mask_size // 2, y - mask_size // 2
        for i in range(start_row_index, end_row_index):
            for j in range(start_column_index, end_column_index):
                image[i][j] = self.calculationPoint(image, i, j, mask)
        return image

    def gFilter(self, image, mask_size, sigma=1):
        mask = self.getGaussianKernel(mask_size, sigma)
        return self.convolution(image, mask, mask_size)

    def mFilter(self, image, mask_size):
        x = image.shape[0]
        y = image.shape[1]
        start_row_index, start_column_index = mask_size // 2, mask_size // 2
        end_row_index, end_column_index = x - mask_size // 2, y - mask_size // 2
        for i in range(start_row_index, end_row_index):
            for j in range(start_column_index, end_column_index):
                image[i][j] = self.getMedian(image, i, j, mask_size)
        return image

    def aFilter(self, image, mask_size):
        mask = self.getAverageKernel(mask_size)
        return self.convolution(image, mask, mask_size)


def getFileName(s):
    return s.split('/')[-1].split('.')[0]


if __name__ == "__main__":
    file_paths = ['../images/Grayscale_noisy/fig1_original.jpg',
                  '../images/Grayscale_noisy/fig1_gaussian_noise.jpg',
                  '../images/Grayscale_noisy/fig1_saltnpepper_noise.jpg',
                  '../images/Grayscale_noisy/fig2_original.jpg',
                  '../images/Grayscale_noisy/fig2_saltnpepper_noise.jpg',
                  '../images/Grayscale_noisy/fig2_gaussian_noise.jpg']
    images = []
    for file_path in file_paths:
        images.append(cv2.imread(file_path, 0))
    f = Filter()
    for i in range(3, 8, 2):
        for j in range(len(images)):
            print(f"{'../result/Grayscale_noisy/' + getFileName(file_paths[j]) + '_gaussian' + str(i) + '.jpg'} 진행 중")
            cv2.imwrite('../result/Grayscale_noisy/' + getFileName(file_paths[j]) + '_gaussian' + str(i) + '.jpg',
                        f.gFilter(images[j], i))
            print(f"{'../result/Grayscale_noisy/' + getFileName(file_paths[j]) + '_gaussian' + str(i) + '.jpg'} 저장 완료")

            print(f"{'../result/Grayscale_noisy/' + getFileName(file_paths[j]) + '_median' + str(i) + '.jpg'} 진행 중")
            cv2.imwrite('../result/Grayscale_noisy/' + getFileName(file_paths[j]) + '_median' + str(i) + '.jpg',
                        f.mFilter(images[j], i))
            print(f"{'../result/Grayscale_noisy/' + getFileName(file_paths[j]) + '_median' + str(i) + '.jpg'} 저장 완료")

            print(f"{'../result/Grayscale_noisy/' + getFileName(file_paths[j]) + '_average' + str(i) + '.jpg'} 진행 중")
            cv2.imwrite('../result/Grayscale_noisy/' + getFileName(file_paths[j]) + '_average' + str(i) + '.jpg',
                        f.aFilter(images[j], i))
            print(f"{'../result/Grayscale_noisy/' + getFileName(file_paths[j]) + '_average' + str(i) + '.jpg'} 저장 완료")
