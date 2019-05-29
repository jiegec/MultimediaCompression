from PIL import Image
import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

def dct2d(data):
    return fftpack.dct(fftpack.dct(data, norm='ortho').T, norm='ortho').T

def idct2d(data):
    return fftpack.idct(fftpack.idct(data, norm='ortho').T, norm='ortho').T

Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]], dtype=np.float)

def quantize(matrix, data, a):
    qq = a * matrix
    return np.round(data / qq) * qq

def psnr(data1, data2):
    mse = np.mean((data1 - data2) ** 2)
    psnr = 10 * np.log10(255.0 ** 2 / mse)
    return psnr

img = Image.open('lena_grayscale.png')
data = np.array(img, dtype=np.float)
print('orig img', data)

x, y = data.shape
psnr_8x8_quan = [0.0] * 100

for i in range(int(x/8)):
    for j in range(int(y/8)):
        submatrix = data[i*8:(i+1)*8, j*8:(j+1)*8]
        dct = dct2d(submatrix)

        for quan in range(1, 100):
            idct_quan = idct2d(quantize(Q, dct, quan / 50.0))
            psnr_8x8_quan[quan] += psnr(submatrix, idct_quan)


for quan in range(1, 100):
    print('psnr 2ddct 8x8 a=%f:' % (quan / 50), psnr_8x8_quan[quan] / (x / 8) / (y / 8))

plt.plot(np.arange(1, 100) / 50, np.array(psnr_8x8_quan[1:100]) / (x / 8) / (y / 8))
plt.xlabel('a')
plt.ylabel('PSNR')
plt.savefig('plot.png')

