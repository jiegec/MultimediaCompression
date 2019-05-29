from PIL import Image
import numpy as np
from scipy import fftpack

def dct2d(data):
    return fftpack.dct(fftpack.dct(data, norm='ortho').T, norm='ortho').T

def idct2d(data):
    return fftpack.idct(fftpack.idct(data, norm='ortho').T, norm='ortho').T

Q_canon = np.array([[1, 1, 1, 2, 3, 6, 8, 10],
    [1, 1, 2, 3, 4, 8, 9, 8],
    [2, 2, 2, 3, 6, 8, 10, 8],
    [2, 2, 3, 4, 7, 12, 11, 9],
    [3, 3, 8, 11, 10, 16, 15, 11],
    [3, 5, 8, 10, 12, 15, 16, 13],
    [7, 10, 11, 12, 15, 17, 17, 14],
    [14, 13, 13, 15, 15, 14, 14, 14]], dtype=np.float)

Q_nikon = np.array([[2, 1, 1, 2, 3, 5, 6, 7],
    [1, 1, 2, 2, 3, 7, 7, 7],
    [2, 2, 2, 3, 5, 7, 8, 7],
    [2, 2, 3, 3, 6, 10, 10, 7],
    [2, 3, 4, 7, 8, 13, 12, 9],
    [3, 4, 7, 8, 10, 12, 14, 11],
    [6, 8, 9, 10, 12, 15, 14, 12],
    [9, 11, 11, 12, 13, 12, 12, 12]], dtype=np.float)

def quantize(matrix, data, a):
    qq = a * matrix
    return np.round(data / qq) * qq

def psnr(data1, data2):
    mse = np.mean((data1 - data2) ** 2)
    psnr = 10 * np.log10(255.0 ** 2 / mse)
    return psnr

img = Image.open('custom_grayscale.png')
data = np.array(img, dtype=np.float)
x, y = data.shape

def measure_q(Q):
    psnr_ans = 0.0
    for i in range(int(x/8)):
        for j in range(int(y/8)):
            submatrix = data[i*8:(i+1)*8, j*8:(j+1)*8]
            dct = dct2d(submatrix)

            idct = idct2d(quantize(Q, dct, 1))
            psnr_ans += psnr(submatrix, idct)
    return psnr_ans / (x / 8) / (y / 8)

print('psnr 2ddct 8x8 canon:', measure_q(Q_canon))
print('psnr 2ddct 8x8 nikon:', measure_q(Q_nikon))

best_psnr = 0
best_q = Q_canon

for gen in range(0, 1000):
    temp_q = (Q_canon + Q_nikon) / 2
    temp_q += np.random.rand(8, 8) * 4 - 2
    temp_q = np.round(temp_q)
    psnr_current = measure_q(temp_q)
    if psnr_current > best_psnr:
        best_psnr = psnr_current
        best_q = temp_q


print('psnr 2ddct 8x8 best:', best_psnr)
print('Q:', best_q)
