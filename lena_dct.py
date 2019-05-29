from PIL import Image
import numpy as np
from scipy import fftpack


def dct2d(data):
    return fftpack.dct(fftpack.dct(data, norm='ortho').T, norm='ortho').T

def idct2d(data):
    return fftpack.idct(fftpack.idct(data, norm='ortho').T, norm='ortho').T

def matrix_select(data, side):
    x, y = data.shape
    result = np.zeros(data.shape)
    for i in range(int(x/side)):
        for j in range(int(y/side)):
            result[i,j] = data[i,j]
    return result

img = Image.open('lena_grayscale.png')
data = np.array(img, dtype=np.float)
print('orig img', data)

dct = dct2d(data)
Image.fromarray(dct.clip(0, 255).astype('uint8')).save('lena_1ddct.png')
Image.fromarray(dct.clip(0, 255).astype('uint8')).save('lena_2ddct.png')

idct = idct2d(dct)
Image.fromarray(idct.clip(0, 255).astype('uint8')).save('lena_1ddct_1didct.png')
Image.fromarray(idct.clip(0, 255).astype('uint8')).save('lena_2ddct_2didct.png')

mse = np.mean((data - idct) ** 2)
psnr = 10 * np.log10(255.0 ** 2 / mse)
print('idct img', idct)
print('psnr 2ddct:', psnr)

x, y = data.shape
idct_8x8 = np.zeros(data.shape)
idct_8x8_4 = np.zeros(data.shape)
idct_8x8_16 = np.zeros(data.shape)
idct_8x8_64 = np.zeros(data.shape)

for i in range(int(x/8)):
    for j in range(int(y/8)):
        submatrix = data[i*8:(i+1)*8, j*8:(j+1)*8]
        dct = dct2d(submatrix)
        Image.fromarray(dct.clip(0, 255).astype('uint8')).save('lena_2ddct_8x8/lena_2ddct_%d_%d.png' % (i, j))

        idct = idct2d(dct)
        idct_8x8[i*8:(i+1)*8, j*8:(j+1)*8] = idct

        # 1/4 coefs
        dct_4 = matrix_select(dct,2)
        Image.fromarray(dct_4.clip(0, 255).astype('uint8')).save('lena_2ddct_8x8_4/lena_2ddct_%d_%d.png' % (i, j))
        idct = idct2d(dct_4)
        idct_8x8_4[i*8:(i+1)*8, j*8:(j+1)*8] = idct

        # 1/16 coefs
        dct_16 = matrix_select(dct,4)
        Image.fromarray(dct_16.clip(0, 255).astype('uint8')).save('lena_2ddct_8x8_16/lena_2ddct_%d_%d.png' % (i, j))
        idct = idct2d(dct_16)
        idct_8x8_16[i*8:(i+1)*8, j*8:(j+1)*8] = idct

        # 1/64 coefs
        dct_64 = matrix_select(dct,8)
        Image.fromarray(dct_64.clip(0, 255).astype('uint8')).save('lena_2ddct_8x8_64/lena_2ddct_%d_%d.png' % (i, j))
        idct = idct2d(dct_64)
        idct_8x8_64[i*8:(i+1)*8, j*8:(j+1)*8] = idct


Image.fromarray(idct_8x8.clip(0, 255).astype('uint8')).save('lena_2ddct_8x8_2didct.png')
mse = np.mean((data - idct_8x8) ** 2)
psnr = 10 * np.log10(255.0 ** 2 / mse)
print('idct 8x8 img', idct_8x8)
print('psnr 2ddct 8x8:', psnr)

Image.fromarray(idct_8x8_4.clip(0, 255).astype('uint8')).save('lena_2ddct_8x8_4_2didct.png')
mse = np.mean((data - idct_8x8_4) ** 2)
psnr = 10 * np.log10(255.0 ** 2 / mse)
print('idct 8x8 1/4 img', idct_8x8_4)
print('psnr 2ddct 8x8 1/4:', psnr)

Image.fromarray(idct_8x8_16.clip(0, 255).astype('uint8')).save('lena_2ddct_8x8_16_2didct.png')
mse = np.mean((data - idct_8x8_16) ** 2)
psnr = 10 * np.log10(255.0 ** 2 / mse)
print('idct 8x8 1/16 img', idct_8x8_16)
print('psnr 2ddct 8x8 1/16:', psnr)

Image.fromarray(idct_8x8_64.clip(0, 255).astype('uint8')).save('lena_2ddct_8x8_64_2didct.png')
mse = np.mean((data - idct_8x8_64) ** 2)
psnr = 10 * np.log10(255.0 ** 2 / mse)
print('idct 8x8 1/64 img', idct_8x8_64)
print('psnr 2ddct 8x8 1/64:', psnr)
