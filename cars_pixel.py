from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import fftpack
import sys

def dct2d(data):
    return fftpack.dct(fftpack.dct(data, norm='ortho').T, norm='ortho').T

def select(data):
    return data.diagonal()

def select2(data):
    return np.concatenate((data.diagonal(), data.T.diagonal().T))

def dct2d_select(data):
    return dct2d(data)[0:8, 0:8]

def dct2d_select2(data):
    return dct2d(data)[0:2, 0:2]

def corner(data):
    gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    return cv2.cornerHarris(gray, 2, 3, 0.04)

def draw_rect(data, x, y):
    data = np.copy(data)
    data[x:(x+1),y:(y+17)] = 0
    data[(x+16):(x+17),y:(y+17)] = 0
    data[x:(x+17),y:(y+1)] = 0
    data[x:(x+17),(y+16):(y+17)] = 0
    return data


# white car in the middle
if sys.argv[1] == 'car':
    x = 166
    y = 181
    begin = 1
    end = 50

# bus in the right
else:
    x = 125
    y = 333
    begin = 10
    end = 170

img = Image.open('cars/frame%d.png' % begin)
data = np.array(img, dtype=np.uint8)

Image.fromarray(draw_rect(data, x, y)).save('car_detect/rect_0.png')

output_images = []

plt.xlim(0, data.shape[1])
plt.ylim(0, data.shape[0])
plt.gca().invert_yaxis()

def identity(data):
    return data

#transformer = identity
#transformer = dct2d
#transformer = select
#transformer = select2
#transformer = dct2d_select
#transformer = dct2d_select2
transformer = corner

target_block = transformer(data[x:(x+16), y:(y+16)])
orig_target_block = target_block

radius = 32
mses = []
for i in range(begin, end):
    print('finding %d image' % i)
    # find target block in current image
    new_img = Image.open('cars/frame%d.png' % i)
    new_data = np.array(new_img, dtype=np.uint8)
    h, w, c = new_data.shape
    min_mse = 9999999
    min_mse_hh = 0
    min_mse_ww = 0
    for hh in range(0, h-16):
        for ww in range(0, w-16):
            if hh >= x - radius and hh <= x + radius and ww >= y - radius and ww <= y + radius:
                subimage = transformer(new_data[hh:(hh+16),ww:(ww+16)])
                #mse = np.mean((subimage - target_block)**2) + np.mean((subimage - orig_target_block) ** 2)
                mse = np.mean((subimage - target_block)**2)
                if mse < min_mse:
                    min_mse = mse
                    min_mse_hh = hh
                    min_mse_ww = ww

    print('mse', min_mse, 'x', x, 'y', y)
    #plt.arrow(x, y, min_mse_hh - x, min_mse_ww - y)
    plt.annotate("", xytext=(y, x), xy=(min_mse_ww, min_mse_hh), arrowprops=dict(arrowstyle="->"))
    output = Image.fromarray(draw_rect(new_data, min_mse_hh, min_mse_ww))
    output.save('car_detect/rect_%d.png' % i)
    output_images.append(output)

    target_block = transformer(new_data[min_mse_hh:(min_mse_hh+16),min_mse_ww:(min_mse_ww+16)])
    x = min_mse_hh
    y = min_mse_ww
    mses.append(min_mse)

output_images[0].save('car_detect/track_%s_%s.gif' % (sys.argv[1], transformer.__name__), save_all=True, append_images=output_images[1:], duration=100, loop=0)
plt.savefig('mv_%s_%s.png' % (sys.argv[1], transformer.__name__))

plt.cla()
plt.plot(range(len(mses)), mses)
plt.savefig('mse_%s_%s.png' % (sys.argv[1], transformer.__name__))
