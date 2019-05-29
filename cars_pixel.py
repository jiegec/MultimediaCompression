from PIL import Image
import numpy as np
import cv2

def draw_rect(data, x, y):
    data = np.copy(data)
    data[x:(x+1),y:(y+17)] = 0
    data[(x+16):(x+17),y:(y+17)] = 0
    data[x:(x+17),y:(y+1)] = 0
    data[x:(x+17),(y+16):(y+17)] = 0
    return data

# white car in the middle
#x = 166
#y = 181
#begin = 1
#end = 50

# bus in the right
x = 125
y = 333
begin = 10
end = 50

img = Image.open('cars/frame%d.png' % begin)
data = np.array(img, dtype=np.uint8)
target_block = data[x:(x+16), y:(y+16)]
orig_target_block = target_block

Image.fromarray(draw_rect(data, x, y)).save('car_detect/rect_0.png')
Image.fromarray(target_block).save('car_detect/target_0.png')

output_images = []

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
            subimage = new_data[hh:(hh+16),ww:(ww+16)]
            mse = np.mean((subimage - target_block)**2) + np.mean((subimage - orig_target_block) ** 2)
            if mse < min_mse:
                min_mse = mse
                min_mse_hh = hh
                min_mse_ww = ww

    print('mse', min_mse)
    output = Image.fromarray(draw_rect(new_data, min_mse_hh, min_mse_ww))
    output.save('car_detect/rect_%d.png' % i)
    output_images.append(output)
    target_block = new_data[min_mse_hh:(min_mse_hh+16),min_mse_ww:(min_mse_ww+16)]

output_images[0].save('car_detect/track.gif', save_all=True, append_images=output_images[1:], duration=100, loop=0)

