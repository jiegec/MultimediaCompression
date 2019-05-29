from PIL import Image
img = Image.open('lena.bmp').convert('L')
img.save('lena_grayscale.png')
