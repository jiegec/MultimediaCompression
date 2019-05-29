from PIL import Image
img = Image.open('lena.bmp').convert('L')
img.save('lena_grayscale.png')

img = Image.open('custom.png').convert('L')
img.save('custom_grayscale.png')
