import cv2
vidcap = cv2.VideoCapture('cars.avi')
success,image = vidcap.read()
count = 0
while success and count <= 200:
  #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  cv2.imwrite("cars/frame%d.png" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
