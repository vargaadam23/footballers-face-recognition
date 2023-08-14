# import the time module
import time
from cv2 import VideoCapture, imshow, imwrite, waitKey, destroyWindow
import cv2
import numpy as np

COUNTODOWN_TIME = 3

def resizeAndPad(img, size, padColor=255):   
    h, w = img.shape[:2]
    sh, sw = size
    
    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    
    else: # stretching image
        interp = cv2.INTER_CUBIC
    
    # aspect ratio of image
    aspect = float(w)/h 
    saspect = float(sw)/sh
    
    if (saspect >= aspect) or ((saspect == 1) and (aspect <= 1)):  # new horizontal image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = float(sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    
    elif (saspect < aspect) or ((saspect == 1) and (aspect >= 1)):  # new vertical image
        new_w = sw
        new_h = np.round(float(new_w) / aspect).astype(int)
        pad_vert = float(sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    
    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3
    
    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)
    
    return scaled_img

def countdown(t, executable):
    print('Picture taken in...')

    while t:
        mins, secs = divmod(t, 60)
        time.sleep(1)
        print(t)
        t -= 1
        
    return executable()

def menu():
    print('Select what you want to do:')
    print('1. Take webcam image.')
    print('2. Compare with external image')
    print('0. Exit application')

    return input('Select next action: ')

def takePicture(imageName):
    cam = VideoCapture(0)

    x = lambda: cam.read()

    result, image = countdown(COUNTODOWN_TIME, x)
    newimage = resizeAndPad(cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX), (400,400), 127)
    
    if result and image.any() and newimage.any():
        imshow(imageName, newimage)
        
        imwrite(imageName+".png", newimage)

        print("Click 0 to proceed")

        waitKey(0)
        destroyWindow(imageName)
        cam.release()

        return newimage
    
    cam.release()
    return False


