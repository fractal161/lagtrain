'''
load entire movie, then allow seeking through frames
'''

import numpy as np
from numpy.lib.stride_tricks import as_strided
import datetime
import cv2

def seek_frame(n, cap, window_name):
    '''
    Display the given frame with the existing filter.
    '''
    cap.set(cv2.CAP_PROP_POS_FRAMES, n-1)
    ret, frame = cap.read()
    if not ret:
        raise ValueError('probably invalid frame number')
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Grayscale for speed
    actual_frame = cv2.resize(frame, (1536, 864))
    red_frame = actual_frame
    pixel_frame = pixel_filter(frame)
    pixel_frame = cv2.resize(pixel_frame, (1536, 864), interpolation=cv2.INTER_NEAREST)
    alpha=0.5
    blended_frame = cv2.addWeighted(red_frame, alpha, pixel_frame, 1-alpha, 0.0)
    # draw grid
    pixel_frame = cv2.cvtColor(pixel_frame, cv2.COLOR_GRAY2BGR)
    for i in range(1,18):
        pixel_frame[48*i,:] = (0,0,255)
    for i in range(1,32):
        pixel_frame[:,48*i] = (0,0,255)
    cv2.imshow(window_name, pixel_frame)

def pixel_filter(frame, black_cutoff=150, white_cutoff=200):
    '''
    Given 1920x1080 image, return 256x144 image with correct threshold
    '''
    result = np.empty((144, 256), dtype=np.uint8)
    def clamp_pixel(p):
        if p < black_cutoff:
            return 0
        if p > white_cutoff:
            return 255
        return 178
    for i in range(2):
        for j in range(2):
            view_shape = (72, 128, 7+i, 7+j)
            view_stride = tuple(frame.strides[i]*15
                    for i in range(2))
            frame_view = as_strided(frame[7*i:,7*j:], view_shape, 
                    view_stride + frame.strides)
            downscaled = np.mean(frame_view, axis=(2,3))
            result[i::2, j::2] = downscaled
    return np.vectorize(clamp_pixel)(result).astype(np.uint8)


def pixel_filter_old(frame, cutoff=160):
    '''
    Given 1920x1080 image, return 256x144 image with correct threshold
    '''
    def get_pixel(i, j, cutoff):
        i_low = 1080*i // 144
        i_high = 1080*(i+1) // 144
        j_low = 1920*j // 256
        j_high = 1920*(j+1) // 256
        pixel_region = frame[i_low:i_high,j_low:j_high]
        pixel_value = np.mean(pixel_region, axis=(0,1))
        if pixel_value <= cutoff:
            return 0
        if pixel_value >= 190:
            return 255
        return 178
    return np.array([[get_pixel(i, j, cutoff) for j in range(256)] for i in range(144)], dtype=np.uint8)

if __name__ == '__main__':
    lagtrain = cv2.VideoCapture('lagtrain.webm')
    if not lagtrain.isOpened():
        print('video not found')
        exit()
    
    n_frames = int(lagtrain.get(cv2.CAP_PROP_FRAME_COUNT))
    f_width = int(lagtrain.get(cv2.CAP_PROP_FRAME_WIDTH))
    f_height = int(lagtrain.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = lagtrain.get(cv2.CAP_PROP_FPS)

    window_name = 'test'
    seek_frame_bar = lambda n : seek_frame(n, lagtrain, window_name)
    seek_frame_bar(0)
    cv2.createTrackbar('frame', window_name, 0, n_frames-1, seek_frame_bar)

    print(f'{f_width}x{f_height}')
    print(f'{fps} fps')

    # hack to allow closing windows normally
    while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) > 0:
        cv2.waitKey(50)

    lagtrain.release()
    cv2.destroyAllWindows()
