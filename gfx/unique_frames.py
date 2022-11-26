'''
identifies all unique frames
'''

import numpy as np
import cv2

def get_nth_frame(cap, n):
    old_n = cap.get(cv2.CAP_PROP_POS_FRAMES)
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ret, frame = cap.read()
    if not ret:
        raise ValueError('invalid frame number')
    cap.set(cv2.CAP_PROP_POS_FRAMES, old_n)
    return frame

def phash_to_int(phash):
    res = np.uint64(0)
    for i in phash[0]:
        res = res*np.uint64(256) + np.uint64(i)
    return res

if __name__ == '__main__':
    lagtrain = cv2.VideoCapture('lagtrain.webm')
    n_frames = int(lagtrain.get(cv2.CAP_PROP_FRAME_COUNT))
    f_width = int(lagtrain.get(cv2.CAP_PROP_FRAME_WIDTH))
    f_height = int(lagtrain.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = lagtrain.get(cv2.CAP_PROP_FPS)
    print(f'{f_width}x{f_height}')
    print(f'{n_frames} frames ({fps} fps)')
    ret, frame = lagtrain.read()
    # while ret:
    hashes = []
    for _ in range(200):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # to try and minimize the impact of the white text, we take
        # 3 separate hashes
        bands = [frame[:380,:], frame[475:780,:], frame[830:,:]]
        frame_hash = cv2.img_hash.radialVarianceHash(frame)
        print(frame_hash.shape)
        subs = frame[780:830,:]
        frame_pos = int(lagtrain.get(cv2.CAP_PROP_POS_FRAMES))
        print(f'FRAME# {frame_pos:4d}: {frame_hash}')

        ret, frame = lagtrain.read()
