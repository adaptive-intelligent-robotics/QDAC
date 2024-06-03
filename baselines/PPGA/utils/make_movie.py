
import os
import cv2
import argparse
from attrdict import AttrDict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str)
    parser.add_argument('--video_name', type=str)
    parser.add_argument('--fps', type=int)
    args = parser.parse_args()
    return AttrDict(vars(args))


if __name__ == '__main__':
    cfg = parse_args()
    images = sorted([img for img in os.listdir(cfg.image_folder) if img.endswith(".png")])
    frame = cv2.imread(os.path.join(cfg.image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(cfg.video_name, fourcc, cfg.fps, (width, height))

    for image in images:
        print(image)
        video.write(cv2.imread(os.path.join(cfg.image_folder, image)))

    cv2.destroyAllWindows()
    video.release()