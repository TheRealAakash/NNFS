import cv2
import numpy as np
import subprocess as sp
import shlex


def getVideoWriter(width, height, fps, output_path):
    process = sp.Popen(shlex.split(
        f'ffmpeg -y -s {width}x{height} -pixel_format bgr24 -f rawvideo -r {fps} -i pipe: -vcodec libx264 -pix_fmt yuv420p -crf 24 {output_path}'),
        stdin=sp.PIPE)
    return process


class VideoWriter:
    def __init__(self, width, height, fps, output_path):
        self.process = getVideoWriter(width, height, fps, output_path)

    def write(self, frame):
        self.process.stdin.write(frame.tobytes())

    def release(self):
        self.process.stdin.close()
        self.process.wait()
        self.process.terminate()
