import cv2
import dlib
import os
import shutil
import time
import tempfile
from os.path import join
import glob
import threading
from Queue import Queue

import openface
import openface.helper
from openface.data import iterImgs

from.config import opencv_haarcascade_frontalface, dlib_shape_predictor

OUTPUT_SIZE_WIDTH = 775
OUTPUT_SIZE_HEIGHT = 600


def capture_faces(person, working_dir=None, limit=100, prune=False, processes=3, resolution=(320, 240), size=96,
                  video_device=0):

    face_cascade = cv2.CascadeClassifier(opencv_haarcascade_frontalface)
    face_db_path = join(working_dir, 'faces')
    face_person_path = join(face_db_path, person)

    tmp_path = tempfile.mkdtemp() + "/" + person
    os.mkdir(tmp_path)
    idx = 0

    if prune:
        prune_db(face_person_path)

    try:
        last = sorted(glob.glob(face_person_path + "/image-*"))[-1]
        idx = int(last[-8:-4]) + 1
    except IndexError:
        pass

    end_idx = idx + limit

    capture = cv2.VideoCapture(video_device)
    cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("result-image", 400, 100)

    cv2.startWindowThread()
    tracker = dlib.correlation_tracker()

    trackingFace = 0
    rectangleColor = (0, 165, 255)

    while idx <= end_idx:
        rc, fullSizeBaseImage = capture.read()
        baseImage = cv2.resize(fullSizeBaseImage, resolution)

        # give opencv time to draw the window
        cv2.waitKey(2)

        resultImage = baseImage.copy()

        if not trackingFace:

            gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            maxArea = 0
            x = 0
            y = 0
            w = 0
            h = 0

            for (_x,_y,_w,_h) in faces:
                if _w * _h > maxArea:
                    x = int(_x)
                    y = int(_y)
                    w = int(_w)
                    h = int(_h)
                    maxArea = w*h

            if maxArea > 0:
                tracker.start_track(baseImage,
                                    dlib.rectangle(x-10, y-20, x+w+10, y+h+20))
                trackingFace = 1

        if trackingFace:

            trackingQuality = tracker.update( baseImage )

            if trackingQuality >= 8.75:
                tracked_position = tracker.get_position()
                cv2.imwrite("{0}/image-{1:04d}.png".format(tmp_path, idx), fullSizeBaseImage)
                idx += 1

                t_x = int(tracked_position.left())
                t_y = int(tracked_position.top())
                t_w = int(tracked_position.width())
                t_h = int(tracked_position.height())
                cv2.rectangle(resultImage, (t_x, t_y),
                                            (t_x + t_w , t_y + t_h),
                                            rectangleColor ,2)

            else:
                trackingFace = 0

        largeResult = cv2.resize(resultImage, (OUTPUT_SIZE_WIDTH, OUTPUT_SIZE_HEIGHT))
        cv2.imshow("result-image", largeResult)

    cv2.destroyAllWindows()
    del capture
    # let opencv time to destroy the windows
    cv2.waitKey(2)
    align_images(tmp_path, face_db_path, processes=processes, size=size)


class AlignWorker(threading.Thread):

    def __init__(self, queue):
        super(AlignWorker, self).__init__()
        self.running = False
        self.queue = queue
        self.align = openface.AlignDlib(dlib_shape_predictor)

    def run(self):

        while not self.queue.empty():
            imgObject, output_path, size = self.queue.get(block=True)
            outDir = os.path.join(output_path, imgObject.cls)
            openface.helper.mkdirP(outDir)
            outputPrefix = os.path.join(outDir, imgObject.name)
            imgName = outputPrefix + ".png"

            if os.path.isfile(imgName):
                pass
            else:
                rgb = imgObject.getRGB()
                if rgb is None:
                    outRgb = None
                else:
                    outRgb = self.align.align(size, rgb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE,
                                              skipMulti=True)
                if outRgb is not None:
                    outBgr = cv2.cvtColor(outRgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(imgName, outBgr)
        self.running = False


def align_images(input_path, output_path, processes=3, size=96):
    openface.helper.mkdirP(output_path)
    imgs = list(iterImgs(input_path))
    queue = Queue()

    for imgObject in imgs:
        queue.put((imgObject, output_path, size))

    threads = []
    for i in range(processes):
        t = AlignWorker(queue)
        t.start()
        threads.append(t)

    while any(t.running for t in threads):
        print("{0} images left...".format(queue.qsize()))
        time.sleep(0.5)


def prune_db(path, threshold=None):
    exts = ["jpg", "png"]
    for subdir, dirs, files in os.walk(path):
        if subdir == path:
            continue
        nImgs = 0
        for fName in files:
            (imageClass, imageName) = (os.path.basename(subdir), fName)
            if any(imageName.lower().endswith("." + ext) for ext in exts):
                nImgs += 1
        if threshold is None or nImgs < threshold:
            print("Removing {}".format(subdir))
            shutil.rmtree(subdir)
