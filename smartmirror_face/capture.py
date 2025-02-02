import cv2
import dlib
import os
import shutil
import time
import tempfile
from os.path import join
import glob
import multiprocessing

import openface
import openface.helper
from openface.data import iterImgs

from.config import opencv_haarcascade_frontalface, dlib_shape_predictor

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

OUTPUT_SIZE_WIDTH = 775
OUTPUT_SIZE_HEIGHT = 600
MARGIN = 10


def capture_faces(person, working_dir=None, limit=100, prune=False, processes=3, resolution=(640, 480),
                  size=96, video_device=0):

    face_cascade = cv2.CascadeClassifier(opencv_haarcascade_frontalface)
    face_db_path = join(working_dir, 'faces')
    face_person_path = join(face_db_path, person)

    tmp_path = tempfile.mkdtemp() + "/" + person
    os.mkdir(tmp_path)
    idx = 0

    if prune:
        logger.info("prune class...")
        prune_db(face_person_path)

    try:
        last = sorted(glob.glob(face_person_path + "/image-*"))[-1]
        idx = int(last[-8:-4]) + 1
    except IndexError:
        pass

    end_idx = idx + limit

    logger.debug("Set capture device")
    capture = video_device if not isinstance(video_device, int) else cv2.VideoCapture(video_device)
    cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("result-image", 400, 100)

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
                # if a face is tracked with the (faster) opencv detection
                # it is passed to the dlib tracker
                tracker.start_track(baseImage,
                                    dlib.rectangle(x-10, y-20, x+w+10, y+h+20))
                trackingFace = 1

        if trackingFace:
            logger.debug("Face found")

            trackingQuality = tracker.update(baseImage)

            if trackingQuality >= 8.75:
                tracked_position = tracker.get_position()

                t_x = int(tracked_position.left())
                t_y = int(tracked_position.top())
                t_w = int(tracked_position.width())
                t_h = int(tracked_position.height())

                # make sure the margin does not overflow the image
                y1 = max(t_y-MARGIN, 0)
                y2 = min(t_y+t_h+MARGIN, resolution[1])
                x1 = max(t_x-MARGIN, 0)
                x2 = min(t_x+t_w+MARGIN, resolution[0])
                cropped = baseImage[y1:y2, x1:x2]

                # save cropped result to temp folder
                cv2.imwrite("{0}/image-{1:04d}.png".format(tmp_path, idx), cropped)
                # add a rectangle for easier evaluation if the right face has been tracked
                cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w , t_y + t_h), rectangleColor, 2)
                idx += 1
                logger.info("Stored face image with id {0}".format(idx))
            else:
                trackingFace = 0

        # show grabbed image plus rectangle (if a face had been tracked)
        largeResult = cv2.resize(resultImage, (OUTPUT_SIZE_WIDTH, OUTPUT_SIZE_HEIGHT))
        cv2.imshow("result-image", largeResult)

    logger.info("Finished capturing")
    cv2.destroyAllWindows()
    # in case we have created the device on our own, we can delete it now since it is not needed anymore
    if isinstance(video_device, int):
        capture.release()

    # let opencv time to destroy the windows
    cv2.waitKey(2)
    logger.info("Start alignment")
    align_images(tmp_path, face_db_path, processes=processes, size=size)


class AlignWorker(multiprocessing.Process):

    def __init__(self, input, output, size):
        super(AlignWorker, self).__init__()
        self.done = False
        self.input = input
        self.output = output
        self.align = openface.AlignDlib(dlib_shape_predictor)
        self.size = size
        self.processed = 0

    def run(self):
        try:
            while not self.input.empty():
                imgName, rgb = self.input.get(block=True)
                outRgb = self.align.align(self.size, rgb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE,
                                          skipMulti=True)
                self.output.put((imgName, outRgb))
                self.processed += 1

        except Exception as e:
            print(e)

        # signal that we are done
        self.output.put((None, None))


def align_images(input_path, output_path, processes=3, size=96):
    # not available in all opencv 2.4.x versions
    try:
        # since 2.4.13 opencv supports multithreading out of the box
        # this does not play well with multiprocessing of python
        # we need multiprocessing to spread the load of image alignment to all the cores
        # as python threading is limited to one core only
        cv2.setNumThreads(0)
    except AttributeError:
        pass

    openface.helper.mkdirP(output_path)
    imgs = list(iterImgs(input_path))
    m = multiprocessing.Manager()
    workers = m.Queue()
    results = m.Queue()

    # fill the worker queue before starting the processes as the workers
    # will exit in case the queue is empty
    logger.info('preparing images')
    for imgObject in imgs:
        outDir = os.path.join(output_path, imgObject.cls)
        openface.helper.mkdirP(outDir)
        outputPrefix = os.path.join(outDir, imgObject.name)
        imgName = outputPrefix + ".png"
        rgb = imgObject.getRGB()
        workers.put((imgName, rgb))

    threads = []
    logger.info('running workers')
    for i in range(processes):
        t = AlignWorker(workers, results, size)
        t.start()
        threads.append(t)

    workers_done = 0
    while not workers_done == processes:
        logger.info("{0} images left...".format(workers.qsize()))
        while not results.empty():
            imgName, outRgb = results.get(block=True)
            # the last message a worker will send is a 'None' image to signal it is done
            # as the worker runs in another process you cannot access ressources/properties
            # of the worker directly
            if imgName is None:
                workers_done += 1
            elif outRgb is not None:
                # dlib works with RGB but opencv requires BGR order for writing images
                outBgr = cv2.cvtColor(outRgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(imgName, outBgr)
        time.sleep(0.5)


def prune_db(path, threshold=None):
    exts = ["jpg", "png"]
    # python does not offer simple ways to delete folders with content (this is probably a good thing)
    # os walk cleans a folder (just PNG and JPGs) and removes it afterwards
    for subdir, dirs, files in os.walk(path):
        nImgs = 0
        for fName in files:
            (imageClass, imageName) = (os.path.basename(subdir), fName)
            if any(imageName.lower().endswith("." + ext) for ext in exts):
                nImgs += 1
        if threshold is None or nImgs < threshold:
            logger.info("Removing {}".format(subdir))
            shutil.rmtree(subdir)
