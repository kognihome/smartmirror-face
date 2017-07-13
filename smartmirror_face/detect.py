import pickle
import sys
from os import remove
from os.path import join, split, dirname, exists
import subprocess
from operator import itemgetter
from time import sleep

import numpy as np
import pandas as pd

import cv2
from sklearn.mixture import GMM
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import openface

from .config import dlib_shape_predictor, openface_network_model, lua_dir, model_abort
from .config import model_detect, model_paused, unknown_person_label
from .smoothing import Smoother

np.set_printoptions(precision=2)


class Detector(object):

    def __init__(self, model_path, align, net, img_dim=96):
        self.align = align
        self.net = net
        self.img_dim = img_dim

        with open(model_path, 'r') as f:
            if sys.version_info[0] < 3:
                self.le, self.clf = pickle.load(f)  # le - label and clf - classifer
            else:
                self.le, self.clf = pickle.load(f, encoding='latin1')  # le - label and clf - classifer

    def infer(self, img):
        reps = self._get_rep(img)
        persons = []
        confidences = []
        for rep in reps:
            try:
                rep = rep.reshape(1, -1)
            except AttributeError:
                print ("No Face detected")
                return None, None
            predictions = self.clf.predict_proba(rep).ravel()
            maxI = np.argmax(predictions)
            persons.append(self.le.inverse_transform(maxI))
            confidences.append(predictions[maxI])
            if isinstance(self.clf, GMM):
                dist = np.linalg.norm(rep - self.clf.means_[maxI])
                print("  + Distance from the mean: {}".format(dist))
                pass
        return persons, confidences

    def _get_rep(self, bgrImg):
        if bgrImg is None:
            raise Exception("Unable to load image/frame")

        rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
        bb = self.align.getAllFaceBoundingBoxes(rgbImg)

        if bb is None:
            # raise Exception("Unable to find a face: {}".format(imgPath))
            return None

        alignedFaces = []
        for box in bb:
            alignedFaces.append(
                self.align.align(self.img_dim, rgbImg, box,
                                 landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE))

        reps = []
        for alignedFace in alignedFaces:
            reps.append(self.net.forward(alignedFace))

        return reps


def detect(model, model_path, video_device=0, resolution=None, roi=None, cuda=False, img_dim=96, threshold=0.5):
    align = openface.AlignDlib(dlib_shape_predictor)
    net = openface.TorchNeuralNet(openface_network_model, imgDim=img_dim, cuda=cuda)

    detector = Detector(model_path, align, net, img_dim)
    smoother = Smoother(model)

    # Capture device. Usually 0 will be webcam and 1 will be usb cam.
    video_capture = cv2.VideoCapture(video_device)
    if resolution is not None:
        video_capture.set(3, resolution[0])
        video_capture.set(4, resolution[1])

    if roi is not None:
        vw, vh = video_capture.get(3), video_capture.get(4)
        x, y = int(roi[0] * vw), int(roi[1] * vh)
        roi_params = (x, y, x + int(roi[3] * vw), y + int(roi[4] * vh))
    else:
        roi_params = None

    try:
        while model.mode in [model_detect, model_paused]:
            if model.mode == model_detect:
                ret, frame = video_capture.read()
                if roi_params is not None:
                    roi = frame[roi_params[1]:roi_params[3],
                                roi_params[0]:roi_params[2]]
                else:
                    roi = frame

                persons, confidences = detector.infer(roi)
                for i, c in enumerate(confidences):
                    if c <= threshold:  # 0.5 is kept as threshold for known face.
                        persons[i] = unknown_person_label

                smoother.detect(persons)
                conf_display = ["{0:.2f}".format(c) for c in confidences]
                cv2.putText(frame, "P: {0} C: {1} => {2}:{3}".format(persons, conf_display,
                                                                 smoother.current_value, model.current),
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                if roi_params is not None:
                    cv2.rectangle(frame, (roi_params[0], roi_params[1]), (roi_params[2], roi_params[3]), (0, 165, 255))
                cv2.imshow('', frame)

                # update winodws
                cv2.waitKey(1)
            elif model.mode == model_paused:
                sleep(0.3)
    except KeyboardInterrupt:
        model.mode = model_abort
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


def train(input_path, output_path, cuda=False):
    if exists(input_path + "/cache.t7"):
        remove(input_path + "/cache.t7")
    main_lua = join(lua_dir, 'main.lua')
    call = [main_lua, '-data', input_path, '-outDir', output_path, '-model', openface_network_model]
    if cuda:
        call.append('--cuda')
    subprocess.check_call(call)

    print("Loading embeddings.")
    fname = "{}/labels.csv".format(output_path)
    labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
    labels = map(itemgetter(1), map(split, map(dirname, labels)))  # Get the directory.
    fname = "{}/reps.csv".format(output_path)
    embeddings = pd.read_csv(fname, header=None).as_matrix()
    le = LabelEncoder().fit(labels)
    labelsNum = le.transform(labels)
    nClasses = len(le.classes_)
    print("Training for {} classes.".format(nClasses))

    clf = SVC(C=1, kernel='linear', probability=True)
    clf.fit(embeddings, labelsNum)

    fName = "{}/classifier.pkl".format(output_path)
    print("Saving classifier to '{}'".format(fName))
    with open(fName, 'w') as f:
        pickle.dump((le, clf), f)
