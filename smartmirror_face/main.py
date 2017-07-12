import argparse

from .capture import capture_faces, prune_db
from .detect import detect, train
from .model import Model
from .config import model_abort, model_detect

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('mode', type=str, help='face detection mode')
parser.add_argument('workdir', type=str, help='directory for database and model')
parser.add_argument('name', type=str, help='label for training', nargs='?', default="person")

args = parser.parse_args()

if args.mode == 'capture':
    capture_faces(args.name, args.workdir)

if args.mode == 'train':
    train(args.workdir + "/faces", args.workdir + "/features")

if args.mode == 'detect':
    # modes: 'paused', 'detect', 'exit', '<person_name>[:clean]'
    model = Model()
    while model.mode != model_abort:
        if model.mode == model_detect:
            detect(model, args.workdir + "/features/classifier.pkl")
        else:
            person = model.mode.split(':')
            if 'clean' == person[1]:
                prune_db(args.workdir + "/faces/" + person[0])
            capture_faces(person[0], args.workdir)
            train(args.workdir + "/faces", args.workdir + "/features")
            model.mode = model_detect


