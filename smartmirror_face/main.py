import argparse

from .capture import capture_faces
from .detect import detect, train

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
    detect(args.workdir + "/features/classifier.pkl")
