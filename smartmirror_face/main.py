import argparse
from cv2 import VideoCapture

from .capture import capture_faces
from .detect import detect, train
from .model import Model
from .config import model_abort, model_detect

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def start():
    logging.basicConfig(level=logging.INFO)
    # # in case you need detailed output
    # logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('action', choices=['capture', 'train', 'detect'])
    parser.add_argument('-w', '--workdir', type=str, help='database and feature directory [required]')
    parser.add_argument('-p', '--prune', action='store_true', help='prune person dataset before capturing')
    parser.add_argument('-n', '--name', type=str, help='name/label for captured data')
    parser.add_argument('-v', '--video', nargs=2, type=int, default=[320, 240],
                        help='face detection image resolution (w h)')
    parser.add_argument('-r', '--roi', nargs=4, type=float,
                        help="region of interest for face detection (x y w h) in relative(!) coordinates")
    parser.add_argument('-c', '--cuda', action='store_true', help="use cuda")
    parser.add_argument('-t', '--threads', type=int, default=3, help="number of image alignment parallel threads")
    parser.add_argument('-l', '--limit', type=int, default=100, help="amount of images to capture")
    parser.add_argument('-s', '--size', type=int, default=96, help="image width of database faces")
    parser.add_argument('-d', '--device', type=int, default=0, help="opencv video device id")
    parser.add_argument('-m', '--min-confidence', type=float, default=0.5, dest='confidence',
                        help="minimal detection confidence")

    args = parser.parse_args()
    # requires since argparse passes lists but opencv REQUIRES tuples
    args.video = tuple(args.video)

    if not args.workdir:
        print("Working directory is required!")
        parser.print_help()
        exit(1)

    # the detection operates in three modes, 'capture' and 'train' are meant for bootstrapping and will
    # return after their job is done. Capturing the images for the database and training the model are
    # separated since capturing should be done for different persons before training is conducted
    if args.action == 'capture':
        logger.info("capture new images")
        capture_faces(args.name, args.workdir, prune=args.prune, processes=args.threads, limit=args.limit,
                      resolution=args.video, size=args.size, video_device=args.device)

    elif args.action == 'train':
        logger.info("train data")
        train(args.workdir + "/faces", args.workdir + "/features", cuda=args.cuda)

    # 'detect' is more or less the operation mode which requires a trained model
    # (which is shipped with the project though). Capturing and training can be triggered by RSB messages
    # containing the name of the person to track and an optional suffix :clean in case the previous
    # training data _FOR THAT CLASS_ should be pruned.
    elif args.action == 'detect':
        logger.info("grabbing video device")
        capture = VideoCapture(args.device)
        # modes: 'paused', 'detect', 'exit', '<person_name>[:clean]'
        model = Model()
        while model.mode != model_abort:
            logger.info("entering detection loop")
            if model.mode == model_detect:
                detect(model, args.workdir + "/features/classifier.pkl", cuda=args.cuda, img_dim=args.size,
                       video_device=capture, resolution=args.video, threshold=args.confidence)
            else:
                logging.info("train model")
                person = model.mode.split(':')
                model.mode = 'capture'
                prune = len(person) > 1 and person[1] == 'clean'
                capture_faces(person[0], args.workdir, prune=prune, processes=args.threads, limit=args.limit,
                              resolution=args.video, size=args.size, video_device=capture)
                model.mode = 'train'
                train(args.workdir + "/faces", args.workdir + "/features", cuda=args.cuda)
                model.mode = model_detect
                logger.info("loop exited")


if __name__ == "__main__":
    start()
