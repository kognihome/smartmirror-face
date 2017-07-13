import argparse

from .capture import capture_faces, prune_db
from .detect import detect, train
from .model import Model
from .config import model_abort, model_detect

parser = argparse.ArgumentParser(description='Process some integers.')
subparsers = parser.add_subparsers(title='action', dest='action')
parser.add_argument('-w', '--workdir', type=str, help='database and feature directory [required]')
parser.add_argument('-p', '--prune', action='store_true', help='prune person dataset before capturing')
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


capture_parser = subparsers.add_parser('capture')
train_parser = subparsers.add_parser('train')
detect_parser = subparsers.add_parser('detect')

capture_parser.add_argument('name', type=str, help='person name', default="person")

args = parser.parse_args()
args.video = tuple(args.video)

if not args.workdir:
    print("Working directory is required!")
    parser.print_help()
    exit(1)

if args.action == 'capture':
    capture_faces(args.name, args.workdir, prune=args.prune, processes=args.threads, limit=args.limit,
                  resolution=args.video, size=args.size, video_device=args.device)

if args.action == 'train':
    train(args.workdir + "/faces", args.workdir + "/features", cuda=args.cuda)

if args.action == 'detect':
    # modes: 'paused', 'detect', 'exit', '<person_name>[:clean]'
    model = Model()
    while model.mode != model_abort:
        if model.mode == model_detect:
            detect(model, args.workdir + "/features/classifier.pkl", cuda=args.cuda, img_dim=args.size,
                   video_device=args.device, resolution=args.video, threshold=args.confidence)
        else:
            person = model.mode.split(':')
            prune = len(person) > 1 and person[1] == 'clean'
            capture_faces(person[0], args.workdir, prune=prune, processes=args.threads, limit=args.limit,
                          resolution=args.video, size=args.size, video_device=args.device)
            train(args.workdir + "/faces", args.workdir + "/features", cuda=args.cuda)
            model.mode = model_detect


