from os.path import dirname, join

file_dir = dirname(__file__)
model_dir = join(file_dir, 'models')
lua_dir = join(file_dir, 'lua')

dlib_shape_predictor = join(model_dir, "shape_predictor_68_face_landmarks.dat")
openface_network_model = join(model_dir, "nn4.small2.v1.t7")
opencv_haarcascade_frontalface = join(model_dir, 'haarcascade_frontalface_default.xml')
rsb_face_scope = "/io/display/mirror/face"
rsb_face_enabled = "/io/display/mirror/faceMode"

unknown_person_label = "_unknown"

model_abort = 'abort'
model_detect = 'detect'
model_paused = 'paused'
