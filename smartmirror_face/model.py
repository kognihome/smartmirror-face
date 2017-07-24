import rsb

from .config import rsb_face_scope, rsb_face_mode


class Model(object):

    def __init__(self):
        self._current = None
        self._mode = 'detect'
        self.rsb_informer = rsb.createInformer(rsb_face_scope)
        self.rsb_listener = rsb.createListener(rsb_face_mode)
        self.rsb_listener.addHandler(self.on_mode_change)

    def __del__(self):
        del self.rsb_listener
        del self.rsb_informer

    def on_mode_change(self, evt):
        # in case someone sends empty mode strings
        if len(evt.data) > 0:
            self._mode = evt.data

    @property
    def current(self):
        return self._current

    @current.setter
    def current(self, value):
        if value != self._current:
            # smartmirror exptects a string which can be checked as a personal token
            # in kognidb, we currently use icl.face.<person_name> to assign face detection
            # tokens to people
            self.rsb_informer.publishData('icl.face.' + value if value is not None else '')
            self._current = value

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value
        with rsb.createInformer(rsb_face_mode) as informer:
            informer.publishData(value)
