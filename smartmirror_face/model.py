import rsb

from .config import rsb_face_scope, rsb_face_enabled


class Model(object):

    def __init__(self):
        self._current = None
        self._update = True
        self.rsb_informer = rsb.createInformer(rsb_face_scope)
        self.rsb_listener = rsb.createListener(rsb_face_enabled)
        self.rsb_listener.addHandler(self.on_enabled)

    def __del__(self):
        del self.rsb_listener
        del self.rsb_informer

    def on_enabled(self, evt):
        self._update = evt.data

    @property
    def current(self):
        return self._current

    @current.setter
    def current(self, value):
        if value != self._current:
            self.rsb_informer.publishData(value if value is not None else '')
            self._current = value

    @property
    def update(self):
        return self._update
