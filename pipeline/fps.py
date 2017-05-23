from time import time


class FPS:

    def __init__(self):
        self._start = 0
        self._end = 0
        self._nframes = 0

    def start(self):
        self._start = time()

    def update(self):
        if self.elapsed() >= 1:
            print('fps {}'.format(self.fps()))
            self._nframes = 0
            self._start = self._end
        else:
            self._nframes += 1

    def elapsed(self):
        self._end = time()
        return (self._end - self._start)

    def fps(self):
        return self._nframes / self.elapsed()


