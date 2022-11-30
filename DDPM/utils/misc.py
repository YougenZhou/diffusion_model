import time


class Timer(object):
    def __init__(self):
        self._pass_time = 0
        self._start_time = None
        return

    def start(self):
        self._start_time = time.time()

    def pause(self):
        self._pass_time += time.time() - self._start_time
        self._start_time = None

    def reset(self):
        self._pass_time = 0

    @property
    def pass_time(self):
        if self._start_time is None:
            return self._pass_time
        else:
            return self._pass_time + time.time() - self._start_time
