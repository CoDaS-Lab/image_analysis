import time
from functools import wraps


def timeit(thresh=None, classname=''):
    def decorator(method):
        @wraps(method)
        def timed(*args, **kw):
            start = time.time()
            result = method(*args, **kw)
            elapsed = time.time() - start

            print('\n {0}: {1}() timing: {2:.5f} sec'
                  .format(classname, method.__name__, elapsed))

            if thresh is not None and elapsed > thresh:
                print('{0}() ran slower than thresh={1}'
                      .format(method.__name__, thresh))
            return result
        return timed
    return decorator
