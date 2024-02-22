import time

def timeit(method):
    def timed(*args, **kw):
        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()

        print(f"{method.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return timed