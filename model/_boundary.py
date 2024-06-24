## decoration of periodic boundary condition
def _periodically_continued(a, b):
    interval = b - a
    return lambda f: lambda x: f((x - a) % interval + a)