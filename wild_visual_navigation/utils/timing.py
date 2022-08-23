import torch


class Timer:
    def __init__(self, name="") -> None:
        self.name = name

    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.end.record()
        torch.cuda.synchronize()
        print(f"Time {self.name}: ", self.start.elapsed_time(self.end), "ms")


def accumulate_time(method):
    def timed(*args, **kw):
        if hasattr(args[0], "not_time"):
            if args[0].not_time:
                return method(*args, **kw)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = method(*args, **kw)
        end.record()
        torch.cuda.synchronize()

        st = start.elapsed_time(end)
        if hasattr(args[0], "time_summary"):
            summary = args[0].time_summary
        else:
            args[0].time_summary = {}

        if method.__name__ in args[0].time_summary:
            args[0].time_summary[method.__name__] += st
        else:
            args[0].time_summary[method.__name__] = st

        return result

    return timed


if __name__ == "__main__":

    print("Start timing using context manager")

    with Timer("Test1"):
        s1, s2 = 1000, 1000
        a = torch.zeros((s1, s2))
        for x in range(s1):
            for y in range(s2):
                a[x, y] = x * y
    with Timer("Test2"):
        s1, s2 = 1000, 1000
        a = torch.zeros((s1, s2))
        for x in range(s1):
            for y in range(s2):
                a[y, x] = x * y
