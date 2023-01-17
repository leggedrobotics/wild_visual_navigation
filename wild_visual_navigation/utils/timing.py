import torch
import time


class CpuTimer:
    def __init__(self, name="") -> None:
        self.name = name

    def __enter__(self):
        self.tic()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(f"Time {self.name}: ", self.toc(), "ms")

    def tic(self):
        self.start = time.perf_counter()

    def toc(self):
        self.end_time = time.perf_counter()
        return self.end - self.start


class Timer:
    def __init__(self, name="") -> None:
        self.name = name
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.tic()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(f"Time {self.name}: ", self.toc(), "ms")

    def tic(self):
        self.start.record()

    def toc(self):
        self.end.record()
        torch.cuda.synchronize()
        return self.start.elapsed_time(self.end)


def accumulate_time(method):
    def timed(*args, **kw):
        if hasattr(args[0], "slt_not_time"):
            if args[0].slt_not_time:
                return method(*args, **kw)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = method(*args, **kw)
        end.record()
        torch.cuda.synchronize()

        st = start.elapsed_time(end)
        if not hasattr(args[0], "slt_time_summary"):
            args[0].slt_time_summary = {}
            args[0].slt_n_summary = {}
            args[0].slt_n_level = {}

        if method.__name__ in args[0].slt_time_summary:
            args[0].slt_time_summary[method.__name__] += st
            args[0].slt_n_summary[method.__name__] += 1
        else:
            args[0].slt_time_summary[method.__name__] = st
            args[0].slt_n_summary[method.__name__] = 1
            args[0].slt_n_level[method.__name__] = 0

        return result

    return timed


class SystemLevelContextTimer:
    def __init__(self, parent, name="") -> None:
        self.parent = parent
        if hasattr(self.parent, "slt_not_time"):
            if self.parent.slt_not_time:
                return

        self.name = name
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        if hasattr(self.parent, "slt_not_time"):
            if self.parent.slt_not_time:
                return
        self.tic()

    def __exit__(self, exc_type, exc_value, exc_tb):
        if hasattr(self.parent, "slt_not_time"):
            if self.parent.slt_not_time:
                return

        st = self.toc()

        if not hasattr(self.parent, "slt_time_summary"):
            self.parent.slt_time_summary = {}
            self.parent.slt_n_summary = {}
            self.parent.slt_n_level = {}

        if self.name in self.parent.slt_time_summary:
            self.parent.slt_time_summary[self.name] += st
            self.parent.slt_n_summary[self.name] += 1
        else:
            self.parent.slt_time_summary[self.name] = st
            self.parent.slt_n_summary[self.name] = 1
            self.parent.slt_n_level[self.name] = 1

    def tic(self):
        self.start.record()

    def toc(self):
        self.end.record()
        torch.cuda.synchronize()
        return self.start.elapsed_time(self.end)


class SystemLevelTimer:
    def __init__(self, objects, names) -> None:
        self.objects = objects
        self.names = names

    def __str__(self):
        s = ""
        for n, o in zip(self.names, self.objects):
            if hasattr(o, "slt_time_summary"):
                s += n
                for (k, v) in o.slt_time_summary.items():
                    n = o.slt_n_summary[k]
                    spacing = int(o.slt_n_level[k] * 5)

                    s += (
                        "\n  +"
                        + "-" * spacing
                        + f"-  {k}:".ljust(35 - spacing)
                        + f"{round(v,2)}ms".ljust(20)
                        + f"counts: {n} ".ljust(15)
                    )
                s += "\n"
        return s


def time_function(method):
    def timed(*args, **kw):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = method(*args, **kw)
        end.record()
        torch.cuda.synchronize()
        st = start.elapsed_time(end)
        print(f"Function took: {st}")
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
