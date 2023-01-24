import torch
import time
import copy
import pickle
import os
from functools import wraps


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
    @wraps(method)
    def timed(*args, **kw):
        if not hasattr(args[0], "slt_enabled"):
            return method(*args, **kw)
        elif not args[0].slt_enabled:
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
            args[0].slt_time_squared_summary = {}

        if method.__name__ in args[0].slt_time_summary:
            args[0].slt_time_summary[method.__name__] += st
            args[0].slt_time_squared_summary[method.__name__] += st**2
            args[0].slt_n_summary[method.__name__] += 1
        else:
            args[0].slt_time_summary[method.__name__] = st
            args[0].slt_time_squared_summary[method.__name__] = st**2
            args[0].slt_n_summary[method.__name__] = 1
            args[0].slt_n_level[method.__name__] = 0

        return result

    return timed


class SystemLevelContextTimer:
    def __init__(self, parent, name="") -> None:
        self.parent = parent
        if not hasattr(self.parent, "slt_enabled"):
            return
        elif not self.parent.slt_enabled:
            return

        self.name = name
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        if not hasattr(self.parent, "slt_enabled"):
            return
        elif not self.parent.slt_enabled:
            return

        self.tic()

    def __exit__(self, exc_type, exc_value, exc_tb):
        if not hasattr(self.parent, "slt_enabled"):
            return
        elif not self.parent.slt_enabled:
            return

        st = self.toc()

        if not hasattr(self.parent, "slt_time_summary"):
            self.parent.slt_time_summary = {}
            self.parent.slt_time_squared_summary = {}
            self.parent.slt_n_summary = {}
            self.parent.slt_n_level = {}

        if self.name in self.parent.slt_time_summary:
            self.parent.slt_time_summary[self.name] += st
            self.parent.slt_time_squared_summary[self.name] += st**2
            self.parent.slt_n_summary[self.name] += 1
        else:
            self.parent.slt_time_summary[self.name] = st
            self.parent.slt_time_squared_summary[self.name] = st**2
            self.parent.slt_n_summary[self.name] = 1
            self.parent.slt_n_level[self.name] = 1

    def tic(self):
        self.start.record()

    def toc(self):
        self.end.record()
        torch.cuda.synchronize()
        return self.start.elapsed_time(self.end)


class SystemLevelTimer:
    def __init__(self, objects, names, enabled=True) -> None:
        self.objects = objects
        self.names = names

        for o in self.objects:
            o.slt_enabled = enabled

    def __str__(self):
        s = ""
        for n, o in zip(self.names, self.objects):
            if hasattr(o, "slt_time_summary"):
                s += n
                for (k, v) in o.slt_time_summary.items():
                    n = o.slt_n_summary[k]
                    spacing = int(o.slt_n_level[k] * 5)
                    mean = v / n
                    std = round(((o.slt_time_squared_summary[k] / n) - (mean**2)) ** 0.5, 3)
                    s += (
                        "\n  +"
                        + "-" * spacing
                        + f"-  {k}:".ljust(35 - spacing)
                        + f"{round(v,2)}ms".ljust(18)
                        + f"counts: {n} ".ljust(18)
                        + f"std: {std} ".ljust(18)
                        + f"mean: {round(mean,3)} ".ljust(18)
                    )
                s += "\n"
        return s

    def store(self, folder, key="system_level_timer"):
        res = {}
        for n, o in zip(self.names, self.objects):
            if hasattr(o, "slt_time_summary"):
                mean = {k: o.slt_time_summary[k] / o.slt_n_summary[k] for k in o.slt_time_summary.keys()}
                std = {
                    k: (o.slt_time_squared_summary[k] / o.slt_n_summary[k] - mean[k] ** 2) ** 0.5
                    for k in o.slt_time_summary.keys()
                }
                store = [o.slt_n_level, o.slt_time_summary, std, mean]
                res[n] = copy.deepcopy(store)

        with open(os.path.join(folder, key + ".pkl"), "wb") as f:
            pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)

        s = self.__str__()
        with open(os.path.join(folder, key + ".txt"), "w") as f:
            for l in s.split("\n"):
                f.write(l + "\n")


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
