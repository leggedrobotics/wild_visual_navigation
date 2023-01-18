import torch
import os
import numpy as np


def pynvml_gpu_memory_query(device, pid):
    import pynvml
    from pynvml import NVMLError_DriverNotLoaded

    try:
        pynvml.nvmlInit()
    except NVMLError_DriverNotLoaded:
        return 0
    # Query memory
    device = torch.cuda._get_device_index(device, optional=True)
    handle = pynvml.nvmlDeviceGetHandleByIndex(device)
    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    mem = 0
    if len(procs) == 0:
        return 0
    # Filter by pid number
    for p in procs:
        mem = p.usedGpuMemory / (1024 * 1024) if p.pid == pid else 0
    return mem


def jtop_gpu_memory_query(device, pid):
    from jtop import jtop
    import psutil

    process = psutil.Process(pid)
    mem = process.memory_info().rss / (1024 * 1024)
    return mem


def gpu_memory_query(device, pid):
    # Try nvidia-smi query
    try:
        return pynvml_gpu_memory_query(device, pid)
    except ModuleNotFoundError:
        pass

    # Try jetson-stats query
    try:
        return jtop_gpu_memory_query(device, pid)
    except ModuleNotFoundError:
        pass

    return 0


class GpuMonitor:
    def __init__(self, name="", device=0) -> None:
        self.name = name
        self.pid = os.getpid()
        self.mem_start = 0
        self.device = device

    def __enter__(self):
        self.mem_start = self.query_gpu_memory()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(f"Memory {self.name}: {self.query_gpu_memory() - self.mem_start} MB")

    def query_gpu_memory(self):
        return gpu_memory_query(self.device, self.pid)


def accumulate_memory(method, **kw):
    def measured(*args, **kw):
        if not hasattr(args[0], "slg_enabled") or torch.cuda.device_count() < 1:
            if not args[0].slg_enabled:
                return method(*args, **kw)

        start = gpu_memory_query(args[0].slg_device, args[0].slg_pid)
        result = method(*args, **kw)
        end = gpu_memory_query(args[0].slg_device, args[0].slg_pid)
        delta = end - start

        method_name = method.__name__

        # Check if we enabled the option to store samples
        if args[0].slg_store_samples:
            if not hasattr(args[0], "slg_memory_samples"):
                args[0].slg_memory_samples = {}

            if method_name in args[0].slg_memory_samples:
                args[0].slg_memory_samples[method_name]["step"].append(args[0].slg_step)
                args[0].slg_memory_samples[method_name]["time"].append(args[0].slg_time)
                args[0].slg_memory_samples[method_name]["delta_memory"].append(delta)
            else:
                args[0].slg_memory_samples[method_name] = {
                    "step": [args[0].slg_step],
                    "time": [args[0].slg_time],
                    "delta_memory": [delta],
                }

        return result

    return measured


class SystemLevelGpuMonitor:
    def __init__(
        self, objects, names, enabled=True, device=None, store_samples=False, step_reference=0, time_reference=0
    ) -> None:
        self.objects = objects
        self.names = names

        for o in self.objects:
            o.slg_enabled = enabled
            o.slg_pid = os.getpid()
            o.slg_device = device
            o.slg_store_samples = store_samples
            o.slg_step = step_reference
            o.slg_time = time_reference

    def __str__(self):
        pass

    def store(self, folder):
        # We iterate the list of object to access the recorded data
        for n, o in zip(self.names, self.objects):
            if hasattr(o, "slg_memory_samples"):
                # Each object has a slg_memory_samples dict
                # Each entry of the dict is a numpy array with samples
                for (k, v) in o.slg_memory_samples.items():
                    out_filename = f"{folder}/{n}_{k}.npy"
                    print(f"Saving gpu samples to {out_filename}...")
                    np.save(out_filename, v)


if __name__ == "__main__":

    print("GPU memory measuring using context manager")

    tensors = []
    with GpuMonitor(f"Test: Total experiment"):
        # The total memory associated to the experiment should be the default
        # amount allocated by pytorch, will be free after termination

        with GpuMonitor(f"Test: Total increment"):
            # This will allocate Pytorch's default, and will extend the size
            # only if the new tensor doesn't fit
            for i in range(5):
                s = 10**i
                with GpuMonitor(f"Test: Create {s}x{s} tensor"):
                    tensors.append(torch.zeros((s, s), device="cuda"))

        with GpuMonitor(f"Test: Delete tensors"):
            # This doesn't do anything to the memory
            del tensors

        with GpuMonitor(f"Test: Empty cache"):
            # This frees the total memory allocated without freeing the default
            torch.cuda.empty_cache()

    print("GPU memory measuring using SystemLevelGpuMonitor")

    class MyTest:
        def __init__(self):
            pass

        @accumulate_memory
        def test(self, s):
            tensors = []
            tensors.append(torch.zeros((s, s), device="cuda"))

    my_test = MyTest()
    gpu_monitor = SystemLevelGpuMonitor(
        objects=[my_test],
        names=["test"],
        enabled=True,
        device="cuda",
        store_samples=True,
        step_reference=0,
        time_reference=0.1,
    )
    for i in range(5):
        s = 10**i
        my_test.test(10)

    gpu_monitor.store("/tmp")
