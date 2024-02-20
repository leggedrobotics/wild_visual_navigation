#
# Copyright (c) 2022-2024, ETH Zurich, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
import torch
import os
import pandas as pd
from functools import wraps


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


def accumulate_memory(method):
    @wraps(method)
    def measured(*args, **kw):
        if not hasattr(args[0], "slg_enabled") or torch.cuda.device_count() < 1:
            return method(*args, **kw)

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

            if not hasattr(args[0], "slg_step") or not hasattr(args[0], "slg_time"):
                # Wait until the step and step time is initialized
                return result

            store_sample = args[0].slg_step % args[0].slg_skip_n_samples == 0

            if method_name in args[0].slg_memory_samples and store_sample:
                args[0].slg_memory_samples[method_name]["step"].append(args[0].slg_step)
                args[0].slg_memory_samples[method_name]["time"].append(args[0].slg_time)
                args[0].slg_memory_samples[method_name]["ini_memory"].append(start)
                args[0].slg_memory_samples[method_name]["end_memory"].append(end)
                args[0].slg_memory_samples[method_name]["delta_memory"].append(delta)
            else:
                args[0].slg_memory_samples[method_name] = {
                    "step": [args[0].slg_step],
                    "time": [args[0].slg_time],
                    "ini_memory": [start],
                    "end_memory": [end],
                    "delta_memory": [delta],
                }

        return result

    return measured


class SystemLevelContextGpuMonitor:
    def __init__(self, parent, name="") -> None:
        self.parent = parent
        if hasattr(self.parent, "slg_enabled"):
            if not self.parent.slg_enabled:
                return
        self.name = name

    def __enter__(self):
        if (
            hasattr(self.parent, "slg_enabled")
            and hasattr(self.parent, "slg_device")
            and hasattr(self.parent, "slg_pid")
        ):
            if self.parent.slg_enabled:
                self.mem_start = gpu_memory_query(self.parent.slg_device, self.parent.slg_pid)

    def __exit__(self, exc_type, exc_value, exc_tb):
        if (
            hasattr(self.parent, "slg_enabled")
            and hasattr(self.parent, "slg_device")
            and hasattr(self.parent, "slg_pid")
        ):
            if self.parent.slg_enabled:
                mem_end = gpu_memory_query(self.parent.slg_device, self.parent.slg_pid)
                mem_delta = mem_end - self.mem_start
        else:
            return

        if hasattr(self.parent, "slg_enabled") and hasattr(self.parent, "slg_store_samples"):
            if self.parent.slg_enabled and self.parent.slg_store_samples:
                # Check if we enabled the option to store samples
                if not hasattr(self.parent, "slg_memory_samples"):
                    self.parent.slg_memory_samples = {}

                if not hasattr(self.parent, "slg_step") or not hasattr(self.parent, "slg_time"):
                    # Wait until the step and step time is initialized
                    return

                store_sample = self.parent.slg_step % self.parent.slg_skip_n_samples == 0

                if self.name in self.parent.slg_memory_samples and store_sample:
                    self.parent.slg_memory_samples[self.name]["step"].append(self.parent.slg_step)
                    self.parent.slg_memory_samples[self.name]["time"].append(self.parent.slg_time)
                    self.parent.slg_memory_samples[self.name]["ini_memory"].append(self.mem_start)
                    self.parent.slg_memory_samples[self.name]["end_memory"].append(mem_end)
                    self.parent.slg_memory_samples[self.name]["delta_memory"].append(mem_delta)
                else:
                    self.parent.slg_memory_samples[self.name] = {
                        "step": [self.parent.slg_step],
                        "time": [self.parent.slg_time],
                        "ini_memory": [self.mem_start],
                        "end_memory": [mem_end],
                        "delta_memory": [mem_delta],
                    }

    def tic(self):
        self.start.record()

    def toc(self):
        self.end.record()
        torch.cuda.synchronize()
        return self.start.elapsed_time(self.end)


class SystemLevelGpuMonitor:
    def __init__(
        self,
        objects,
        names,
        enabled=True,
        device=None,
        store_samples=False,
        skip_n_samples=1,
    ) -> None:
        self.objects = objects
        self.names = names

        for o in self.objects:
            o.slg_skip_n_samples = max(skip_n_samples, 1)
            o.slg_enabled = enabled
            o.slg_pid = os.getpid()
            o.slg_device = device
            o.slg_store_samples = store_samples

    def __str__(self):
        pass

    def update(self, step, time):
        # print(f"New step {step}, time {time}")
        for o in self.objects:
            if o.slg_enabled:
                o.slg_step = step
                o.slg_time = time

    def store(self, folder):
        base_folder = folder + "/gpu_monitor"
        os.makedirs(base_folder, exist_ok=True)
        print(f"Saving gpu samples to {base_folder}...")
        # We iterate the list of object to access the recorded data
        for n, o in zip(self.names, self.objects):
            if hasattr(o, "slg_memory_samples"):
                # Each object has a slg_memory_samples dict
                # Each entry of the dict is a numpy array with samples
                for k, v in o.slg_memory_samples.items():
                    out_filename = f"{base_folder}/{n}_{k}.csv"
                    print(f"   {out_filename}")
                    df = pd.DataFrame(v)
                    df.to_csv(out_filename)


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
            self.step = 0
            self.time = 0.1
            self.tensors = []

        @accumulate_memory
        def test(self, s):
            self.tensors.append(torch.zeros((s, s), device="cuda"))

        @accumulate_memory
        def test2(self, s):
            self.tensors.append(torch.zeros((4, s, s), device="cuda"))

    i = 0
    t = 0.1
    my_test = MyTest()
    gpu_monitor = SystemLevelGpuMonitor(
        objects=[my_test],
        names=["test"],
        enabled=True,
        device="cuda",
        store_samples=True,
        skip_n_samples=1,
    )
    for n in range(400):
        step = n
        time = n / 10
        gpu_monitor.update(step, time)
        my_test.test(n)
        my_test.test2(n)

    gpu_monitor.store("/tmp")
