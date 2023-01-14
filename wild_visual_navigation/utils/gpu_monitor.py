import torch
import os


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


if __name__ == "__main__":

    print("Start gpu memory measuring using context manager")

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
