from pynvml import *
import torch
import os

class GpuUsage:
    def __init__(self, name="") -> None:
        self.name = name
        nvmlInit()
        self.pid = os.getpid()
        self.handle = nvmlDeviceGetHandleByIndex(0)
        self.info = nvmlDeviceGetMemoryInfo(self.handle)
        self.mem_start = 0
        self.mem_end = 0

    def __enter__(self):
        self.mem_start = self.query_gpu_memory()
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        print(f"Memory {self.name}: {self.query_gpu_memory() - self.mem_start} MB")

    def query_gpu_memory(self):
        mem = 0
        # Get info of running processes
        processes = nvmlDeviceGetComputeRunningProcesses(self.handle)

        for proc in processes:
            # We only show the info of the current process
            if proc.pid == self.pid:
                mem =  proc.usedGpuMemory / (1024 * 1024)
        
        return mem

if __name__ == "__main__":

    print("Start gpu memory measuring using context manager")

    tensors = []
    with GpuUsage(f"Test: Total experiment"):
        # The total memory associated to the experiment should be the default 
        # amount allocated by pytorch, will be free after termination

        with GpuUsage(f"Test: Total increment"):
            # This will allocate Pytorch's default, and will extend the size 
            # only if the new tensor doesn't fit
            for i in range(5):
                s = 10**i
                with GpuUsage(f"Test: Create {s}x{s} tensor"):
                    tensors.append(torch.zeros((s,s), device="cuda"))
        
        with GpuUsage(f"Test: Delete tensors"):
            # This doesn't do anything to the memory
            del tensors
        
        with GpuUsage(f"Test: Empty cache"):
            # This frees the total memory allocated without freeing the default
            torch.cuda.empty_cache()
    
