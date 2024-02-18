from wild_visual_navigation.utils import SystemLevelGpuMonitor, accumulate_memory
from pytictac import accumulate_time
import time
import torch


def test_monitoring():
    class MyTest:
        def __init__(self):
            self.step = 0
            self.time = 0.1
            self.tensors = []

        @accumulate_memory
        @accumulate_time
        def test_memory_then_timing(self, s):
            time.sleep(s / 1000)
            self.tensors.append(torch.zeros((10 * s, 10 * s), device="cuda"))

        @accumulate_time
        @accumulate_memory
        def test_timing_then_memory(self, s):
            time.sleep(s / 1000)
            self.tensors.append(torch.zeros((4, 10 * s, 10 * s), device="cuda"))

    # Create objects
    my_test = MyTest()
    gpu_monitor = SystemLevelGpuMonitor(
        objects=[my_test],
        names=["test"],
        enabled=True,
        device="cuda",
        store_samples=True,
        skip_n_samples=1,
    )
    # time_monitor = ClassContextTimer(
    #     objects=[my_test],
    #     names=["test"],
    # )

    # Run loop
    for n in range(100):
        print(f"step {n}")
        step = n
        t = n / 10
        gpu_monitor.update(step, t)
        my_test.test_memory_then_timing(n)
        my_test.test_timing_then_memory(n)

    gpu_monitor.store("/tmp")
    # time_monitor.store("/tmp")


if __name__ == "__main__":
    test_monitoring()
