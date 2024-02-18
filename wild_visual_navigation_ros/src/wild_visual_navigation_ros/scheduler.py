#                                                                               
# Copyright (c) 2022-2024, ETH Zurich, Matias Mattamala, Jonas Frey.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#                                                                               
class Scheduler:
    """
    Implements a modified weighted round-robin scheduler
    https://en.wikipedia.org/wiki/Weighted_round_robin

    Given a list of processes and corresponding weights,
    it will produce a process schedule that assigns a fair ordering for
    the processes given their weight

    Note: This does not handle process execution but only the schedule
    """

    def __init__(self):
        self._processes = {}
        self._schedule = []
        self._idx = 0

    def add_process(self, name, weight: int = 1):
        # Adds a process to the schedule with some weight (int)
        self._processes[name] = weight
        self._make_schedule()

    def step(self):
        # Move the queue to the next process
        self._idx = (self._idx + 1) % len(self._schedule)

    def get(self):
        # Gets the current process in the schedule
        if len(self._schedule) == 0:
            return None
        else:
            # Get the current scheduled process
            return self._schedule[self._idx]

    @property
    def schedule(self):
        return self._schedule

    def _make_schedule(self):
        # Reset schedule
        self._schedule = []

        # Prepare some lists
        weights = list(self._processes.values())
        processes = list(self._processes.keys())

        # Get the total weight
        w_total = sum(weights)

        # Get the number of processes
        num_processes = len(processes)

        # Prepare queues
        queues = [[p] * w for p, w in zip(processes, weights)]

        # Fill schedule
        for w in range(w_total):
            for i in range(num_processes):
                if len(queues[i]) > 0 and weights[i] > w:
                    self._schedule.append(queues[i].pop())


def run_scheduler():
    s = Scheduler()
    s.add_process("p1", 1)
    s.add_process("p2", 1)
    s.add_process("p3", 1)
    print(f"Schedule: {s.schedule}")

    assert s.get() == "p1"
    s.step()
    assert s.get() == "p2"
    s.step()
    assert s.get() == "p3"
    s.step()
    assert s.get() == "p1"
    s.step()

    s = Scheduler()
    s.add_process("p1", 2)
    s.add_process("p2", 1)
    s.add_process("p3", 2)
    s.add_process("p4", 1)
    print(f"Schedule: {s.schedule}")

    assert s.get() == "p1"
    s.step()
    assert s.get() == "p2"
    s.step()
    assert s.get() == "p3"
    s.step()
    assert s.get() == "p4"
    s.step()
    assert s.get() == "p1"
    s.step()
    assert s.get() == "p3"
    s.step()


if __name__ == "__main__":
    run_scheduler()
