"""Bounded histograms for process and end-to-end latency (seconds)."""
from collections import defaultdict, deque
import math


class Histograms:
    def __init__(self, capacity=8192):
        self.values = defaultdict(lambda: deque(maxlen=capacity))
        self.counts = defaultdict(int)

    def observe(self, name, value):
        if math.isfinite(value) and value >= 0:
            self.values[name].append(float(value))
            self.counts[name] += 1

    def snapshot(self):
        output = {}
        for name, values in self.values.items():
            ordered = sorted(values)
            if not ordered:
                continue
            output[name] = {"count": self.counts[name], "retained": len(ordered),
                            **{f"p{q}": ordered[min(len(ordered)-1, int((len(ordered)-1)*q/100))] for q in (50, 90, 99)},
                            "max": ordered[-1]}
        return output
