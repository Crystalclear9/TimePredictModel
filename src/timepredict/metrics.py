# 文件作用：有界滚动延迟样本统计；分位数来自保留窗口，总 count 单独累计。
"""Bounded histograms for process and end-to-end latency (seconds)."""
from collections import defaultdict, deque
import math


# 按指标名维护固定容量样本队列与累计数量。
class Histograms:
    # 初始化每个指标的滚动队列，默认最多保留 8192 个值。
    def __init__(self, capacity=8192):
        self.values = defaultdict(lambda: deque(maxlen=capacity))
        self.counts = defaultdict(int)

    # 只记录有限且非负的秒数。
    def observe(self, name, value):
        if math.isfinite(value) and value >= 0:
            self.values[name].append(float(value))
            self.counts[name] += 1

    # 排序保留样本，输出 P50/P90/P99、窗口最大值及总计数。
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
