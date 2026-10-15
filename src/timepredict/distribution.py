# 文件作用：对数正态分布数学：在 T>age 条件下计算剩余分位数、窗口概率和期望。
import math
from dataclasses import dataclass

from scipy.integrate import quad
from scipy.special import log_ndtr, ndtri_exp


# 表示分布参数、等待年龄或尾部支持不可靠。
class UnreliableDistribution(ValueError):
    pass


# 不可变的 log(T) 正态参数；mu 为对数位置，sigma 为对数尺度。
@dataclass(frozen=True)
class LogNormal:
    mu: float
    sigma: float

    # 拒绝非有限参数和非正 sigma。
    def __post_init__(self):
        if not math.isfinite(self.mu) or not math.isfinite(self.sigma) or self.sigma <= 0:
            raise UnreliableDistribution("invalid_parameters")

    # 在对数域计算 log P(T>t)，减少长尾概率下溢。
    def log_survival(self, t: float) -> float:
        if t < 0 or not math.isfinite(t):
            raise UnreliableDistribution("invalid_age")
        return 0.0 if t == 0 else float(log_ndtr((self.mu - math.log(t)) / self.sigma))

    # 计算条件剩余时间摘要；生存概率低于 1e-12 时拒绝外推。
    def summarize(self, age: float, horizon: float) -> dict:
        if not math.isfinite(horizon) or horizon <= 0:
            raise ValueError("H_seconds must be positive and finite")
        ls = self.log_survival(age)
        if ls < math.log(1e-12):
            raise UnreliableDistribution("unsupported_tail")

        # 反演条件生存概率；使用 expm1 稳定计算总分位数减去已等待时间。
        def quantile(p):
            log_total = self.mu - self.sigma * float(ndtri_exp(ls + math.log1p(-p)))
            return math.exp(log_total) if age == 0 else age * math.expm1(log_total - math.log(age))

        try:
            shifted = 0.0 if age == 0 else float(log_ndtr((self.mu + self.sigma**2 - math.log(age)) / self.sigma))
            log_mean = self.mu + self.sigma**2 / 2 + shifted - ls
            mean = math.exp(log_mean) if age == 0 else age * math.expm1(log_mean - math.log(age))
            p50, p90 = quantile(0.5), quantile(0.9)
            log_tail_ratio = self.log_survival(age+horizon)-ls
            within = -math.expm1(log_tail_ratio)
            if horizon <= p50:
                # The integrand stays >= 1/2 here. Normalize the interval so
                # absolute integration tolerances also work for tiny horizons.
                capped = horizon*quad(lambda v: math.exp(self.log_survival(age+horizon*v)-ls),
                                      0., 1., epsabs=1e-10, epsrel=1e-10)[0]
            else:
                # A single integral over [0,H] can miss all probability mass
                # when microsecond calls are capped at seconds. Use the exact
                # truncated first moment, with stable differences in log space.
                shifted_end = float(log_ndtr((self.mu+self.sigma**2-math.log(age+horizon))/self.sigma))
                fraction = -math.expm1(shifted_end-shifted)
                moment = math.exp(log_mean+math.log(fraction)) if fraction > 0 else 0.
                capped = math.fsum((moment, -age*within, horizon*math.exp(log_tail_ratio)))
            capped = min(capped, horizon, mean)
            if capped <= 0:
                raise ValueError("nonpositive_capped_cost")
            result = {
                "remaining_p50_seconds": p50,
                "remaining_p90_seconds": p90,
                "prob_return_within_H": within,
                "expected_remaining_seconds": mean,
                "expected_capped_remaining_seconds": capped,
            }
        except (OverflowError, ValueError) as exc:
            raise UnreliableDistribution("nonfinite_result") from exc
        if any(not math.isfinite(v) or v < 0 for v in result.values()):
            raise UnreliableDistribution("nonfinite_result")
        return result
