import math
from dataclasses import dataclass

from scipy.integrate import quad
from scipy.special import log_ndtr, ndtri_exp


class UnreliableDistribution(ValueError):
    pass


@dataclass(frozen=True)
class LogNormal:
    mu: float
    sigma: float

    def __post_init__(self):
        if not math.isfinite(self.mu) or not math.isfinite(self.sigma) or self.sigma <= 0:
            raise UnreliableDistribution("invalid_parameters")

    def log_survival(self, t: float) -> float:
        if t < 0 or not math.isfinite(t):
            raise UnreliableDistribution("invalid_age")
        return 0.0 if t == 0 else float(log_ndtr((self.mu - math.log(t)) / self.sigma))

    def summarize(self, age: float, horizon: float) -> dict:
        if not math.isfinite(horizon) or horizon <= 0:
            raise ValueError("H_seconds must be positive and finite")
        ls = self.log_survival(age)
        if ls < math.log(1e-12):
            raise UnreliableDistribution("unsupported_tail")

        def quantile(p):
            log_total = self.mu - self.sigma * float(ndtri_exp(ls + math.log1p(-p)))
            return math.exp(log_total) if age == 0 else age * math.expm1(log_total - math.log(age))

        try:
            shifted = 0.0 if age == 0 else float(log_ndtr((self.mu + self.sigma**2 - math.log(age)) / self.sigma))
            log_mean = self.mu + self.sigma**2 / 2 + shifted - ls
            mean = math.exp(log_mean) if age == 0 else age * math.expm1(log_mean - math.log(age))
            capped = quad(lambda u: math.exp(self.log_survival(age + u) - ls), 0, horizon, epsabs=1e-8)[0]
            result = {
                "remaining_p50_seconds": quantile(0.5),
                "remaining_p90_seconds": quantile(0.9),
                "prob_return_within_H": -math.expm1(self.log_survival(age + horizon) - ls),
                "expected_remaining_seconds": mean,
                "expected_capped_remaining_seconds": capped,
            }
        except (OverflowError, ValueError) as exc:
            raise UnreliableDistribution("nonfinite_result") from exc
        if any(not math.isfinite(v) or v < 0 for v in result.values()):
            raise UnreliableDistribution("nonfinite_result")
        return result
