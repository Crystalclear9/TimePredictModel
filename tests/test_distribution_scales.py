"""A wide residence horizon must not erase a narrow, short-duration distribution."""
import math
import itertools

import pytest
from scipy.stats import norm
from scipy.integrate import quad

from timepredict.distribution import LogNormal


@pytest.mark.parametrize("sigma", [.1, .5, 1.])
@pytest.mark.parametrize("waited", [False, True])
def test_capped_cost_for_microsecond_calls_with_seconds_horizon(sigma, waited):
    mu, horizon = -12., 2.
    age = math.exp(mu) if waited else 0.
    expected = math.exp(mu+sigma*sigma/2)
    if waited:
        expected = expected*norm.sf(-sigma)/.5-age
    # Even the widest case has negligible probability/moment beyond two seconds.
    result = LogNormal(mu, sigma).summarize(age, horizon)
    assert result["expected_capped_remaining_seconds"] == pytest.approx(expected, rel=1e-9)
    assert 0 < result["expected_capped_remaining_seconds"] <= horizon


def test_capped_cost_against_independent_normal_density_integral_across_scales():
    # Integrate over Z~N(0,1), not over remaining time, so the oracle cannot
    # miss a narrow duration peak in a large physical-time interval.
    for mu, sigma, age_ratio, horizon_ratio in itertools.product(
            (-20., -5., 0., 15.), (.1, .5, 1., 2.), (0., .2, 1., 2.), (1e-6, 1e-3, 1., 1e3, 1e6)):
        scale = math.exp(mu)
        lower = math.log(age_ratio)/sigma if age_ratio else -12.
        survival = norm.sf(lower)
        if survival < 1e-12:
            continue
        crossing = math.log(age_ratio+horizon_ratio)/sigma
        points = [crossing] if lower < crossing < 12. else []
        expected = quad(lambda z: min(max(math.exp(sigma*z)-age_ratio, 0.), horizon_ratio)
                        *math.exp(-z*z/2)/math.sqrt(2*math.pi)/survival,
                        lower, 12., points=points, epsabs=1e-14, epsrel=1e-10)[0]
        result = LogNormal(mu, sigma).summarize(scale*age_ratio, scale*horizon_ratio)
        actual = result["expected_capped_remaining_seconds"]/scale
        assert actual == pytest.approx(expected, rel=2e-8, abs=1e-13), (mu, sigma, age_ratio, horizon_ratio)
