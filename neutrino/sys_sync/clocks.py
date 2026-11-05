"""Same-host globaltimer alignment with integer-rational affine arithmetic.

The reported uncertainty is conditional on an affine clock model. It is not a
certified hardware bound, and overlapping uncertainty bands do not prove order.
"""
from dataclasses import asdict, dataclass


class ClockError(ValueError):
    pass


def _samples(rows):
    if len(rows) < 2:
        raise ClockError('At least two bracketed samples per phase are required')
    result = []
    for row in rows:
        if len(row) != 3 or any(type(x) is not int or x <= 0 for x in row):
            raise ClockError('Invalid host/GPU/host clock sample')
        before, tick, after = row
        if before > after:
            raise ClockError('Reversed host clock bracket')
        if result and (before < result[-1][2] or tick < result[-1][1]):
            raise ClockError('Nonmonotonic clock calibration')
        result.append(tuple(row))
    return result


@dataclass(frozen=True)
class AffineClock:
    first: tuple[int, int, int]
    last: tuple[int, int, int]
    residual_ns: int
    sample_count: int

    @property
    def denominator(self):
        return 2 * (self.last[1] - self.first[1])

    @property
    def numerator(self):
        return self.last[0] + self.last[2] - self.first[0] - self.first[2]

    def _twice_host_numerator(self, tick):
        delta = self.last[1] - self.first[1]
        return ((self.first[0] + self.first[2]) * delta
                + (tick - self.first[1]) * self.numerator)

    def map_ns(self, tick):
        if type(tick) is not int or not self.first[1] <= tick <= self.last[1]:
            raise ClockError('Event is outside its contemporaneous calibration anchors')
        value = self._twice_host_numerator(tick)
        return (value + self.denominator // 2) // self.denominator

    def uncertainty_ns(self, tick):
        self.map_ns(tick)
        distance = self.last[1] - self.first[1]
        left = self.last[1] - tick
        right = tick - self.first[1]
        weighted = (left * (self.first[2] - self.first[0])
                    + right * (self.last[2] - self.last[0]))
        return (weighted + 2 * distance - 1) // (2 * distance) + self.residual_ns + 1

    def manifest(self):
        return {**asdict(self), 'reference': 'CLOCK_MONOTONIC_RAW',
                'mapping': 'integer-rational-two-anchor',
                'slope_numerator': self.numerator, 'slope_denominator': self.denominator,
                'drift_ppm': (self.numerator - self.denominator) * 1e6 / self.denominator,
                'uncertainty': 'host brackets plus calibration residual; conditional on affine model',
                'cross_gpu_order': 'unresolved when event uncertainty bands overlap'}


def fit_clock(before, after):
    before, after = _samples(before), _samples(after)
    if before[-1][2] >= after[0][0] or before[-1][1] >= after[0][1]:
        raise ClockError('Calibration phases overlap or clocks reset')
    first = min(before, key=lambda row: row[2] - row[0])
    last = min(after, key=lambda row: row[2] - row[0])
    model = AffineClock(first, last, 0, len(before) + len(after))
    # globaltimer is nominally in ns. Reject grossly incompatible domains;
    # bracket uncertainty is retained, not used as a promise of exact drift.
    if abs(model.numerator - model.denominator) * 1000 > model.denominator * 5:
        raise ClockError('GPU and host clock domains differ by more than 5000 ppm')
    residual = 0
    for a, tick, b in before + after:
        value = model._twice_host_numerator(tick)
        outside = max(0, a * model.denominator - value, value - b * model.denominator)
        residual = max(residual, (outside + model.denominator - 1) // model.denominator)
    return AffineClock(first, last, residual, len(before) + len(after))
