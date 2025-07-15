import logging

import numpy as np

logger = logging.getLogger(__name__)


class NPFIFOArray:
    def __init__(self, dtype, empty_value, max_span=120_000, aggregate=np.average):
        self._x = np.full(shape=(max_span,), dtype=dtype, fill_value=empty_value)
        self._nan = empty_value
        self.last_value = self._nan
        self._aggregate = aggregate

    def update(self, values):
        self._x = np.roll(self._x, len(values))
        self._x[: len(values)] = values
        self.last_value = self._aggregate(values)

    def clear(self):
        self._x[...] = self._nan

    def __bool__(self):
        return bool(np.any(self._x != self._nan))

    def __call__(self, *args, **kwargs):
        return self._x[self._x != self._nan]

    @property
    def min(self):
        return np.min(self.__call__())

    @property
    def max(self):
        return np.max(self.__call__())


class AggregatorWithID:
    def __init__(self, dtype, empty_value, max_span=500_000, aggregate=np.sum):
        self._x = np.full(shape=(max_span,), dtype=dtype, fill_value=empty_value)
        self._id = np.full(shape=(max_span,), dtype=int, fill_value=0)
        self._nan = empty_value
        self.last_value = self._nan
        self._aggregate = aggregate
        self.last_processed_index = 0

    def update(self, values, pulse_id):
        if pulse_id is None:
            logger.warning(f"Can't update the aggregator: pulse Id is None")
            return

        # Only store aggregated value per pulse Id
        value = self._aggregate(values)
        self._x = np.roll(self._x, -1)
        self._x[-1] = value
        self.last_value = value

        self._id = np.roll(self._id, -1)
        self._id[-1] = pulse_id

        self.last_processed_index -= 1

    @property
    def count(self) -> int:
        return len(self._x[self._x != self._nan])

    def clear(self):
        self._x[...] = self._nan
        self._id[...] = 0
        self.last_processed_index = 0

    def __bool__(self):
        return bool(np.any(self._x != self._nan))

    def __call__(self, *args, **kwargs):
        if not self or self.last_processed_index == 0:
            return [], []
        start_id = self.last_processed_index
        self.last_processed_index = 0
        return self._x[start_id:], self._id[start_id:]

    @property
    def last(self):
        return self._x[-1], self._id[-1]
