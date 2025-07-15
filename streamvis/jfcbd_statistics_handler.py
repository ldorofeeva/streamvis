import copy
import logging
from threading import RLock

import numpy as np
from bokeh.models import CustomJS, Dropdown

from streamvis.cbd_statistic_tools import AggregatorWithID, NPFIFOArray

PULSE_ID_STEP = 10000

logger = logging.getLogger(__name__)


class CBDStatisticsHandler:
    def __init__(self):
        """Initialize a statistics handler specific for CBD experiments.

        Statistics collected:
            - Number of streaks detected;
            - Length of streaks detected;
            - Bragg Intensity;
            - Background count (Total intensity - Bragg intensity);

        """
        self.number_of_streaks = NPFIFOArray(dtype=int, empty_value=-1, max_span=5_000)
        self.streak_lengths = NPFIFOArray(dtype=float, empty_value=np.nan, max_span=50_000)
        self.bragg_counts = NPFIFOArray(
            dtype=float, empty_value=np.nan, max_span=50_000, aggregate=np.sum
        )
        self.bragg_aggregator = AggregatorWithID(dtype=float, empty_value=np.nan, max_span=750_000)

        self._lock = RLock()

        self.data = dict(
            pulse_id_bins=[],
            nframes=[],
            bad_frames=[],
            sat_pix_nframes=[],
            laser_on_nframes=[],
            laser_on_hits=[],
            laser_on_hits_ratio=[],
            laser_off_nframes=[],
            laser_off_hits=[],
            laser_off_hits_ratio=[],
        )

        self.sum_data = copy.deepcopy(self.data)
        for key, val in self.sum_data.items():
            if key == "pulse_id_bins":
                val.append("Summary")
            else:
                val.append(0)

    @property
    def auxiliary_apps_dropdown(self):
        """Return a button that opens statistics application."""
        js_code = """
        switch (this.item) {
            case "Statistics":
                window.open('/statistics');
                break;
            case "Convergent Beam Diffraction stats":
                window.open('/cbd_stats');
                break;
        }
        """
        auxiliary_apps_dropdown = Dropdown(
            label="Open Auxiliary App", menu=["Convergent Beam Diffraction stats"], width=165
        )
        auxiliary_apps_dropdown.js_on_click(CustomJS(code=js_code))

        return auxiliary_apps_dropdown

    def parse(self, metadata, image):
        """Extract statistics from a metadata and an associated image.

        Args:
            metadata (dict): A dictionary with metadata.
            image (ndarray): An associated image.
        """
        is_hit_frame = metadata.get("is_hit_frame", False)

        if image.shape == (2, 2):
            logger.debug(f"Dummy, skipping")
            return

        # Update Bragg aggregator with hits and non-hits alike
        bragg_counts: list[float] = metadata.get("bragg_counts", [0])
        pulse_id = metadata.get("pulse_id", None)
        self.bragg_aggregator.update(np.array(bragg_counts), pulse_id)

        if not is_hit_frame:
            logger.debug(f"Not hit frame, skipping")
            return

        self.bragg_counts.update(np.array([np.sum(bragg_counts)]))

        number_of_streaks: int = metadata.get("number_of_streaks", 0)
        self.number_of_streaks.update(np.array([number_of_streaks]))

        streak_lengths: list[float] = metadata.get("streak_lengths", [0])
        self.streak_lengths.update(np.array(streak_lengths))

        with self._lock:
            try:
                # since messages can have mixed pulse_id order, search for the current pulse_id_bin
                # in the last 5 entries (5 should be large enough for all data analysis to finish)
                bin_ind = self.data["pulse_id_bins"].index(pulse_id_bin, -5)
            except ValueError:
                # this is a new bin
                bin_ind = -1

            if bin_ind == -1:
                for key, val in self.data.items():
                    if key == "pulse_id_bins":
                        val.append(pulse_id_bin)
                    else:
                        val.append(0)

            self._increment("nframes", bin_ind)

            if "is_good_frame" in metadata and not metadata["is_good_frame"]:
                self._increment("bad_frames", bin_ind)

            if "saturated_pixels" in metadata:
                if metadata["saturated_pixels"] != 0:
                    self._increment("sat_pix_nframes", bin_ind)
            else:
                self.data["sat_pix_nframes"][bin_ind] = np.nan

            if laser_on is not None:
                switch = "laser_on" if laser_on else "laser_off"

                self._increment(f"{switch}_nframes", bin_ind)

                if is_hit_frame:
                    self._increment(f"{switch}_hits", bin_ind)

                self.data[f"{switch}_hits_ratio"][bin_ind] = (
                    self.data[f"{switch}_hits"][bin_ind] / self.data[f"{switch}_nframes"][bin_ind]
                )
                self.sum_data[f"{switch}_hits_ratio"][-1] = (
                    self.sum_data[f"{switch}_hits"][-1] / self.sum_data[f"{switch}_nframes"][-1]
                )
            else:
                self.data["laser_on_nframes"][bin_ind] = np.nan
                self.data["laser_on_hits"][bin_ind] = np.nan
                self.data["laser_on_hits_ratio"][bin_ind] = np.nan
                self.data["laser_off_nframes"][bin_ind] = np.nan
                self.data["laser_off_hits"][bin_ind] = np.nan
                self.data["laser_off_hits_ratio"][bin_ind] = np.nan

    def _increment(self, key, ind):
        self.data[key][ind] += 1
        self.sum_data[key][-1] += 1

    def reset(self):
        """Reset statistics entries."""
        with self._lock:
            for val in self.data.values():
                val.clear()

            for key, val in self.sum_data.items():
                if key != "pulse_id_bins":
                    val[0] = 0

    def clear(self):
        self.number_of_streaks.clear()
        self.streak_lengths.clear()
        self.bragg_counts.clear()
        self.bragg_aggregator.clear()
