"""Test that the lane lines appear to be drivable in every loaded network."""

import os
from typing import Optional, Set
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from pyxodr.road_objects.network import RoadNetwork

ignored_lane_types = [None, set(["sidewalk", "shoulder"]), set(["driving"])]
ignored_lane_types = [None, set(["sidewalk", "shoulder", "curb"]), set(["restricted"])]

if __name__ == "__main__":
    base_path = Path(__file__).parent
    xodr_path = base_path / "xodr" / "multi_intersections.xodr"
    # xodr_path = base_path / "xodr" / "4A_PG_v3.xodr"
    # xodr_path = base_path / "xodr" / "fabriksgatan.xodr"
    # xodr_path = base_path / "xodr" / "Example_1_Good_Quality_fixed.xodr"
    # xodr_path = base_path / "xodr" / "Example_2_Medium_Issues_fixed.xodr"
    # xodr_path = base_path / "xodr" / "Example_3_Major_Issues_fixed.xodr"
    rn = RoadNetwork(xodr_path, ignored_lane_types=ignored_lane_types, resolution=0.01)
    road_network_name = os.path.basename(xodr_path).split(".")[0]

    plt.figure(figsize=(20, 20))
    ax = plt.gca()
    rn.plot(ax, plot_start_and_end=False, label_size=None, plot_junctions=False, plot_lane_centres=True, line_scale_factor=0.3)
    plt.axis("equal")
    plt.savefig(
        os.path.join(
            "output",
            f"{road_network_name}.pdf",
        )
    )
    plt.close()