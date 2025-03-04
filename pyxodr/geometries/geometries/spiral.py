import numpy as np
import matplotlib.pyplot as plt
from pyxodr.geometries._standard_spiral import OdrSpiral
from pyxodr.geometries.base import Geometry


class Spiral(Geometry):
    """
    Class representing an Euler spiral / Clothoid geometry.

    Parameters
    ----------
    length : float
        Length [m] of the spiral.
    curvStart : float
        Curvature at the start of the spiral.
    curvEnd : float
        Curvature at the end of the spiral.
    """

    def __init__(self, length: float, curvStart: float, curvEnd: float):
        self.length = length
        self.curvStart = curvStart
        self.curvEnd = curvEnd
        self.curvature_rate_of_change = (self.curvEnd - self.curvStart) / self.length
        self.standard_spiral = OdrSpiral(self.curvature_rate_of_change)

    def __call__(self, p_array: np.ndarray) -> np.ndarray:
        """
        Return local (u, v) coordinates from an array of parameter p in [0.0, 1.0].

        The method:
          1. Adjusts the parameter s based on the starting curvature.
          2. Computes the standard spiral coordinates.
          3. Translates and rotates the spiral so that its starting point is (0,0)
             and its initial tangent aligns with (1,0).
          4. If the computed tangent deviates too much from (1,0), prints a warning
             and returns an empty array to signal skipping this element.
        """
        # Compute an s offset to account for starting curvature.
        s_offset = self.curvStart / self.curvature_rate_of_change
        offset_s_array = p_array * self.length + s_offset

        # Compute standard spiral coordinates (each is (x, y, t)).
        standard_coords = [self.standard_spiral(s) for s in offset_s_array]
        xy = np.array([(x, y) for x, y, t in standard_coords])

        # Translate so that the spiral starts at (0,0).
        x0, y0, t0 = standard_coords[0]
        xy_at_origin = xy - np.array([x0, y0])

        # Rotate so that the initial tangent aligns with the positive x-axis.
        angular_difference = t0
        c, s = np.cos(angular_difference), np.sin(angular_difference)
        rotation_matrix = np.array([[c, s], [-s, c]])
        rotated_xy_at_origin = np.dot(rotation_matrix, xy_at_origin.T).T

        # Compute the tangent using the first two points.
        if len(rotated_xy_at_origin) >= 2:
            tangent = rotated_xy_at_origin[1] - rotated_xy_at_origin[0]
            norm = np.linalg.norm(tangent)
            if norm > 0:
                tangent /= norm
            # Set tolerance for deviation from (1,0).
            tolerance = 0.15
            deviation = np.linalg.norm(tangent - np.array([1.0, 0.0]))
            if deviation > tolerance:
                print(f"Warning: Skipping Spiral element because tangent deviates too far from (1,0): {tangent}")
                return np.empty((0, 2))
        return rotated_xy_at_origin

    def u_v_from_u(self, u_array: np.ndarray) -> np.ndarray:
        """Raise an error; this geometry is parameteric with no v from u method."""
        raise NotImplementedError("This geometry is only defined parametrically.")
