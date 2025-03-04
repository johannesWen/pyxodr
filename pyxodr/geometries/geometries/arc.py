import numpy as np
import matplotlib.pyplot as plt
from pyxodr.geometries.base import Geometry


class Arc(Geometry):
    """
    Represents an arc with constant curvature. Coordinates are computed using:
    
        x(s) = sin(κ s) / κ,
        y(s) = (1 - cos(κ s)) / κ,
    
    so that the arc starts at (0,0) with an initial tangent of (1,0).
    """

    def __init__(self, curvature: float, length: float):
        self.curvature = curvature
        self.length = length

    def __call__(self, p_array: np.ndarray) -> np.ndarray:
        """
        Compute local (x, y) coordinates for parameters in [0, 1].
        
        If the computed starting tangent deviates too far from (1,0), a warning is printed
        and an empty array is returned so that this Arc element can be skipped downstream.
        """
        # Compute the actual arc-length for each parameter.
        s = p_array * self.length

        # Handle nearly-zero curvature as a straight line.
        if np.isclose(self.curvature, 0.0):
            coords = np.column_stack((s, np.zeros_like(s)))
        else:
            coords = np.column_stack((
                np.sin(self.curvature * s) / self.curvature,
                (1 - np.cos(self.curvature * s)) / self.curvature
            ))
        
        # Verify that the computed arc starts with tangent approximately (1, 0).
        if len(coords) >= 2:
            tangent = coords[1] - coords[0]
            norm = np.linalg.norm(tangent)
            if norm > 0:
                tangent = tangent / norm
            tolerance = 0.15
            # If the tangent deviates too much, skip this Arc element.
            if np.linalg.norm(tangent - np.array([1.0, 0.0])) > tolerance:
                print("Warning: Skipping Arc element because tangent deviates too far from (1,0):", tangent)
                # Return an empty array so that downstream code can detect the skip.
                return np.empty((0, 2))
        return coords

    def u_v_from_u(self, u_array: np.ndarray) -> np.ndarray:
        """Raise an error; this geometry is parameteric with no v from u method."""
        raise NotImplementedError("This geometry is only defined parametrically.")
