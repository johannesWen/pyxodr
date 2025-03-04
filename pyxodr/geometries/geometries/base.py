from abc import ABC, abstractmethod

import numpy as np

from pyxodr.utils.array import fix_zero_directions


class Geometry(ABC):
    """Base class for geometry objects."""

    @staticmethod
    def global_coords_from_offsets(
        local_coords: np.ndarray,
        x_offset: float,
        y_offset: float,
        heading_offset: float,
    ) -> np.ndarray:
        """
        Apply x and y (translation) and heading (rotation) offsets to local coords.

        Thereby generating coords in the global frame from coords in the local frame.
        N = number of coordinates
        D = dimension

        Parameters
        ----------
        local_coords : np.ndarray
            Array of local coords, [N, D].
        x_offset : float
            x offset value to be added to all local coordinates.
        y_offset : float
            y offset value to be added to all local coordinates.
        heading_offset : float
            Heading value (in radians) to rotate all local coordinates by.

        Returns
        -------
        np.ndarray
            Resultant coordinates in the global frame.
        """
        offset_coordinates = np.array([x_offset, y_offset])
        c, s = np.cos(heading_offset), np.sin(heading_offset)
        rotation_matrix = np.array(((c, -s), (s, c)))

        rotated_coords = np.dot(rotation_matrix, local_coords.T).T
        global_coords = rotated_coords + offset_coordinates

        return global_coords

    @staticmethod
    def compute_offset_vectors(
        local_offsets: np.ndarray,
        reference_line_direction_vectors: np.ndarray,
        direction: str = "right",
    ) -> np.ndarray:
        """
        Compute offset vectors from line direction vectors & offset magnitudes.
        
        Parameters
        ----------
        local_offsets : np.ndarray
            Array of offset magnitudes (one per coordinate).
        reference_line_direction_vectors : np.ndarray
            Direction vectors of the reference line (one per coordinate, shape = [N, 2]).
        direction : str, optional
            Direction for the offset ("right" or "left").
            
        Returns
        -------
        np.ndarray
            Array of offset vectors.
        """
        if direction not in {"right", "left"}:
            raise ValueError("Unsupported direction, expected 'right' or 'left'")
        if len(local_offsets) != len(reference_line_direction_vectors):
            raise IndexError(
                f"Expected local offsets ({local_offsets.shape}) and reference line "
                f"({reference_line_direction_vectors.shape}) to be of the same length."
            )
        
        # Create a 3D vector for the z axis based on the offset direction.
        z_vector = np.array([0.0, 0.0, 1.0 if direction == "right" else -1.0])
        
        # Compute perpendicular directions as the cross product (will be 3D).
        perpendicular_directions = np.cross(reference_line_direction_vectors, z_vector)
        # Drop the z component to get back to 2D.
        perpendicular_directions = perpendicular_directions[:, :-1]
        
        # If all rows are zero (degenerate reference line), return zero offsets.
        if np.all(np.isclose(perpendicular_directions, 0)):
            return np.zeros((len(local_offsets), 2))
        
        # Replace any zero rows using adjacent nonzero values.
        perpendicular_directions = fix_zero_directions(perpendicular_directions)
        
        # Normalize each row.
        norms = np.linalg.norm(perpendicular_directions, axis=1)
        # Avoid division by zero (should not happen after the check above).
        scaled_perpendicular_directions_T = perpendicular_directions.T / norms
        offsets = (local_offsets * scaled_perpendicular_directions_T).T
        
        return offsets


    @abstractmethod
    def __call__(self, p_array: np.ndarray) -> np.ndarray:
        r"""
        Return local (u, v) coordinates from an array of parameter $p \in [0.0, 1.0]$.

        (u, v) coordinates are in their own x,y frame: start at origin, and initial
        heading is along the x axis.

        Parameters
        ----------
        p_array : np.ndarray
            p values $\in [0.0, 1.0]$ to compute parametric coordinates.

        Returns
        -------
        np.ndarray
            Array of local (u, v) coordinate pairs.
        """
        ...

    @abstractmethod
    def u_v_from_u(self, u_array: np.ndarray) -> np.ndarray:
        """
        Return local (u, v) coordinates from an array of local u coordinates.

        (u, v) coordinates are in their own x,y frame: start at origin, and initial
        heading is along the x axis.

        Parameters
        ----------
        u_array : np.ndarray
            u values from which to compute v values.

        Returns
        -------
        np.ndarray
            Array of local (u, v) coordinate pairs.
        """
        ...


class NullGeometry(Geometry):
    """Class for a "null geometry" which always returns zeros for local coords."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, p_array: np.ndarray) -> np.ndarray:
        r"""Return $(p, 0.0) \forall p \in p_array$."""
        v_array = np.zeros(len(p_array))

        local_coords = np.stack((p_array, v_array), axis=1)
        return local_coords

    def u_v_from_u(self, u_array: np.ndarray) -> np.ndarray:
        r"""Return $(u, 0.0) \forall u \in u_array$."""
        v_array = np.zeros(len(u_array))

        local_coords = np.stack((u_array, v_array), axis=1)
        return local_coords
