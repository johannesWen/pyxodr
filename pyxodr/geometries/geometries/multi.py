from copy import deepcopy
from typing import List

import numpy as np

from pyxodr.geometries.base import Geometry, NullGeometry


class MultiGeom:
    """
    Class representing sequential geometry objects.

    Parameters
    ----------
    geometries : List[Geometry]
        Ordered list of geometry objects.
    distance_array : np.ndarray
        Distances along reference line at which each of the geometries begin.
    """

    def __init__(self, geometries: List[Geometry], distance_array: np.ndarray):
        self.distance_array = distance_array
        self.geometries = geometries

        if len(self.geometries) != len(self.distance_array):
            raise IndexError("Geometry and distance arrays are of different lengths.")
        elif len(self.geometries) == 0:
            raise IndexError("Geometry and distance arrays are empty.")

    def __call__(self, u_array: np.ndarray) -> np.ndarray:
        """
        Return local (u, v) coordinates from a array of parameter u.

        (u,v) coordinates are in their own x,y frame: start at origin, and initial
        heading is along x axis.
        Note that the u values will be translated to start from 0 in each case, but
        will not be scaled. This is to match use cases for e.g. z computation in the
        OpenDRIVE spec (elevationProfile)

        Parameters
        ----------
        u_array : np.ndarray
            Local u coordinates

        Returns
        -------
        np.ndarray
            Array of local (u, v) coordinate pairs
        """
        # Compute the index of the geometry to use for each s value
        # This is the last index where the distance value is less than the current s
        # value: equivalently one less than the first index where the s value is
        # exceeded
        geometry_indices = (
            np.argmax(
                np.tile(
                    np.concatenate((self.distance_array, np.array([np.inf]))),
                    (len(u_array), 1),
                ).T
                > u_array,
                axis=0,
            )
            - 1
        )

        du_values = u_array - self.distance_array[geometry_indices]

        v_values = []

        for geometry_index, geometry in enumerate(self.geometries):
            du_sub_values = du_values[geometry_indices == geometry_index]

            if len(du_sub_values) != 0:
                v_values.append(geometry.u_v_from_u(du_sub_values)[:, 1])

        local_coords = np.stack((u_array, np.concatenate(v_values)), axis=1)

        return local_coords

    def global_coords_and_offsets_from_reference_line(
        self,
        distance_line: np.ndarray,
        reference_line: np.ndarray,
        offset_line: np.ndarray,
        direction="right",
    ):
        """
        Compute global coordinates and corresponding offset values for the multi-geometry,
        given a reference line. This function partitions the reference line according to
        self.distance_array and applies each geometry's offset.

        Parameters
        ----------
        distance_line : np.ndarray
            The line used to cross-reference s values.
        reference_line : np.ndarray
            The reference line to which offsets will be applied.
        offset_line : np.ndarray
            The line representing an existing offset.
        direction : str, optional
            Direction of the offset ("right" or "left"), by default "right".

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - global_coordinates: the computed global coordinates.
            - all_local_offsets: an array of offset values matching the reference_line length.
        """
        # Check if distance_line has enough points to compute differences.
        if distance_line.shape[0] < 2:
            print("Warning: distance_line has fewer than 2 points; cannot compute direction vectors. Falling back to reference_line with zero offsets.")
            return reference_line.copy(), np.zeros(len(reference_line))

        # Compute differences along the distance_line.
        distance_line_direction_vectors = np.diff(distance_line, axis=0)
        # If no direction vectors were computed, fallback.
        if distance_line_direction_vectors.shape[0] == 0:
            print("Warning: No direction vectors computed; falling back to reference_line with zero offsets.")
            return reference_line.copy(), np.zeros(len(reference_line))
        
        # Compute cumulative distances along the distance_line.
        distance_line_distances = np.cumsum(np.linalg.norm(distance_line_direction_vectors, axis=1))
        distance_line_distances = np.insert(distance_line_distances, 0, 0)
        
        # Append the last direction vector so that the number of rows matches the distance_line.
        distance_line_direction_vectors = np.vstack(
            (distance_line_direction_vectors, distance_line_direction_vectors[-1])
        )
        # Convert to 3D by appending a zero z-component, then back to 2D.
        distance_line_direction_vectors = np.vstack(
            (distance_line_direction_vectors.T, np.zeros(len(distance_line_direction_vectors)).T)
        ).T

        # s_values marks the starting distances of each geometry segment.
        s_values = self.distance_array.copy()
        partition_indices = np.searchsorted(distance_line_distances, s_values)
        
        # Compute offsets between the offset_line and reference_line.
        existing_offsets = np.linalg.norm(offset_line - reference_line, axis=1)
        
        geometries = self.geometries

        # Adjust partition indices if necessary.
        if partition_indices[0] != 0:
            partition_indices = np.insert(partition_indices, 0, 0)
            geometries = [NullGeometry()] + geometries
            s_values = np.insert(s_values, 0, 0.0)
        if partition_indices[-1] != len(reference_line):
            partition_indices = np.append(partition_indices, len(reference_line))
            geometries = geometries + [NullGeometry()]
            s_values = np.append(s_values, s_values[-1])
        
        global_coordinates = []
        all_local_offsets = []
        start_end_indices = zip(partition_indices[:-1], partition_indices[1:])
        for (start_index, end_index), geometry, offset_distance in zip(start_end_indices, geometries, s_values):
            if start_index != end_index:  # Only process non-empty segments.
                sub_reference_line = reference_line[start_index:end_index]
                sub_reference_line_direction_vectors = distance_line_direction_vectors[start_index:end_index]
                sub_u_array = distance_line_distances[start_index:end_index]
                sub_global_offsets = existing_offsets[start_index:end_index]
                # Translate the u values so that the segment starts at zero.
                sub_u_array = sub_u_array - offset_distance
                
                # Compute local offset values (v coordinates) and add the global offset.
                local_coords = geometry.u_v_from_u(sub_u_array)[:, 1] + sub_global_offsets
                local_offsets = geometry.u_v_from_u(sub_u_array)[:, 1] + sub_global_offsets
                all_local_offsets.append(local_offsets)
                
                if len(local_coords) != 0:
                    sub_global_coordinates = geometry.compute_offset_vectors(
                        local_coords, sub_reference_line_direction_vectors, direction=direction
                    )
                    global_coordinates.append(sub_reference_line + sub_global_coordinates)
                else:
                    global_coordinates.append(deepcopy(sub_reference_line))
        
        # If no segments produced any coordinates, fall back.
        if not global_coordinates:
            print("Warning: No segments produced global coordinates; falling back to reference_line with zero offsets.")
            return reference_line.copy(), np.zeros(len(reference_line))
        
        global_coordinates = np.vstack(global_coordinates)
        all_local_offsets = np.concatenate(all_local_offsets) if all_local_offsets else np.zeros(len(reference_line))
        
        # If the total number of offset values does not match the number of reference_line points, interpolate.
        if len(all_local_offsets) != len(reference_line):
            orig_idx = np.linspace(0, 1, len(all_local_offsets))
            target_idx = np.linspace(0, 1, len(reference_line))
            all_local_offsets = np.interp(target_idx, orig_idx, all_local_offsets)
        
        assert len(all_local_offsets) == len(reference_line)
        return global_coordinates, all_local_offsets

