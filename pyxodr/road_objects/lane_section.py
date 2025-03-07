from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from lxml import etree
from shapely.geometry import Polygon

from pyxodr.road_objects.lane import (
    ConnectionPosition,
    Lane,
    LaneOrientation,
    TrafficOrientation,
)
from pyxodr.utils import cached_property


class LaneSection:
    """
    Class representing a LaneSection in an OpenDRIVE file.

    Parameters
    ----------
    road_id : int
        Parent road ID.
    lane_section_ordinal : int
        Index position of this lane section along the road reference line (e.g.
        this is the ith lane section of the parent road.)
    lane_section_xml : etree._Element
        XML element corresponding to this lane section.
    lane_section_offset_line : np.ndarray
        Offset line of this lane section. This will be a sub-section of the lane
        offset line (OpenDRIVE spec 9.3) that just covers this lane section,
        rather than the whole road.
    lane_section_reference_line : np.ndarray
        Reference line of this lane section. This will be a sub-section of the road
        reference line (OpenDRIVE spec 7.1) that just covers this lane section,
        rather than the whole road.
    lane_section_z : np.ndarray
        z coordinates of the reference line of this lane section. This will be the z
        coordinates of a sub-section of the road reference line (OpenDRIVE spec 7.1)
        that just covers this lane section, rather than the whole road.
    traffic_orientation : TrafficOrientation
        The traffic orientation (right/left-hand-drive) for this lane section. See
        OpenDRIVE Spec Section 8.
    ignored_lane_types : Set[str], optional
        A set of lane types that should not be read from the OpenDRIVE file. If
        unspecified, no types are ignored.

    """

    def __init__(
        self,
        road_id: int,
        lane_section_ordinal: int,
        lane_section_xml: etree._Element,
        lane_section_offset_line: np.ndarray,
        lane_section_reference_line: np.ndarray,
        lane_section_z: np.ndarray,
        traffic_orientation: TrafficOrientation,
        ignored_lane_types: Optional[Set[str]] = None,
    ):
        self.road_id = road_id
        self.lane_section_ordinal = lane_section_ordinal
        self.lane_section_xml = lane_section_xml
        self.lane_section_offset_line = lane_section_offset_line
        self.lane_section_reference_line = lane_section_reference_line
        self.lane_section_z = lane_section_z
        self.traffic_orientation = traffic_orientation

        self.ignored_lane_types = (
            set([]) if ignored_lane_types is None else ignored_lane_types
        )

        self.successor_data: Tuple[LaneSection, ConnectionPosition] = (None, None)
        self.predecessor_data: Tuple[LaneSection, ConnectionPosition] = (None, None)

    def __hash__(self):
        return hash((self.road_id, self.lane_section_ordinal))

    def __repr__(self):
        return f"Section_{self.lane_section_ordinal}/Road_{self.road_id}"

    def __get_lanes_by_orientation(
        self, orientation: LaneOrientation, ignored_lanes: bool = False
    ) -> List[Lane]:
        str_orientation = "left" if orientation is LaneOrientation.LEFT else "right"
        lane_xmls = sorted(
            self.lane_section_xml.findall(f"{str_orientation}/lane"),
            key=lambda lane_xml: abs(int(lane_xml.attrib["id"])),
        )
        lanes = []
        inner_lane: Lane = None
        for lane_xml in lane_xmls:
            lane_obj = Lane(
                self.road_id,
                self.lane_section_ordinal,
                lane_xml,
                self.lane_section_offset_line,
                self.lane_section_reference_line,
                orientation,
                self.traffic_orientation,
                self.lane_section_z,
                inner_lane=inner_lane,
            )
            if ignored_lanes:
                if lane_obj.type in self.ignored_lane_types:
                    lanes.append(lane_obj)
            else:
                if lane_obj.type not in self.ignored_lane_types:
                    lanes.append(lane_obj)
            inner_lane = lane_obj

        return lanes

    def get_offset_line(self, get_z: bool = True) -> np.ndarray:
        """
        Return the offset line coordinates of this lane section.

        Parameters
        ----------
        get_z : bool, optional
            If true, return z data in the 3rd column of the returned coordinate array,
            by default True

        Returns
        -------
        np.ndarray
            Array of offset line coordinates.
        """
        offset_line = self.lane_section_offset_line[:]  # To copy
        if get_z:
            offset_line = np.append(
                offset_line, self.lane_section_z[:, np.newaxis], axis=1
            )
        return offset_line

    @cached_property
    def left_lanes(self) -> List[Lane]:
        """
        Return a list of lane objects on the left of the lane offset line.

        Ordered in -> out

        Returns
        -------
        List[Lane]
            List of left lanes
        """
        return self.__get_lanes_by_orientation(LaneOrientation.LEFT)

    @cached_property
    def right_lanes(self) -> List[Lane]:
        """
        Return a list of lane objects on the right of the lane offset line.

        Ordered in -> out

        Returns
        -------
        List[Lane]
            List of right lanes
        """
        return self.__get_lanes_by_orientation(LaneOrientation.RIGHT)

    @property
    def lanes(self) -> List[Lane]:
        """Get all lanes."""
        lanes = self.left_lanes + self.right_lanes
        return lanes

    @property
    def ignored_lane_ids(self) -> Set[int]:
        """Get IDs of lanes with types in self.ignored_lane_types."""
        ignored_lane_ids = set(
            [
                lane.id
                for lane in self.__get_lanes_by_orientation(
                    LaneOrientation.LEFT, ignored_lanes=True
                )
                + self.__get_lanes_by_orientation(
                    LaneOrientation.RIGHT, ignored_lanes=True
                )
            ]
        )
        return ignored_lane_ids

    @cached_property
    def _id_to_lane(self) -> Dict[int, Lane]:
        return {lane.id: lane for lane in self.lanes}

    def get_lane_from_id(self, lane_id: int) -> Lane:
        """Return a lane object from its int ID."""
        try:
            lane_obj = self._id_to_lane[lane_id]
        except KeyError:
            raise KeyError(
                "Error while trying to retrieve lane with id "
                + f"{lane_id} from lane section {self.lane_section_ordinal} "
                + f"in road {self.road_id}"
            )
        return lane_obj

    @cached_property
    def boundary(self) -> Polygon:
        """Return the bounding polygon of this lane section."""
        if self.left_lanes == []:
            left_border = self.lane_section_offset_line
        else:
            left_border = self.left_lanes[-1].boundary_line
        if self.right_lanes == []:
            right_border = self.lane_section_offset_line
        else:
            right_border = self.right_lanes[-1].boundary_line
        bounding_poly = Polygon(np.vstack((left_border, np.flip(right_border, axis=0))))

        return bounding_poly

    def _link_lanes(self):
        """
        Connect all lane objects within this lane section with their neighbours.
        """
        (predecessor_lane_section, predecessor_connection_position) = self.predecessor_data
        successor_lane_section, successor_connection_position = self.successor_data

        if predecessor_lane_section is not None:
            for lane in self.lanes:
                for predecessor_id in lane.predecessor_ids:
                    # Check if the predecessor id exists in the predecessor lane section
                    if predecessor_id not in predecessor_lane_section._id_to_lane:
                        # Optionally, check if it is among ignored lanes and skip silently.
                        if predecessor_id in predecessor_lane_section.ignored_lane_ids:
                            continue
                        # Otherwise log a warning and skip linking for this predecessor
                        print(
                            f"Warning: Lane id {predecessor_id} not found in predecessor lane section "
                            f"of road {self.road_id}. Skipping linking for lane {lane.id}."
                        )
                        continue
                    # If found, link the lane as usual.
                    predecessor_lane_obj = predecessor_lane_section.get_lane_from_id(predecessor_id)
                    lane.predecessor_data.append((predecessor_lane_obj, predecessor_connection_position))
                    # Also duplicate this information in the other lane to ensure symmetry.
                    if predecessor_connection_position is ConnectionPosition.BEGINNING:
                        predecessor_lane_obj.predecessor_data.append((lane, ConnectionPosition.BEGINNING))
                    else:
                        predecessor_lane_obj.successor_data.append((lane, ConnectionPosition.BEGINNING))

        if successor_lane_section is not None:
            for lane in self.lanes:
                for successor_id in lane.successor_ids:
                    if successor_id not in successor_lane_section._id_to_lane:
                        if successor_id in successor_lane_section.ignored_lane_ids:
                            continue
                        print(
                            f"Warning: Lane id {successor_id} not found in successor lane section "
                            f"of road {self.road_id}. Skipping linking for lane {lane.id}."
                        )
                        continue
                    successor_lane_obj = successor_lane_section.get_lane_from_id(successor_id)
                    lane.successor_data.append((successor_lane_obj, successor_connection_position))
                    if successor_connection_position is ConnectionPosition.BEGINNING:
                        successor_lane_obj.predecessor_data.append((lane, ConnectionPosition.END))
                    else:
                        successor_lane_obj.successor_data.append((lane, ConnectionPosition.END))

