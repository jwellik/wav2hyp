from obspy.geodetics import degrees2kilometers, kilometers2degrees, gps2dist_azimuth
import math
from typing import Union, Tuple, Optional

# TODO FILTER(LAT, LON)
# TODO GEORECT, GEOCIRC
# ? TODO GET_ELEV_DATA()
# TODO PLOT_TO_MAP()
# TODO GET_BBOX('bltr', 'lrbt', ...)


class GeoArea:
    """
    A class for handling geographic areas with various creation methods and property access.
    Designed for volcano seismology and earthquake catalog filtering.
    """

    def __init__(self, *args, **kwargs):
        """
        Create a GeoArea object with multiple initialization options:

        1. GeoArea(lat, lon, rad_km) - Circle converted to square bounding box
        2. GeoArea(minlat, minlon, maxlat, maxlon) - Direct bounding box
        3. GeoArea(center_lat=lat, center_lon=lon, width_km=w, height_km=h) - Rectangular area
        4. GeoArea(bbox=(minlat, minlon, maxlat, maxlon)) - Bounding box from tuple
        """

        if len(args) == 3:
            # GeoArea(lat, lon, rad_km) - circle to square
            self._init_from_center_radius(*args)
        elif len(args) == 4:
            # GeoArea(minlat, minlon, maxlat, maxlon)
            self._init_from_bbox(*args)
        elif 'bbox' in kwargs:
            # GeoArea(bbox=(minlat, minlon, maxlat, maxlon))
            self._init_from_bbox(*kwargs['bbox'])
        elif all(k in kwargs for k in ['center_lat', 'center_lon', 'width_km', 'height_km']):
            # GeoArea(center_lat=lat, center_lon=lon, width_km=w, height_km=h)
            self._init_from_center_dimensions(**kwargs)
        else:
            raise ValueError("Invalid arguments for GeoArea initialization")

    def _init_from_center_radius(self, lat: float, lon: float, rad_km: float):
        """Initialize from center point and radius (creates square bounding box)"""
        # Convert radius to degrees for a square bounding box
        lat_delta = kilometers2degrees(rad_km)
        lon_delta = kilometers2degrees(rad_km) / math.cos(math.radians(lat))

        self._minlat = lat - lat_delta
        self._maxlat = lat + lat_delta
        self._minlon = lon - lon_delta
        self._maxlon = lon + lon_delta

        self._center_lat = lat
        self._center_lon = lon

    def _init_from_bbox(self, minlat: float, minlon: float, maxlat: float, maxlon: float):
        """Initialize from bounding box coordinates"""
        self._minlat = min(minlat, maxlat)
        self._maxlat = max(minlat, maxlat)
        self._minlon = min(minlon, maxlon)
        self._maxlon = max(minlon, maxlon)

        self._center_lat = (self._minlat + self._maxlat) / 2
        self._center_lon = (self._minlon + self._maxlon) / 2

    def _init_from_center_dimensions(self, center_lat: float, center_lon: float,
                                     width_km: float, height_km: float):
        """Initialize from center point and dimensions"""
        lat_delta = kilometers2degrees(height_km / 2)
        lon_delta = kilometers2degrees(width_km / 2) / math.cos(math.radians(center_lat))

        self._minlat = center_lat - lat_delta
        self._maxlat = center_lat + lat_delta
        self._minlon = center_lon - lon_delta
        self._maxlon = center_lon + lon_delta

        self._center_lat = center_lat
        self._center_lon = center_lon

    # Corner coordinates (lower-left, upper-right style)
    @property
    def lllat(self) -> float:
        """Lower-left latitude"""
        return self._minlat

    @property
    def lllon(self) -> float:
        """Lower-left longitude"""
        return self._minlon

    @property
    def urlat(self) -> float:
        """Upper-right latitude"""
        return self._maxlat

    @property
    def urlon(self) -> float:
        """Upper-right longitude"""
        return self._maxlon

    # Min/max coordinates
    @property
    def latmin(self) -> float:
        """Minimum latitude"""
        return self._minlat

    @property
    def latmax(self) -> float:
        """Maximum latitude"""
        return self._maxlat

    @property
    def lonmin(self) -> float:
        """Minimum longitude"""
        return self._minlon

    @property
    def lonmax(self) -> float:
        """Maximum longitude"""
        return self._maxlon

    # Directional coordinates (upper, lower, left, right)
    @property
    def ulat(self) -> float:
        """Upper latitude"""
        return self._maxlat

    @property
    def llat(self) -> float:
        """Lower latitude"""
        return self._minlat

    @property
    def llon(self) -> float:
        """Left longitude"""
        return self._minlon

    @property
    def rlon(self) -> float:
        """Right longitude"""
        return self._maxlon

    # Center coordinates
    @property
    def center_lat(self) -> float:
        """Center latitude"""
        return self._center_lat

    @property
    def center_lon(self) -> float:
        """Center longitude"""
        return self._center_lon

    @property
    def center(self) -> Tuple[float, float]:
        """Center coordinates as (lat, lon)"""
        return (self._center_lat, self._center_lon)

    # Bounding box
    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        """Bounding box as (minlat, minlon, maxlat, maxlon)"""
        return (self._minlat, self._minlon, self._maxlat, self._maxlon)

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Alias for bounding_box"""
        return self.bounding_box

    @property
    def lat_range(self) -> Tuple[float, float]:
        """Latitude range as (min, max)"""
        return (self._minlat, self._maxlat)

    @property
    def lon_range(self) -> Tuple[float, float]:
        """Longitude range as (min, max)"""
        return (self._minlon, self._maxlon)

    # Dimensions
    @property
    def width_km(self) -> float:
        """Width in kilometers"""
        # Calculate at center latitude for better accuracy
        dist, _, _ = gps2dist_azimuth(self._center_lat, self._minlon,
                                      self._center_lat, self._maxlon)
        return dist / 1000  # Convert to km

    @property
    def height_km(self) -> float:
        """Height in kilometers"""
        dist, _, _ = gps2dist_azimuth(self._minlat, self._center_lon,
                                      self._maxlat, self._center_lon)
        return dist / 1000  # Convert to km

    @property
    def area_sqkm(self) -> float:
        """Area in square kilometers (approximate for small areas)"""
        return self.width_km * self.height_km

    # Utility methods
    def contains_point(self, lat: float, lon: float) -> bool:
        """Check if a point is within the bounding box"""
        return (self._minlat <= lat <= self._maxlat and
                self._minlon <= lon <= self._maxlon)

    def expand(self, factor: float) -> 'GeoArea':
        """Return a new GeoArea expanded by the given factor"""
        new_width = self.width_km * factor
        new_height = self.height_km * factor
        return GeoArea(center_lat=self._center_lat, center_lon=self._center_lon,
                       width_km=new_width, height_km=new_height)

    def expand_km(self, km: float) -> 'GeoArea':
        """Return a new GeoArea expanded by km in all directions"""
        new_width = self.width_km + 2 * km
        new_height = self.height_km + 2 * km
        return GeoArea(center_lat=self._center_lat, center_lon=self._center_lon,
                       width_km=new_width, height_km=new_height)

    def filter_catalog(self, catalog):
        """
        Filter an ObsPy catalog to events within this geographic area.

        Parameters:
        catalog: ObsPy Catalog object

        Returns:
        ObsPy Catalog object with filtered events
        """
        from obspy import Catalog

        filtered_catalog = Catalog()
        for event in catalog:
            if event.preferred_origin():
                origin = event.preferred_origin()
                if self.contains_point(origin.latitude, origin.longitude):
                    filtered_catalog.append(event)

        return filtered_catalog

    def to_dict(self) -> dict:
        """Convert to dictionary representation"""
        return {
            'minlat': self._minlat,
            'maxlat': self._maxlat,
            'minlon': self._minlon,
            'maxlon': self._maxlon,
            'center_lat': self._center_lat,
            'center_lon': self._center_lon,
            'width_km': self.width_km,
            'height_km': self.height_km,
            'area_sqkm': self.area_sqkm
        }

    def __str__(self) -> str:
        """String representation"""
        return (f"GeoArea(center: {self._center_lat:.4f}°, {self._center_lon:.4f}°, "
                f"size: {self.width_km:.1f}×{self.height_km:.1f} km)")

    def __repr__(self) -> str:
        """Detailed representation"""
        return (f"GeoArea(bbox=({self._minlat:.4f}, {self._minlon:.4f}, "
                f"{self._maxlat:.4f}, {self._maxlon:.4f}))")


# Example usage for volcano seismology
if __name__ == "__main__":
    # Method 1: Center point with radius (creates square)
    kilauea = GeoArea(19.4069, -155.2834, 25)  # 25 km radius around Kilauea
    print("Kilauea area:", kilauea)
    print(f"Bounding box: {kilauea.bounding_box}")
    print(f"Area: {kilauea.area_sqkm:.1f} sq km")

    # Method 2: Direct bounding box
    yellowstone = GeoArea(44.0, -111.0, 45.0, -110.0)
    print("\nYellowstone area:", yellowstone)

    # Method 3: Center with specific dimensions
    cascades = GeoArea(center_lat=46.2, center_lon=-122.18,
                       width_km=100, height_km=150)
    print(f"\nCascades area: {cascades}")
    print(f"Corners: LL({cascades.lllat:.3f}, {cascades.lllon:.3f}) "
          f"UR({cascades.urlat:.3f}, {cascades.urlon:.3f})")

    # Method 4: From tuple
    hawaii = GeoArea(bbox=(18.5, -160.5, 22.5, -154.5))
    print(f"\nHawaii area dimensions: {hawaii.width_km:.1f} × {hawaii.height_km:.1f} km")

    # Expand area for regional study
    regional_kilauea = kilauea.expand(2.0)  # Double the size
    print(f"\nExpanded Kilauea: {regional_kilauea.area_sqkm:.1f} sq km")

    # Example with catalog filtering (pseudo-code)
    # filtered_events = kilauea.filter_catalog(my_catalog)