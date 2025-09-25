from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy.geodetics import kilometers2degrees as km2d


def download_sthelens_inventory():

    # Get inventory for the entire time period
    print("Getting inventory...")

    # Target of Interest
    volcano = "St. Helens"
    lat, lon, elev = 46.2, -122.18, 2549.
    zlim = (-5, 50)

    # Waveform Client
    client = Client("IRIS")

    # Inventory - Radius search
    # Time of Interest - will be set from command line arguments
    t1 = UTCDateTime("2021/06/29")
    t2 = UTCDateTime("2021/06/30")
    inv = client.get_stations(network="UW,CC", latitude=lat, longitude=lon, maxradius=km2d(50.0), starttime=t1, endtime=t2, level="channel")
    inv = inv.select(channel="[EBH]H[ZNE]")  # select normal seismic channel components
    print(inv)
    inv.write("../../examples/sthelens_inventory.xml", format="STATIONXML")
    print()

    # Inventory - REDPy config
    # REDPy Stations: SEP,YEL,HSR,SHW,EDM,STD,JUN,SOS
    # Time of Interest - will be set from command line arguments
    t1 = UTCDateTime("2004/09/20")
    t2 = UTCDateTime("2004/10/01")
    inv = client.get_stations(network="UW", station="SEP,YEL,HSR,SHW,EDM,STD,JUN,SOS", channel="EHZ", latitude=lat, longitude=lon, maxradius=km2d(50.0), starttime=t1, endtime=t2, level="channel")
    print(inv)
    inv.write("../../examples/stehelns_inventory_redpy.xml", format="STATIONXML")


def download_spurr_inventory():

    # Get inventory for the entire time period
    print("Getting inventory...")

    # Target of Interest
    volcano = "Spurr"
    lat, lon, elev = 61.299, -152.251, 3374.
    zlim = (-5, 50)

    # Time of Interest - will be set from command line arguments
    t1 = UTCDateTime("2024/10/14")
    t2 = UTCDateTime("2024/10/15")

    # Waveform Client
    client = Client("IRIS")
    inv = client.get_stations(network="AV,AK", latitude=lat, longitude=lon, maxradius=km2d(15.0), starttime=t1, endtime=t2, level="channel")
    inv = inv.select(channel="[EBH]H[ZNE]")  # select normal seismic channel components

    print(inv)
    inv.write("../results/spurr/spurr_inventory.xml", format="STATIONXML")


def prepare_awu_inventory():
    # Convert Awu station csv to ObsPy Inventory

    #
    #
    #

    # inv.write("../results/awu/awu_inventory.xml", format="STATIONXML")
    pass


if __name__ == "__main__":
    download_sthelens_inventory()
    # download_spurr_inventory()