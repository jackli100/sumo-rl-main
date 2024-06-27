import os 
import sys

# Check if SUMO_HOME is set and append SUMO tools to the system path
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please set the SUMO_HOME environment variable")

import sumolib
from sumolib import net, geomhelper, xml

# Read the network
net = sumolib.net.readNet(r"D:\trg1vr\sumo-rl-main\sumo-rl-main\sumo_rl\nets\2way-single-intersection\single-intersection-2.net.xml", withPedestrianConnections=False)


# Example coordinates for conversion (replace with actual values)
lon, lat = 8.541694, 47.376887

# Network coordinates (lower left network corner is at x=0, y=0)
x, y = net.convertLonLat2XY(lon, lat)
lon, lat = net.convertXY2LonLat(x, y)
print(f"Network Coordinates: x={x}, y={y}")
print(f"Converted Back: lon={lon}, lat={lat}")

# Raw UTM coordinates
x_utm, y_utm = net.convertLonLat2XY(lon, lat, True)
lon_utm, lat_utm = net.convertXY2LonLat(x_utm, y_utm, True)
print(f"UTM Coordinates: x={x_utm}, y={y_utm}")
print(f"Converted Back (UTM): lon={lon_utm}, lat={lat_utm}")

# Example lane ID and position (replace with actual values)
laneID = "lane_0"
lanePos = 50.0

# Lane/offset coordinates
# From lane position to network coordinates
x_lane, y_lane = geomhelper.positionAtShapeOffset(net.getLane(laneID).getShape(), lanePos)
print(f"Lane Coordinates: x={x_lane}, y={y_lane}")

# From network coordinates to lane position
radius = 1.0  # Example radius for finding neighboring lanes
lane, d = net.getNeighboringLanes(x_lane, y_lane, radius)[0]
lanePos_converted, dist = geomhelper.polygonOffsetAndDistanceToPoint((x_lane, y_lane), lane.getShape())
print(f"Lane Position: {lanePos_converted}, Distance: {dist}")
