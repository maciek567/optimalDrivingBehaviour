import random
import os
from subprocess import check_output
input_file = open("osm.passenger.trips-copy.xml", "r")
line_with_trip = "<trip"

trips_counter = 0
trips = []

for line in input_file:
	if line.count(line_with_trip) == 1:
		trips_counter += 1
		trips.append(line)


output_file = open("generated.trips.xml", "w")
header = """<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n"""
footer = """</routes>\n"""
output_file.write(header)

for i in range(trips_counter):
	acceleration = random.uniform(0.1, 3.0)
	deceleration = random.uniform(1.0, 6.0)
	min_gap = random.uniform(0.2, 5.0)
	desired_max_speed = random.randint(20, 150)

	v_type = f"	<vType id=\"veh_{i}\" vClass=\"passenger\" carFollowModel=\"IDM\" accel=\"{acceleration}\" decel=\"{deceleration}\" minGap=\"{min_gap}\"" \
			f" desiredMaxSpeed=\"{desired_max_speed}\" />"
	trip = trips[i]
	trip = trip.replace(f"type=\"veh_passenger\"", f"type=\"veh_{i}\"")
	output_file.write(v_type)
	output_file.write(trip)

output_file.write(footer)

command = "sumo.exe --configuration-file=osm.sumocfg  --message-log=general_statistics.txt --tripinfo-output=trips_statistics.xml"

#stream = os.popen(command)
#output = stream.read()
#os.system(command)

#check_output(command, shell=True)