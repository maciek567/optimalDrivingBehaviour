import random
import os
import sys
import numpy as np
import re
import matplotlib.pyplot as plt
import traci
import statistics as stat
from subprocess import check_output
from enum import Enum


statistics_file_name = "general_statistics.txt"
trips_statistics_file_name = "trips_statistics.xml"
emissions_file_name = "emissions.xml"
pso_statistics_file_name="pso_statistics.txt"


#****************************************** SIMULATION PREPARATION *********************************************************************************


def load_trips(trips_for_one_driver):
	input_file = open("osm.passenger.trips.xml", "r")
	line_with_trip = "<trip "
	trips = []
	trips_count = 0

	for line in input_file:
		if line.count(line_with_trip) == 1:
			trips.append(line)
			trips_count += 1
			if trips_count == trips_for_one_driver:
				return trips

	return trips


def adjust_trips(trips):
	trips_starting_instantly = []
	for trip in trips:
		trips_starting_instantly.append(re.sub("depart=\"\w+.\w+\"", "depart=\"0.00\"", trip))
	return trips_starting_instantly


def generate_driver(i, driver_characteristics):
	acceleration = driver_characteristics[0]
	deceleration = driver_characteristics[1]
	min_gap = driver_characteristics[2]
	desired_max_speed = driver_characteristics[3]

	v_type = f"	<vType id=\"driver_{i}\" vClass=\"passenger\" carFollowModel=\"IDM\" accel=\"{acceleration}\" decel=\"{deceleration}\" minGap=\"{min_gap}\"" \
			f" desiredMaxSpeed=\"{desired_max_speed}\" />\n"

	return v_type


def assign_driver_to_trips(driver_number, driver_characteristics, trips):
	output_file = open(f"driver_{driver_number}.trips.xml", "w")
	header = """<?xml version="1.0" encoding="UTF-8"?>
	<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n"""
	footer = """</routes>\n"""

	output_file.write(header)
	output_file.write(driver_characteristics)

	for i in range(len(trips)):
		trip = trips[i]
		trip = trip.replace(f"type=\"veh_passenger\"", f"type=\"driver_{driver_number}\"")
		output_file.write(trip)

	output_file.write(footer)


def assign_drivers_to_trips(different_drivers, drivers_characteristics, trips):
	output_file = open("trips.xml", "w")
	header = """<?xml version="1.0" encoding="UTF-8"?>
	<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n"""
	footer = """</routes>\n"""
	trips_for_one_driver_type = int(len(trips) / different_drivers)

	output_file.write(header)

	for i in range(different_drivers):
		output_file.write(drivers_characteristics[i])
		for j in range(trips_for_one_driver_type):
			trip = trips[i * trips_for_one_driver_type + j]
			trip = trip.replace(f"type=\"veh_passenger\"", f"type=\"driver_{i}\"")
			output_file.write(trip)

	output_file.write(footer)


def set_trips_in_simulation(file_name):
	sumo_file_name = "osm.sumocfg"
	sumo_temp_file_name = "osm2.sumocfg"
	with open(sumo_file_name, "r") as sumo_file:
		line_with_trips = "<route-files"
		new_settings = ""

		for line in sumo_file:
			if line.count(line_with_trips) == 1:
				new_settings += f"        <route-files value=\"{file_name}\"/>\n"
			else:
				new_settings += line

	with open(sumo_temp_file_name, "w") as new_sumo_file:
		new_sumo_file.write(new_settings)

	os.remove(sumo_file_name)
	os.rename(sumo_temp_file_name, sumo_file_name)


#****************************************** RUNNING SIMULATION *********************************************************************************


def verify_requirements():
	if 'SUMO_HOME' in os.environ:
		tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
		sys.path.append(tools)
	else:
		sys.exit("please declare environment variable 'SUMO_HOME'")


def get_colliders():
	collider_types = []
	if traci.simulation.getCollidingVehiclesNumber() > 0:
		collisions = list(traci.simulation.getCollisions())
		for collision in collisions:
			collider_types.append(collision.colliderType)
		print(f"Collisions cause by driver type: {collider_types}")
	return collider_types


def run_simulation(options, number_of_finished_trips_to_exit):
	sumoCmd = ["sumo.exe", "-c", "osm.sumocfg", options]
	print('Run ' + ' '.join(sumoCmd))

	traci.start(sumoCmd)

	finished_trips = 0
	vehicles_finished = []
	collider_types = []
	while True:
		traci.simulationStep()

		collider_types += get_colliders()

		finished_trips += traci.simulation.getArrivedNumber()
		vehicles_finished += traci.simulation.getArrivedIDList()
		if finished_trips > number_of_finished_trips_to_exit: 
			print(f"Finished trips: {finished_trips} in {traci.simulation.getTime()} time steps")
			traci.close()
			break

	no_duplicates_and_empty_colliders = list(dict.fromkeys(filter(None, collider_types)))
	print(f"COLLIDER TYPES TO EXCLUDE: {no_duplicates_and_empty_colliders}")
	return vehicles_finished, no_duplicates_and_empty_colliders


#****************************************** METRIC CALCULATION *********************************************************************************


def load_average_speed():
	global statistics_file_name
	statistics_file = open(statistics_file_name, "r")
	for line in statistics_file:
		if line.count("Speed:") == 1:
			avg_speed = line.strip().split(" ")[1]
			return float(avg_speed)


class DriverStatistic:
	def __init__(self, driver_type, time_steps_count, searched_characteristic_value_sum):
		self.driver_type = driver_type
		self.time_steps_count = time_steps_count
		self.searched_characteristic_value_sum = searched_characteristic_value_sum

class DriverTypeStatistic:
	def __init__(self, drivers_number, searched_characteristic_value_sum):
		self.drivers_number = drivers_number
		self.searched_characteristic_value_sum = searched_characteristic_value_sum


class Metric(Enum):
	MAX_AVERAGE_SPEED = "Max average speed"
	MIN_FUEL_CONSUMPTION = "Min fuel consumption"


def extract_value_float(string):
	return float(string.split("=")[1].strip("\""))


def extract_driver_type(string):
	driver_type = string.split("=")[1].strip("\"")
	temp = int(driver_type.split("_")[1])
	return int(driver_type.split("_")[1])


def calculate_drivers_statistics(drivers_of_different_type, trips_for_one_driver_type, file_name, characteristic_mark, metric, finished_vehicles): 
	# Common
	VEHICLE_ID = 1
	# Trips statistics file
	DURATION = 11
	ROUTE_LENGTH = 12
	VEHICLE_TYPE_TRIPS = 20
	# Emissions file
	FUEL_CONSUMPTION = 8
	VEHICLE_TYPE_EMISSIONS = 12

	vehicles_size_including_ids_gaps = int(float(drivers_of_different_type) * float(trips_for_one_driver_type) * 1.1)

	vehicles = {}
	for i in range(vehicles_size_including_ids_gaps):
		vehicles[i] = DriverStatistic("", 0, 0.0)

	finished_vehicles = [int(vehicle.strip("veh")) for vehicle in finished_vehicles]
	print(f"FINISHED VEHICLES: {finished_vehicles}")

	statistic_file = open(file_name, "r")
	for line in statistic_file:
		if line.count(characteristic_mark) == 1:
			parameters = line.strip().split(" ")
			vehicle_id = int(parameters[VEHICLE_ID].split("veh")[1].strip("\""))
			if vehicle_id in finished_vehicles:
				vehicles[vehicle_id].time_steps_count += 1
				if metric == Metric.MAX_AVERAGE_SPEED:
					vehicles[vehicle_id].driver_type = extract_driver_type(parameters[VEHICLE_TYPE_TRIPS])
					vehicles[vehicle_id].searched_characteristic_value_sum += extract_value_float(parameters[ROUTE_LENGTH]) / extract_value_float(parameters[DURATION])
				elif metric == Metric.MIN_FUEL_CONSUMPTION:
					vehicles[vehicle_id].driver_type = extract_driver_type(parameters[VEHICLE_TYPE_EMISSIONS])
					vehicles[vehicle_id].searched_characteristic_value_sum += extract_value_float(parameters[FUEL_CONSUMPTION])

	return vehicles


def calculate_drivers_types_statistics(vehicles, drivers_of_different_type):
	vehicles_types = {}
	for i in range(drivers_of_different_type):
		vehicles_types[i] = DriverTypeStatistic(0, 0.0)

	for vehicle in vehicles.values():
		if vehicle.time_steps_count > 0:
			vehicles_types[vehicle.driver_type].drivers_number += 1
			vehicles_types[vehicle.driver_type].searched_characteristic_value_sum += vehicle.searched_characteristic_value_sum / vehicle.time_steps_count

	#for id, vehicle in vehicles.items():
	#	print(f"{id}: {vehicle.driver_type} {vehicle.time_steps_count} {vehicle.searched_characteristic_value_sum}")
	print("Driver type: drivers that finished trip | searched characteristic value sum")
	for id, vehicle in vehicles_types.items():
		print(f"{id}: {vehicle.drivers_number} | {vehicle.searched_characteristic_value_sum}")

	average_driver_type_statistics = []
	for driver_type in vehicles_types.values():
		if driver_type.drivers_number != 0:
			average_driver_type_statistics.append(driver_type.searched_characteristic_value_sum / driver_type.drivers_number)
		else:
			average_driver_type_statistics.append(0.0)

	return average_driver_type_statistics


def exclude_colliders(average_driver_type_statistics, colliders_to_exclude):
	for collider in colliders_to_exclude:
		id = int(collider.split("_")[1])
		average_driver_type_statistics[id] = 0.0
	average_driver_type_statistics = [i for i in average_driver_type_statistics if i != 0.0]

	return average_driver_type_statistics


def calculate_statistics(drivers_of_different_type, trips_for_one_driver_type, file_name, characteristic_mark, metric, finished_vehicles, colliders_to_exclude):
	vehicles = calculate_drivers_statistics(drivers_of_different_type, trips_for_one_driver_type, file_name, characteristic_mark, metric, finished_vehicles)
	average_driver_type_statistics = calculate_drivers_types_statistics(vehicles, drivers_of_different_type)
	return exclude_colliders(average_driver_type_statistics, colliders_to_exclude)


#****************************************** FITNESS EVALUATION *********************************************************************************


def fitness_many_simulations_for_generation(drivers_of_different_types, drivers_characteristics, characteristic_mark, metric, drivers_of_one_type, proportion_of_finished_trips_to_exit_simulation):
	global statistics_file_name
	global emissions_file_name
	number_of_finished_trips_to_exit = drivers_of_different_types * drivers_of_one_type * proportion_of_finished_trips_to_exit_simulation
	results = []

	trips = load_trips(drivers_of_one_type)
	trips = adjust_trips(trips)

	for i in range (0, drivers_of_different_types):
		driver = generate_driver(i, drivers_characteristics[i])
		assign_driver_to_trips(i, driver, trips)
		set_trips_in_simulation(f"driver_{i}.trips.xml")

		if metric == Metric.MAX_AVERAGE_SPEED:
			run_simulation(f"--message-log={statistics_file_name}", number_of_finished_trips_to_exit)
			results.append(load_average_speed())
		elif metric == Metric.MIN_FUEL_CONSUMPTION:
			run_simulation(f"--emission-output={emissions_file_name}", number_of_finished_trips_to_exit)
			results.append(calculate_statistics(1, drivers_of_one_type, emissions_file_name, "fuel=", Metric.MIN_FUEL_CONSUMPTION)[0])
	return results


def fitness_one_simulation_for_generation(drivers_of_different_types, drivers_characteristics, metric, drivers_of_one_type, proportion_of_finished_trips_to_exit_simulation):
	global trips_statistics_file_name
	global emissions_file_name
	number_of_finished_trips_to_exit = drivers_of_different_types * drivers_of_one_type * proportion_of_finished_trips_to_exit_simulation
	drivers = []

	trips = load_trips(drivers_of_different_types * drivers_of_one_type)
	trips = adjust_trips(trips)

	for i in range (0, drivers_of_different_types):
		drivers.append(generate_driver(i, drivers_characteristics[i]))

	assign_drivers_to_trips(drivers_of_different_types, drivers, trips)
	set_trips_in_simulation("trips.xml")

	if metric == Metric.MAX_AVERAGE_SPEED:
		finished_vehicles, colliders_to_exclude = run_simulation(f"--tripinfo-output={trips_statistics_file_name}", number_of_finished_trips_to_exit)
		return calculate_statistics(drivers_of_different_types, drivers_of_one_type, trips_statistics_file_name, "<tripinfo ", Metric.MAX_AVERAGE_SPEED, finished_vehicles, colliders_to_exclude)
	elif metric == Metric.MIN_FUEL_CONSUMPTION:
		finished_vehicles, colliders_to_exclude = run_simulation(f"--emission-output={emissions_file_name}", number_of_finished_trips_to_exit)
		return calculate_statistics(drivers_of_different_types, drivers_of_one_type, emissions_file_name, "fuel=", Metric.MIN_FUEL_CONSUMPTION, finished_vehicles, colliders_to_exclude)


#****************************************** PSO OPTIMALIZATION *********************************************************************************


class Boundary:
	def __init__(self, min, max):
		self.min = min
		self.max = max


def update_velocity(particle, velocity, pbest, gbest, w_min=0.5, max=1.0, c=0.1):
	# Initialise new velocity array
	num_particle = len(particle)
	new_velocity = np.array([0.0 for i in range(num_particle)])

	# Randomly generate r1, r2 and inertia weight from normal distribution
	r1 = random.uniform(0,max)
	r2 = random.uniform(0,max)
	w = random.uniform(w_min,max)
	c1 = c
	c2 = c

	# Calculate new velocity
	for i in range(num_particle):
		new_velocity[i] = w*velocity[i] + c1*r1*(pbest[i]-particle[i])+c2*r2*(gbest[i]-particle[i])

	return new_velocity


def update_position(particle, velocity, boundaries):
	# Move particles by adding velocity
	new_particle = []
	for i in range(len(particle)):
		new_param = particle[i] + velocity[i]
		if new_param < boundaries[i].min:
			new_particle.append(boundaries[i].min)
		elif new_param > boundaries[i].max:
			new_particle.append(boundaries[i].max)
		else:
			new_particle.append(new_param)

	return new_particle


def get_random_driver_characteristics(boundaries):
	acceleration = random.uniform(boundaries[0].min, boundaries[0].max)
	deceleration = random.uniform(boundaries[1].min, boundaries[1].max)
	min_gap = random.uniform(boundaries[2].min, boundaries[2].max)
	desired_max_speed = random.randint(boundaries[3].min, boundaries[3].max)

	driver_characteristics = []
	driver_characteristics.extend([acceleration, deceleration, min_gap, desired_max_speed])

	return driver_characteristics


def pso(metric, population, drivers_of_one_type, generations, proportion_of_finished_trips_to_exit_simulation):
	# Initialisation
	global pso_statistics_file_name
	dimension = 4
	fitness_criterion = 10e-2
	numpy_optimization_type = None
	optimization_type = None
	if metric == Metric.MAX_AVERAGE_SPEED:
		numpy_optimization_type = np.argmax
		optimization_type = max
	elif metric == Metric.MIN_FUEL_CONSUMPTION:
		numpy_optimization_type = np.argmin
		optimization_type = min
	else:
		raise ValueError(f"Metric must be one of the following: {[e.value for e in Metric]}")

	# particles boundary
	boundaries = [Boundary(0.1, 3.0), Boundary(0.2, 5.0), Boundary(0.2, 5.0), Boundary(10, 50)]
	# Population
	particles = [get_random_driver_characteristics(boundaries) for i in range(population)]
	# Particle's best position
	pbest_position = particles
	# Fitness
	pbest_fitness = fitness_function(population, particles, metric, drivers_of_one_type, proportion_of_finished_trips_to_exit_simulation)
	print(f"FITNESS: {pbest_fitness}")
	# Index of the best particle
	gbest_index = numpy_optimization_type(pbest_fitness)
	# Global best particle position
	gbest_position = pbest_position[gbest_index]
	# Velocity (starting from 0 speed)
	velocity = [[0.0 for j in range(dimension)] for i in range(population)]
	# save statistics
	pso_statistics = open(pso_statistics_file_name, "w")

	# Loop for the number of generations
	for t in range(generations):
		# Stop if the average fitness value reached a predefined success criterion
		if np.average(pbest_fitness) <= fitness_criterion:
			break
		else:
		    for n in range(population):
			    # Update the velocity of each particle
			    velocity[n] = update_velocity(particles[n], velocity[n], pbest_position[n], gbest_position)
			    # Move the particles to new position
			    particles[n] = update_position(particles[n], velocity[n], boundaries)
		# Calculate the fitness value
		pbest_fitness = fitness_function(population, particles, metric, drivers_of_one_type, proportion_of_finished_trips_to_exit_simulation)
		print(f"FITNESS: {pbest_fitness}")
		# Find the index of the best particle
		gbest_index = numpy_optimization_type(pbest_fitness)
		# Update the position of the best particle
		gbest_position = pbest_position[gbest_index]
		statistics = "".join((f"Generation: {t}:\n Best_position={[round(x, 2) for x in gbest_position]} | Best_fitness={round(optimization_type(pbest_fitness), 2)}",
				f" | Average_fitness={round(np.average(pbest_fitness), 2)} | Fitness_value={[round(x, 2) for x in pbest_fitness]}\n\n"))
		print(statistics)
		pso_statistics.write(statistics)

	# Print the final results
	statistics = "".join((f"\nNumber of generations: {t+1}\n Best position: {[round(x, 2) for x in gbest_position]}\n Fitness value: {[round(x, 2) for x in pbest_fitness]}\n",
			f" Best fitness value: {round(optimization_type(pbest_fitness), 2)}\n Average fitness value: {round(np.average(pbest_fitness), 2)}"))
	print(statistics)
	pso_statistics.write(statistics)


#****************************************** RESULTS PRESENTATION *********************************************************************************


def draw_fitness_evolution(generations, drivers_types, drivers_of_one_type):
	global pso_statistics_file_name
	BEST_POSITION = 0
	BEST_FITNESS = 1
	AVERAGE_FITNESS = 2
	FITNESS = 3

	statistics_file = open(pso_statistics_file_name, "r")
	best_fitness = []
	average_fitness = []
	std_deviations = []
	for line in statistics_file:
		line_parts = line.split(" | ")
		if len(line_parts) > 1:
			best_fitness.append(float(line_parts[BEST_FITNESS].split("=")[1]))

			average_fitness.append(float(line_parts[AVERAGE_FITNESS].split("=")[1]))

			fitness_list=line_parts[FITNESS].split("=")[1]
			fitness_values = fitness_list.strip("\n").strip("[").strip("]").split(", ")
			std_deviations.append(stat.stdev([float(x) for x in fitness_values]))

	fig = plt.figure()
	ax = fig.add_subplot(1, 2, 1)
	ax.plot(np.arange(generations), best_fitness, label="Best fitness in generation")
	ax.plot(np.arange(generations), average_fitness, label="Average fitness in generation")
	ax.set_title(f"Fitness evolution: {metric.value}, {drivers_types} types, {drivers_of_one_type} for every type ")
	ax.set_xlabel("Generations")
	ax.set_ylabel("Fitness function value")
	ax2 = fig.add_subplot(1, 2, 2)
	ax2.plot(np.arange(generations), std_deviations, "g", label="Fitness std dev in generation")
	ax2.set_title(f"Evolution of fitness std dev")
	ax2.set_xlabel("Generations")
	ax2.set_ylabel("Standard deviation")
	plt.show()	


#****************************************** PARAMETERS AND OPTIMALIZATION START ************************************************************************


fitness_function = fitness_one_simulation_for_generation  # [fitness_many_simulations_for_generation, fitness_one_simulation_for_generation]
metric = Metric.MAX_AVERAGE_SPEED  # [Metric.MAX_AVERAGE_SPEED, Metric.MIN_FUEL_CONSUMPTION]
drivers_types = 15
drivers_of_one_type = 100
generations = 60
proportion_of_finished_trips_to_exit_simulation = 0.1

pso(metric, drivers_types, drivers_of_one_type, generations, proportion_of_finished_trips_to_exit_simulation)
draw_fitness_evolution(generations, drivers_types, drivers_of_one_type)

