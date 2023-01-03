import random
import os
import numpy as np
from subprocess import check_output
from enum import Enum
import matplotlib.pyplot as plt


statistics_file_name = "general_statistics.txt"
trips_statistics_file_name = "trips_statistics.xml"
emissions_file_name = "emissions.xml"
pso_statistics_file_name="pso_statistics.txt"


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


def run_simulation(options):
	command = f"sumo.exe --configuration-file=osm.sumocfg {options}"
	check_output(command, shell=True)


def load_average_speed():
	global statistics_file_name
	statistics_file = open(statistics_file_name, "r")
	for line in statistics_file:
		if line.count("Speed:") == 1:
			avg_speed = line.strip().split(" ")[1]
			return float(avg_speed)


class VehicleStatistic:
	def __init__(self, time_steps_count, searched_characteristic_value_sum):
	 	self.time_steps_count = time_steps_count
	 	self.searched_characteristic_value_sum = searched_characteristic_value_sum


class Metric(Enum):
	MAX_AVERAGE_SPEED = "Max average speed"
	MIN_FUEL_CONSUMPTION = "Min fuel consumption"


def calculate_statistics(drivers_of_different_type, trips_for_one_driver_type, file_name, characteristic_mark, metric):
	vehicles = {}
	for i in range(drivers_of_different_type * trips_for_one_driver_type):
		vehicles[i] = VehicleStatistic(0, 0.0)

	statistic_file = open(file_name, "r")
	for line in statistic_file:
		if line.count(characteristic_mark) == 1:
			parameters = line.strip().split(" ")
			vehicle_id = int(parameters[1].split("veh")[1].strip("\""))
			if vehicle_id < drivers_of_different_type * trips_for_one_driver_type:
				vehicles[vehicle_id].time_steps_count += 1
				if metric == Metric.MAX_AVERAGE_SPEED:
					vehicles[vehicle_id].searched_characteristic_value_sum += float(parameters[12].split("=")[1].strip("\"")) / float(parameters[11].split("=")[1].strip("\""))
				elif metric == Metric.MIN_FUEL_CONSUMPTION:
					vehicles[vehicle_id].searched_characteristic_value_sum += float(parameters[8].split("=")[1].strip("\""))

	average_driver_type_statistics = []
	for i in range(drivers_of_different_type):
		vehicles_of_same_type_sum = 0
		vehicles_of_same_type_sucessful_trips = 0
		for j in range(trips_for_one_driver_type):
			vehicle = vehicles[i * trips_for_one_driver_type + j]
			if vehicle.time_steps_count > 0:
				vehicles_of_same_type_sucessful_trips += 1
				average_vehicle_statistics = vehicle.searched_characteristic_value_sum / vehicle.time_steps_count
				vehicles_of_same_type_sum += average_vehicle_statistics
		average_driver_type_statistics.append(vehicles_of_same_type_sum / vehicles_of_same_type_sucessful_trips)

	return average_driver_type_statistics


def fitness_many_simulations_for_generation(drivers_of_different_types, drivers_characteristics, metric):
	global statistics_file_name
	global emissions_file_name
	trips_for_one_driver_type = 300
	results = []

	trips = load_trips(trips_for_one_driver_type)

	for i in range (0, drivers_of_different_types):
		driver = generate_driver(i, drivers_characteristics[i])
		assign_driver_to_trips(i, driver, trips)
		set_trips_in_simulation(f"driver_{i}.trips.xml")

		if metric == Metric.MAX_AVERAGE_SPEED:
			run_simulation(f"--message-log={statistics_file_name}")
			results.append(load_average_speed())
		elif metric == Metric.MIN_FUEL_CONSUMPTION:
			run_simulation(f"--emission-output={emissions_file_name}")
			results.append(calculate_statistics(1, trips_for_one_driver_type, emissions_file_name, "fuel=", Metric.MIN_FUEL_CONSUMPTION)[0])
	return results


def fitness_one_simulation_for_generation(drivers_of_different_types, drivers_characteristics, metric):
	global trips_statistics_file_name
	global emissions_file_name
	trips_for_one_driver_type = 50
	drivers = []

	trips = load_trips(drivers_of_different_types * trips_for_one_driver_type)

	for i in range (0, drivers_of_different_types):
		drivers.append(generate_driver(i, drivers_characteristics[i]))

	assign_drivers_to_trips(drivers_of_different_types, drivers, trips)
	set_trips_in_simulation("trips.xml")

	if metric == Metric.MAX_AVERAGE_SPEED:
		run_simulation(f"--tripinfo-output={trips_statistics_file_name}")
		return calculate_statistics(drivers_of_different_types, trips_for_one_driver_type, trips_statistics_file_name, "<tripinfo ", Metric.MAX_AVERAGE_SPEED)
	elif metric == Metric.MIN_FUEL_CONSUMPTION:
		run_simulation(f"--emission-output={emissions_file_name}")
		return calculate_statistics(drivers_of_different_types, trips_for_one_driver_type, emissions_file_name, "fuel=", Metric.MIN_FUEL_CONSUMPTION)


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


def pso(population, generation, metric):
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
	pbest_fitness = fitness_function(population, particles, metric)
	print(f"FITNESS: {pbest_fitness}")
	# Index of the best particle
	gbest_index = numpy_optimization_type(pbest_fitness)
	# Global best particle position
	gbest_position = pbest_position[gbest_index]
	# Velocity (starting from 0 speed)
	velocity = [[0.0 for j in range(dimension)] for i in range(population)]
	# save statistics
	pso_statistics = open(pso_statistics_file_name, "w")

	# Loop for the number of generation
	for t in range(generation):
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
		pbest_fitness = fitness_function(population, particles, metric)
		print(f"FITNESS: {pbest_fitness}")
		# Find the index of the best particle
		gbest_index = numpy_optimization_type(pbest_fitness)
		# Update the position of the best particle
		gbest_position = pbest_position[gbest_index]
		statistics = f'Generation: {t}:  Best position: {gbest_position},  Best fitness: {optimization_type(pbest_fitness)},  Average fitness: {np.average(pbest_fitness)}\n\n'
		print(statistics)
		pso_statistics.write(statistics)

	# Print the results
	statistics = f'\nNumber of generations: {t+1}\n Best position: {gbest_position}\n Fitness value: {pbest_fitness}\n Best fitness value: {optimization_type(pbest_fitness)}\n Average fitness value: {np.average(pbest_fitness)}'
	print(statistics)
	pso_statistics.write(statistics)


def draw_fitness_evolution(generation):
	global pso_statistics_file_name
	statistics_file = open(pso_statistics_file_name, "r")
	best_fitness = []
	average_fitness = []
	for line in statistics_file:
		line_parts = line.split("Best fitness:")
		if len(line_parts) > 1:
			best_fitness.append(float(line_parts[1].split(" ")[1].strip(",")))
			average_fitness.append(float(line_parts[1].split(" ")[5].strip("\n")))

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.plot(np.arange(generation), best_fitness, label="Best fitness in generation")
	ax.plot(np.arange(generation), average_fitness, label="Average fitness in generation")
	ax.set_title(f"Evolution of fitness function: {metric.value}")
	ax.set_xlabel("Generations")
	ax.set_ylabel("Fitness function value")
	legend = ax.legend()
	plt.show()	


fitness_function = fitness_one_simulation_for_generation  # [fitness_many_simulations_for_generation, fitness_one_simulation_for_generation]
metric = Metric.MIN_FUEL_CONSUMPTION  # [Metric.MAX_AVERAGE_SPEED, Metric.MIN_FUEL_CONSUMPTION]
population = 20
generation = 100

pso(population, generation, metric)
draw_fitness_evolution(generation)
