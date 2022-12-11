import random
import os
import numpy as np
from subprocess import check_output


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


def set_trips_in_simulation(driver_number):
	sumo_file_name = "osm.sumocfg"
	sumo_temp_file_name = "osm2.sumocfg"
	with open(sumo_file_name, "r") as sumo_file:
		line_with_trips = "<route-files"
		new_settings = ""

		for line in sumo_file:
			if line.count(line_with_trips) == 1:
				new_settings += f"        <route-files value=\"driver_{driver_number}.trips.xml\"/>\n"
			else:
				new_settings += line

	with open(sumo_temp_file_name, "w") as new_sumo_file:
		new_sumo_file.write(new_settings)

	os.remove(sumo_file_name)
	os.rename(sumo_temp_file_name, sumo_file_name)


def run_simulation(statistics_file_name):
	command = f"sumo.exe --configuration-file=osm.sumocfg  --message-log={statistics_file_name} --tripinfo-output=trips_statistics.xml"
	check_output(command, shell=True)


def load_average_speed(statistics_file_name):
	statistics_file = open(statistics_file_name, "r")
	for line in statistics_file:
		if line.count("Speed:") == 1:
			avg_speed = line.strip().split(" ")[1]
			return float(avg_speed)


def fitness_function(drivers_of_different_types, drivers_characteristics):
	trips_for_one_driver_type = 300
	statistics_file_name = "general_statistics.txt"
	results = []

	trips = load_trips(trips_for_one_driver_type)

	for i in range (0, drivers_of_different_types):
		driver = generate_driver(i, drivers_characteristics[i])
		assign_driver_to_trips(i, driver, trips)
		set_trips_in_simulation(i)

		run_simulation(statistics_file_name)

		results.append(load_average_speed(statistics_file_name))

	return results


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


def pso(population, dimension, generation, fitness_criterion):
	# Initialisation
	# particles boundary
	boundaries = [Boundary(0.1, 3.0), Boundary(0.2, 5.0), Boundary(0.2, 5.0), Boundary(10, 50)]
	# Population
	particles = [get_random_driver_characteristics(boundaries) for i in range(population)]
	# Particle's best position
	pbest_position = particles
	# Fitness
	pbest_fitness = [fitness_function(population, particles)]
	# Index of the best particle
	gbest_index = np.argmax(pbest_fitness)
	# Global best particle position
	gbest_position = pbest_position[gbest_index]
	# Velocity (starting from 0 speed)
	velocity = [[0.0 for j in range(dimension)] for i in range(population)]
	# save statistics
	pso_statistics = open("pso_statistics.txt", "w")

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
		pbest_fitness = [fitness_function(population, particles)]
		# Find the index of the best particle
		gbest_index = np.argmax(pbest_fitness)
		# Update the position of the best particle
		gbest_position = pbest_position[gbest_index]
		statistics = f'Generation: {t}:  Best position: {gbest_position},  Best fitness: {max(pbest_fitness)},  Average best fitness: {np.average(pbest_fitness)}\n'
		print(statistics)
		pso_statistics.write(statistics)

	# Print the results
	statistics = f'\nNumber of generations: {t}\n Global best position: {gbest_position}\n Best fitness value: {max(pbest_fitness)}\n Average particle best fitness value: {np.average(pbest_fitness)}'
	print(statistics)
	pso_statistics.write(statistics)


population = 20
dimension = 4
generation = 100
fitness_criterion = 10e-2

pso(population, dimension, generation, fitness_criterion)