import math
import random
import json
import os

def seconds_to_hms(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{int(hours)} Jam, {int(minutes)} Menit, {int(seconds)} Detik'

def sin_angle(x):
    radian = x * (math.pi / 180)
    result = math.sin(radian)
    return result

def cos_angle(x):
    radian = x * (math.pi / 180)
    result = math.cos(radian)
    return result

def iae(error):
    return abs(error)

def ise(error):
    return error * error

def itae(error, time):
    return abs(error) * time

def itse(error, time):
    
    return error * error * time

def dropna(data):
    return [value for value in data if value is not None and not (isinstance(value, float) and math.isnan(value))]

def clip(arr, min_value, max_value):
    clipped_arr = []
    for value in arr:
        clipped_value = max(min(value, max_value), min_value)
        clipped_arr.append(clipped_value)
    return clipped_arr

def random_uniform(low, high, size=None):
    if size is None:
        return random.uniform(low, high)
    else:
        return [random.uniform(low, high) for _ in range(size)]
    
def lerp(a, b, t):
    return a + t * (b - a)

def save_results(cars_history, best_car_params, fitness_arr, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    fitness_history_path = f'{output_folder}/fitness_history.json'
    with open(fitness_history_path, 'w') as fitness_file:
        json.dump(fitness_arr, fitness_file)

    cars_history_path = f'{output_folder}/cars_history.json'
    with open(cars_history_path, 'w') as cars_file:
        json.dump(cars_history, cars_file)

    best_car_params_path = f'{output_folder}/best_car_params.json'
    with open(best_car_params_path, 'w') as best_params_file:
        json.dump(best_car_params, best_params_file)

    print('Results saved successfully.')

