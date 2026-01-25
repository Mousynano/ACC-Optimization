def update_step_response_data(error, car):
    if isinstance(car, dict):
        step_response_result = car['step_response_result']
        time = car['time']
    elif isinstance(car, Car):
        step_response_result = car.step_response_result
        time = car.time
    rise_time = find_rise_time(time, error)
    settling_time = find_settling_time(time, error)

    step_response_result['riseTime'].append(rise_time)
    step_response_result['settlingTime'].append(settling_time)

    overshoot = find_overshoot(error)
    step_response_result['overshoot'].append(overshoot)
    step_response_result['overshootPercentage'].append(
        (overshoot / error[1]) * 100) if error[1] > 0 else 0
    return step_response_result

def find_rise_time(time, error, start_percentile=0.9, end_percentile=0.1):
    start_index = next(
        (i for i, x in enumerate(error) if x <= error[1] * start_percentile),
        None)
    if start_index is None:
        return 0

    end_index = next((i for i, e in enumerate(error)
                      if e >= error[1] * end_percentile and i > start_index),
                     None)
    if end_index is not None and end_index > start_index:
        start_value = time[start_index]
        end_value = time[end_index]
        result = end_value - start_value
        return result
    else:
        return 0

def find_settling_time(time, error):
    error_arr = [abs(e) for e in error]
    settled_index = None
    max_error = max(error_arr)
    if max_error == 0:
        return 0

    for i in range(1, len(error_arr)):
        settled_index = i if error_arr[i] / max_error * 100 >= 0.02 else -1
    return time[settled_index] - time[1]

def find_overshoot(error):
    errorArr = [abs(e) for e in error]
    valleys = []
    overshoot = None

    for i in range(1, len(errorArr)):
        if len(valleys) == 2 or i == len(errorArr) - 1:
            break
        elif (errorArr[i] < errorArr[i - 1] and errorArr[i] < errorArr[i + 1]):
            if i + 2 < len(errorArr):
                if (errorArr[i] < errorArr[i - 1] and errorArr[i] < errorArr[i + 2]):
                    valleys.append(i)
                    continue
            valleys.append(i)
            continue

    if len(valleys) == 2:
        overshoot = max(errorArr[valleys[0]:valleys[1] + 1])

    if overshoot is None:
        overshoot = 0
    return overshoot
