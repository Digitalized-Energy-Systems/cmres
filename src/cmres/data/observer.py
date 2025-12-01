step_data = {}


def gather(key, data):
    if key in step_data:
        step_data[key] += [data]
    else:
        step_data[key] = [data]


def data():
    return step_data


def clear():
    step_data.clear()
