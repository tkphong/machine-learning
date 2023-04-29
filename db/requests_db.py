import numpy as np
import requests


def gen_dummy_temp(lower_bound=29, upper_bound=35, k=192):
    return np.random.choice(upper_bound - lower_bound, k) + np.array([lower_bound] * k)


def get_temp():
    result = requests.get(
        "https://mdp-weather-api.herokuapp.com/api/getRecords/"
    )
    temp_list = [record["temperature"] for record in result.json()]
    return temp_list
