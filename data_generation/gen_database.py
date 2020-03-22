import pandas as pd
import numpy as np
from data_generation.simulation import gen_city, WROCLAW, brownian_2d, state_transition
from datetime import datetime, timedelta

# Columns representing events that can be registered in calendar
EVENTS_COLUMNS = [
    "dusznosc",
    "zmeczenie",
    "bol_glowy",
    "bol_miesni",
    "bol_gardla",
    "zaburzenie_wechu",
    "zaburzenie_smaku",
    "katar",
    "kichanie",
    "nudnosci",
    "biegunka",
    "bol_brzucha",
    "zawroty_glowy",
    "niepokoj",
    "kolatanie_serca",
    "zime_dreszcze",
    "zaparcia",
    "zgaga",
    "powiekszenie_wezlow_chlonnych",
    "goraczka",
    "wysypka",
    "splatanie",
    "krwioplucie"
]


def gen_users(n: int) -> pd.DataFrame:
    """
    Generates a dataframe containing users informations
    :param n: Number of users to generate

    :return: DataFrame containing each user info
    """
    names_m = pd.read_csv("names/first_m.txt", header=None).values.reshape(-1)
    names_f = pd.read_csv("names/first_f.txt", header=None).values.reshape(-1)
    last_names_m = pd.read_csv("names/last_m.txt", header=None).values.reshape(-1)
    last_names_f = pd.read_csv("names/last_f.txt", header=None).values.reshape(-1)

    users = []

    for i in range(n):
        sex = np.random.choice([0, 1])
        age = np.random.randint(1.0, 90.0)
        name = np.random.choice(names_m) if sex == 1 else np.random.choice(names_f)
        surname = np.random.choice(last_names_m) if sex == 1 else np.random.choice(last_names_f)

        users.append((i, name, surname, age, sex))

    return pd.DataFrame(users, columns=["user_id", "first_name", "last_name", "age", "sex"])


def gen_locations(users: pd.DataFrame, n_timesteps: int) -> pd.DataFrame:
    """
    Generates localization history for each user
    :param users: DataFrame containing information about each user
    :param n_timesteps: Number of timesteps to generate for each user. Timesteps delta is 1h

    :return: DataFrame containing localization information history each user
    """
    result = []

    start = gen_city(WROCLAW, len(users))
    locations = brownian_2d(start, n_timesteps, 0.001, 0.003).transpose((1, 0, 2))

    for i, (index, row) in enumerate(users.iterrows()):
        for j, location in enumerate(locations[i]):
            result.append((
                index,
                datetime.now() - timedelta(hours=j),
                location[0],
                location[1]
            ))

    return pd.DataFrame(result, columns=["user_id", "datatime", "long", "lang"])


def gen_symptoms(users: pd.DataFrame, n_days: int) -> pd.DataFrame:
    """
    Generate symptoms for each user in a markovian process

    :param users: DataFrame containing information about each user
    :param n_days: Number of calendar days with symptoms to generate for each user
    :return: Dataframe containing day-by-day symptoms history of users
    """
    result = []
    symptoms_start = np.random.binomial(1, 0.1, (len(users), len(EVENTS_COLUMNS)))
    transition_probs = np.random.uniform(0.001, 0.1, len(EVENTS_COLUMNS))
    symptoms = state_transition(symptoms_start, transition_probs, n_days).transpose((1, 0, 2))

    for i, (index, row) in enumerate(users.iterrows()):
        for j, symptom in enumerate(symptoms[i]):
            temperature = 36.6
            if np.random.binomial(1, 0.1) == 1:
                temperature = np.random.normal(38, 1.0)

            result.append((
                index,
                datetime.now().date() - timedelta(days=j),
                temperature,
                *(symptom.astype(np.bool))
            ))

    return pd.DataFrame(result, columns=["user_id", "date", "temperatura", *EVENTS_COLUMNS])


if __name__ == "__main__":
    user = gen_users(500)
    user.rename_axis("id").to_csv("database/users.csv")
    locations = gen_locations(user, 300).rename_axis("id").to_csv("database/locations.csv")
    gen_symptoms(user, 14).rename_axis("id").to_csv("database/symptoms.csv")
