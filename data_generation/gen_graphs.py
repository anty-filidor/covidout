import pandas as pd
from data_generation.simulation import gen_weighted_connections, save_edges
from datetime import datetime, timedelta, date
from data_generation.gen_database import gen_users, gen_locations, gen_symptoms, EVENTS_COLUMNS

users = gen_users(1000)
locations = gen_locations(users, 1)
symptoms = gen_symptoms(users, 13)

CORONAVIRUS_LENGTH_DAYS = 14


def extract_locations(locations_df: pd.DataFrame, timestamp: datetime):
    """
    Extracts locations in form of (node_id, longitude, latitude)
    :param locations_df:
    :param timestamp:
    :return:
    """
    return locations_df.loc[
        locations_df.join(locations_df["datatime"].sub(timestamp).abs(), lsuffix="_delta")
                 .groupby("user_id")["datatime_delta"].idxmin()
    ][["user_id", "long", "lang"]]


def calculate_symptoms(symptoms_df: pd.DataFrame, date: date):
    # Filter out non important data
    last_important = date - timedelta(days=CORONAVIRUS_LENGTH_DAYS)

    users = symptoms_df['user_id'].unique()

    symptoms_df = symptoms_df[symptoms_df["date"] > last_important]
    symptoms_df = symptoms_df.groupby("user_id").mean()

    # add missing entries
    for user in users:
        if user not in symptoms_df.index.values:
            symptoms_df.append({'user_id': user, "temperatura": 36.6, **dict(zip(EVENTS_COLUMNS, [0] * len(EVENTS_COLUMNS)))})

    return symptoms_df


if __name__ == "__main__":
    locations = extract_locations(locations, datetime.now())
    save_edges("graphs/edges.csv", gen_weighted_connections(locations))
    calculate_symptoms(symptoms, datetime.now().date()).rename_axis("node_id").to_csv("graphs/nodes.csv")