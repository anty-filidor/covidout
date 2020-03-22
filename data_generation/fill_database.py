import json
from sqlalchemy import create_engine
import pandas as pd
from data_generation.gen_database import gen_users, gen_symptoms, gen_locations

# Number of users to generate
NUM_USERS = 100

# Number of calendar days filled by each user
NUM_SYMPTOMS_DAYS = 10

# Number of locations stamps generated for each user
NUM_LOCATIONS = 10

if __name__ == "__main__":
    with open("databases.json", "r") as f:
        DATABASES = json.load(f)

    db = DATABASES['production']

    engine_string = "postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}".format(
        user=db['USER'],
        password=db['PASSWORD'],
        host=db['HOST'],
        port=db['PORT'],
        database=db['NAME']
    )

    engine = create_engine(engine_string)

    # clear old database entries
    with engine.connect() as con:
        con.execute("DELETE FROM locations WHERE user_id < 99999;DELETE FROM symptoms WHERE user_id < 99999;DELETE FROM users WHERE user_id <99999")

    users: pd.DataFrame = gen_users(NUM_USERS)
    symptoms: pd.DataFrame = gen_symptoms(users, NUM_SYMPTOMS_DAYS)
    locations: pd.DataFrame = gen_locations(users, NUM_LOCATIONS)

    users.to_sql("users", engine, if_exists="append", index=False)
    symptoms.to_sql("symptoms", engine, if_exists="append", index=False)
    locations.to_sql("locations", engine, if_exists="append", index=False)