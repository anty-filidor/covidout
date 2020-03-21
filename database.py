from sqlalchemy import create_engine
import pandas.io.sql as psql

from sqlalchemy import create_engine
import pandas as pd
from gen_database import gen_users, gen_symptoms, gen_locations

# follows django database settings format, replace with your own settings
DATABASES = {
    'production':{
        'NAME': 'cvdb',
        'USER': 'postgres',
        'PASSWORD': 'admin',
        'HOST': '95.179.154.232',
        'PORT': 5432,
    },
}

# choose the database to use
db = DATABASES['production']

# construct an engine connection string
engine_string = "postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}".format(
    user = db['USER'],
    password = db['PASSWORD'],
    host = db['HOST'],
    port = db['PORT'],
    database = db['NAME']
)

# create sqlalchemy engine
engine = create_engine(engine_string)

with engine.connect() as con:
    con.execute("DELETE FROM locations WHERE user_id < 99999;DELETE FROM symptoms WHERE user_id < 99999;DELETE FROM users WHERE user_id <99999")

users: pd.DataFrame = gen_users(100)
symptoms: pd.DataFrame = gen_symptoms(users, 10)
locations: pd.DataFrame = gen_locations(users, 10)

users.to_sql("users", engine, if_exists="append", index=False)
symptoms.to_sql("symptoms", engine, if_exists="append", index=False)
locations.to_sql("locations", engine, if_exists="append", index=False)