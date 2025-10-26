import psycopg2
from dotenv import load_dotenv
import os
load_dotenv()
print(os.getenv("DATABASE_URL"))
conn = psycopg2.connect(os.getenv("DATABASE_URL"))
print("Connected successfully!")
conn.close()
