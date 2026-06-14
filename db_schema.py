import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()
conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cur = conn.cursor()
cur.execute("SELECT column_name, is_nullable FROM information_schema.columns WHERE table_name = 'users';")
print(cur.fetchall())
