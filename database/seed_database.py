import pandas as pd
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in .env")

engine = create_engine(DATABASE_URL)

data_path = "data/structured_data"

files = [
    ("users.csv", "users"),
    ("products.csv", "products"),
    ("orders.csv", "orders"),
    ("order_items.csv", "order_items"),
    ("support_tickets.csv", "support_tickets"),
    ("ticket_messages.csv", "ticket_messages"),
]

with engine.begin() as connection:

    for filename, table_name in files:
        file_path = os.path.join(data_path, filename)

        df = pd.read_csv(file_path)

        df.to_sql(
            table_name,
            connection,
            if_exists="append",
            index=False
        )

        print(f"Loaded {len(df)} rows into {table_name}")

print("\nDatabase seeding completed successfully.")