# Creates and returns the SQLite connection. Uses a context manager so connections are always closed properly. 
# Sets WAL mode for better concurrent read performance. 
# All other database files import get_connection() from here.

import sqlite3
import os
from pathlib import Path
from contextlib import contextmanager
from dotenv import load_dotenv

# Locates the root folder even if called from /pipeline or /database
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DB_FILENAME = os.getenv("DB_PATH", "database/schemas.db")
DB_PATH = BASE_DIR / DB_FILENAME

@contextmanager
def get_connection():
    # Connect to the SQLite database with a timeout to handle potential locking issues
    conn = sqlite3.connect(str(DB_PATH), timeout=30)

    try:
        # Set row factory to sqlite3.Row to allow accessing columns by name (e.g. row['column_name'])
        conn.row_factory = sqlite3.Row 

        # Set journal mode to WAL for better concurrent read performance
        conn.execute('PRAGMA journal_mode=WAL;')

        # enforce foreign key constraints
        conn.execute("PRAGMA foreign_keys=ON")

        yield conn  # Yield the connection for use in a context manager
    finally:
        conn.close()  # Ensure the connection is closed if error occurs