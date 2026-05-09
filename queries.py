# Every SQL query the bot uses, as named functions: get_cached_profile(), save_profile(), record_bet(), get_bet_history(), get_cached_embedding(), save_embedding().
# No raw SQL anywhere else in the codebase — all queries live here.

import sqlite3
from .connection import get_connection
from typing import Any, Optional, List # For type hints of query results

def initialize_database():
        # Create transcripts_storage table
        with get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS transcripts_storage(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    speaker TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    date_published DATE NOT NULL,
                    transcript_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    extracted_text TEXT,
                    word_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(speaker, event_type, transcript_name)
                )
            ''')
            # Create archive table for old transcripts (optional, can also just delete old ones)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS transcripts_archive(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    speaker TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    date_published DATE NOT NULL,
                    transcript_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    extracted_text TEXT,
                    word_count INTEGER,
                    archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # Create speaker_profiles table
            # hit_rate_recent will take the hit rate in 30 days
            # momentum is hit_rate_recent - hit_rate_lifetime
            conn.execute('''
                CREATE TABLE IF NOT EXISTS speaker_profiles(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    speaker TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    word TEXT NOT NULL,
                    hit_rate_lifetime REAL DEFAULT 0.0,
                    hit_rate_recent REAL DEFAULT 0.0, 
                    momentum REAL DEFAULT 0.0,
                    avg_freq REAL DEFAULT 0.0,
                    recency REAL DEFAULT 0.0,
                    n_samples_lifetime INTEGER DEFAULT 0,
                    n_samples_recent INTEGER DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(speaker, event_type, word)
                )
            ''')
            # Create embeddings table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS trade_log(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mode TEXT NOT NULL DEFAULT 'paper' CHECK(mode IN ('paper', 'live')),
                    ticker TEXT NOT NULL,
                    speaker TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    word TEXT NOT NULL,
                    our_probability REAL NOT NULL,
                    kalshi_odds REAL NOT NULL,
                    ev_per_contract REAL NOT NULL,
                    bet_side TEXT NOT NULL CHECK(bet_side IN ('yes', 'no')), 
                    contracts INTEGER NOT NULL DEFAULT 0,
                    outcome TEXT CHECK(outcome IN ('win', 'loss', 'cancelled')) DEFAULT NULL,
                    payout_cents REAL DEFAULT 0.0,
                    placed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TIMESTAMP
                )
            ''')
            # Create news_cache table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS news_cache(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    headline TEXT NOT NULL,
                    source TEXT,
                    date_published DATE NOT NULL,
                    search_query TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # Embeddings (Crucial for RAG/Similarity)
            # Table reference can be 'speaker_profiles' or 'transcripts_storage' to link back to the source.
            conn.execute('''
                CREATE TABLE IF NOT EXISTS embeddings_cache(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    target_id INTEGER NOT NULL,
                    table_reference TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    UNIQUE(target_id, table_reference)
                )
            ''')

            # Commit the changes to save the new tables
            conn.commit()

def get_cached_profile(speaker: str, word: str, event_type: str = None) -> Optional[sqlite3.Row]:
    # Fetches the current 'Snapshot' of a speaker's habits.
    with get_connection() as conn:
        query = "SELECT * FROM speaker_profiles WHERE speaker = ? AND word = ?"
        params = [speaker, word]
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        
        return conn.execute(query, params).fetchone()

def insert_new_profile(speaker: str, word: str, event_type: str):
    # Initializes a blank profile for the 'Red Path' (New Speakers).
    with get_connection() as conn:
        conn.execute('''
            INSERT OR IGNORE INTO speaker_profiles (speaker, word, event_type)
            VALUES (?, ?, ?)
        ''', (speaker, word, event_type))
        conn.commit()

def update_profile(speaker: str, word: str, event_type: str, hit_rate_lifetime: float, avg_freq: float, recency: float):
    
    # Updates the profile and automatically calculates Momentum.
    # Called after the Scraper/Word Counter finishes a new batch.
    
    with get_connection() as conn:
        # 1. Calculate the 'Recent' (30-day) hit rate from raw transcripts
        recent_sql = """
            SELECT AVG(CASE WHEN extracted_text LIKE ? THEN 1.0 ELSE 0.0 END), COUNT(*)
            FROM transcripts_storage 
            WHERE speaker = ? AND date_published > date('now', '-30 days')
        """
        recent_res = conn.execute(recent_sql, (f"%{word}%", speaker)).fetchone()
        recent_hit_rate = recent_res[0] if recent_res[0] is not None else 0.0
        n_samples_recent = recent_res[1] if recent_res[1] is not None else 0

        # 2. Update the profile with the new numbers
        conn.execute('''
            UPDATE speaker_profiles 
            SET hit_rate_lifetime = ?, 
                hit_rate_recent = ?,
                momentum = ? - ?,
                avg_freq = ?, 
                recency = ?,
                n_samples_lifetime = n_samples_lifetime + 1,
                n_samples_recent = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE speaker = ? AND word = ? AND event_type = ?
        ''', (hit_rate_lifetime, recent_hit_rate, recent_hit_rate, hit_rate_lifetime, 
              avg_freq, recency, n_samples_recent, speaker, word, event_type))
        
        conn.commit()

# Inserts a new transcript or updates the existing one if the same speaker/event_type/transcript_name already exists. 
# This allows us to keep the most recent version of the transcript without creating duplicates, which is important if multiple agents scrape the same event.
def insert_update_transcript(speaker: str, event_type: str, transcript_name: str, file_path: str, text: str, date_pub: str):
    # Prevents duplicate counts even if multiple agents scrape the same event.
    with get_connection() as conn:
        conn.execute('''
            INSERT INTO transcripts_storage (speaker, event_type, transcript_name, file_path, extracted_text, date_published)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(speaker, event_type, transcript_name) DO UPDATE SET
                extracted_text = excluded.extracted_text,
                date_published = excluded.date_published
        ''', (speaker, event_type, transcript_name, file_path, text, date_pub))
        conn.commit()

def get_transcripts(
    speaker: str = None, 
    event_type: str = None, 
    target_word: str = None,
    limit: int = 50,
    offset: int = 0
) -> List[sqlite3.Row]:
    # Universal discovery method for the Logic Agent and maybe Dashboard.
    with get_connection() as conn:
        query = "SELECT * FROM transcripts_storage"
        params = []
        conditions = []

        if speaker:
            conditions.append("speaker = ?")
            params.append(speaker)
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        if target_word:
            conditions.append("extracted_text LIKE ?")
            params.append(f"%{target_word}%")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY date_published DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        return conn.execute(query, params).fetchall()

# Archives old transcripts to free up active storage while retaining data for long-term momentum calculations.
def archive_old_transcripts(weeks: int = 4):
    with get_connection() as conn:
        # Copy to Archive
        conn.execute('''
            INSERT INTO transcripts_archive (speaker, event_type, date_published, transcript_name, file_path, extracted_text, word_count)
            SELECT speaker, event_type, date_published, transcript_name, file_path, extracted_text, word_count 
            FROM transcripts_storage 
            WHERE date_published < date('now', ?)
        ''', (f"-{weeks * 7} days",))
        
        # Now Delete
        conn.execute('DELETE FROM transcripts_storage WHERE date_published < date("now", ?)', (f"-{weeks * 7} days",))
        conn.commit()

def save_embedding(target_id: int, table_ref: str, vector: bytes):
    # Saves semantic vector blobs for similarity searching.
    with get_connection() as conn:
        conn.execute('''
            INSERT OR REPLACE INTO embeddings_cache (target_id, table_reference, vector)
            VALUES (?, ?, ?)
        ''', (target_id, table_ref, vector))
        conn.commit()

def get_cached_embedding(target_id: int, table_ref: str) -> Optional[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute('''
            SELECT * FROM embeddings_cache WHERE target_id = ? AND table_reference = ?
        ''', (target_id, table_ref)).fetchone()

def insert_news_cache(headline: str, source: str, date_published: str, search_query: str):
    with get_connection() as conn:
        conn.execute('''
            INSERT INTO news_cache (headline, source, date_published, search_query)
            VALUES (?, ?, ?, ?)
        ''', (headline, source, date_published, search_query))
        conn.commit()

def get_recent_news_cache(search_query: str, days: int = 1) -> list[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute('''
            SELECT * FROM news_cache 
            WHERE search_query = ? AND created_at > date('now', ?)
            ORDER BY created_at DESC
        ''', (search_query, f"-{days} days")).fetchall()

def delete_old_news_cache(days: int = 7):
    with get_connection() as conn:
        conn.execute('DELETE FROM news_cache WHERE created_at < date("now", ?)', (f"-{days} days",))
        conn.commit()

# This includes without the resolved outcome and payout, which can be updated later when the bet is resolved.
def record_trade_initial(mode: str, ticker: str, speaker: str, event_type: str, word: str, our_probability: float, kalshi_odds: float, ev_per_contract: float, bet_side: str, contracts: int):
    with get_connection() as conn:
        conn.execute('''
            INSERT INTO trade_log (mode, ticker, speaker, event_type, word, our_probability, kalshi_odds, ev_per_contract, bet_side, contracts) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (mode, ticker, speaker, event_type, word, our_probability, kalshi_odds, ev_per_contract, bet_side, contracts))
        conn.commit()

def record_outcome(trade_id: int, outcome: str, payout_cents: float):
    # Updates the log once the Kalshi market settles.
    with get_connection() as conn:
        conn.execute('''
            UPDATE trade_log 
            SET outcome = ?, payout_cents = ?, resolved_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (outcome, payout_cents, trade_id))
        conn.commit()