import sqlite3
from contextlib import contextmanager


class RLDatabase:
    def __init__(self, path: str):
        self.path = path

        # initialize schema once
        with self._connect() as conn:
            conn.executescript("""
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS Episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_index INTEGER NOT NULL,
                start_time INTEGER,
                end_time INTEGER,
                total_steps INTEGER NOT NULL,
                total_reward REAL,
                termination_reason TEXT
            );

            CREATE TABLE IF NOT EXISTS Steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id INTEGER NOT NULL,
                tick INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                action INTEGER NOT NULL,
                reward REAL NOT NULL,

                x_position REAL,
                y_position REAL,
                z_position REAL,
                yaw REAL,
                pitch REAL,

                health REAL,
                food REAL,
                armor REAL,
                xp_level REAL,

                inventory TEXT,
                target TEXT,

                game_time INTEGER,
                raining INTEGER,
                thundering INTEGER,
                difficulty INTEGER,

                entities TEXT,
                blocks TEXT,

                FOREIGN KEY (episode_id) REFERENCES Episodes(id)
            );

            CREATE TABLE IF NOT EXISTS StepDiagnostics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                step_id INTEGER NOT NULL,
                log_prob REAL,
                advantage REAL,
                baseline REAL,
                FOREIGN KEY (step_id) REFERENCES Steps(id)
            );
            """)

    def _connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    @contextmanager
    def conn(self):
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def execute(self, sql: str, params=()):
        with self.conn() as c:
            cur = c.execute(sql, params)

            # only INSERTs produce useful IDs
            if sql.lstrip().upper().startswith("INSERT"):
                return cur.lastrowid

            return None

    def fetchone(self, sql: str, params=()):
        with self.conn() as c:
            cur = c.execute(sql, params)
            return cur.fetchone()

    def fetchall(self, sql: str, params=()):
        with self.conn() as c:
            cur = c.execute(sql, params)
            return cur.fetchall()