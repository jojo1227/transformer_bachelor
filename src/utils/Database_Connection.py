import psycopg2
import contextlib

class DatabaseConnection:
    def __init__(self, host: str, database: str, user: str, password: str, port: int):
        self.connection_params = {
            'host': host,
            'database': database,
            'user': user,
            'password': password,
            'port': port
        }
        self.conn = None
        self.cur = None
        
    @contextlib.contextmanager
    def connect(self):
        try:
            self.conn = psycopg2.connect(**self.connection_params)
            self.conn.set_session(readonly=False)
            self.cur = self.conn.cursor()
            yield self
        finally:
            if self.cur:
                self.cur.close()
            if self.conn:
                self.conn.close()
    def getConnection(self):
        return self.conn
    
    def getCursor(self):
        return self.cur