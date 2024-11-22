from src.utils.Database_Connection import DatabaseConnection


class SequenceEventProcessor:
    def __init__(self, db: DatabaseConnection):
        self.db = db
        self.stats = {
            'total_events': 0,
            'total_sequences': 0
        }
    
    def _ensure_table_exists(self, db: DatabaseConnection):
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS sequence_events (
            sequence_id INTEGER REFERENCES sequence(sequence_id),
            event_order INTEGER,
            event_type TEXT,
            executable TEXT,
            PRIMARY KEY (sequence_id, event_order)
        );
        '''
        db.cur.execute(create_table_query)
        db.conn.commit()
    
    def process_sequence_events(self):
        query = '''
        WITH sequences AS (
            SELECT sequence_id, subject_uuid, executable, instance_number
            FROM sequence
            WHERE ts_end < '2018-04-06 11:20:00'
        )
        SELECT s.sequence_id, e.type, s.executable
        FROM sequences s
        JOIN event e ON s.subject_uuid = e.subject_uuid
        ORDER BY s.sequence_id, e.sequence_long;
        '''
        
        with self.db.connect() as db:
            self._ensure_table_exists(db)
            
            db.cur.execute(query)
            insert_cur = db.conn.cursor()
            
            current_sequence = {
                'id': None,
                'executable': None,
                'events': []
            }
            
            for row in db.cur:
                sequence_id, event_type, executable = row
                self.stats['total_events'] += 1
                
                if current_sequence['id'] is None:
                    current_sequence['id'] = sequence_id
                    current_sequence['executable'] = executable
                    current_sequence['events'] = [event_type]
                elif current_sequence['id'] != sequence_id:
                    self._save_sequence_events(insert_cur, current_sequence)
                    current_sequence = {
                        'id': sequence_id,
                        'executable': executable,
                        'events': [event_type]
                    }
                else:
                    current_sequence['events'].append(event_type)
            
            if current_sequence['id'] is not None:
                self._save_sequence_events(insert_cur, current_sequence)
                
            insert_cur.close()
            db.conn.commit()
    
    def _save_sequence_events(self, cur, sequence: dict):
        if not sequence['events']:
            return
            
        insert_query = '''
        INSERT INTO sequence_events (sequence_id, event_order, event_type, executable)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT DO NOTHING;
        '''
        
        for event_order, event_type in enumerate(sequence['events']):
            cur.execute(insert_query, (
                sequence['id'],
                event_order,
                event_type,
                sequence['executable']
            ))
        
        self.stats['total_sequences'] += 1