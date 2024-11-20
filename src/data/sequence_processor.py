from datetime import datetime
from src.utils.Database_Connection import DatabaseConnection

class SequenceProcessor:
    def __init__(self, db: DatabaseConnection):
        self.db = db
        self.stats = {
            'total_rows': 0,
            'total_sequences': 0,
            'sequences_by_subject': 0,
            'sequences_by_executable': 0,
            'total_length': 0
        }
        
    def _ensure_table_exists(self, db: DatabaseConnection):
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS sequence (
            sequence_id SERIAL PRIMARY KEY,
            subject_uuid TEXT,
            executable TEXT,
            instance_number INTEGER,
            length INTEGER,
            ts_begin TIMESTAMP,
            ts_end TIMESTAMP,
            UNIQUE (subject_uuid, executable, instance_number)
        );
        '''
        db.cur.execute(create_table_query)
        db.conn.commit()
    
    def process_sequences(self):
        
        
        query = '''
        select e.subject_uuid, e.properties_map_exec, e.ts
        from event e
        join subject s 
            on e.subject_uuid = s.uuid
        order by e.subject_uuid, e.sequence_long;
        '''
        
        with self.db.connect() as db:
            # Create table if not exists
            self._ensure_table_exists(db)
            

            
            db.cur.execute(query)
            
            #extra curser zum einfügen
            insert_cur = db.conn.cursor()

            current = {
                'subject_uuid': None,
                'executable': None,
                'ts_begin': None,
                'ts_end': None,
                'no': 0,
                'length': 0
            }
            
            
            
            for row in db.cur:
                subject_uuid, executable, ts = row
                self.stats['total_rows'] += 1
                
                if executable is None:
                    continue
                
                # Behandle erste Zeile
                if current['subject_uuid'] is None:
                    current = self._reset_sequence(current, subject_uuid, executable, ts)
                    current['ts_begin'] = ts
                    current['length'] = 1
                    continue
                
                if self._is_new_sequence(current, subject_uuid, executable):
                    self._save_sequence(insert_cur, current, ts)
                    current = self._reset_sequence(current, subject_uuid, executable, ts)
                
                current['ts_end'] = ts
                current['length'] += 1
            
            # Save last sequence
            if current['subject_uuid'] is not None:
                self._save_sequence(insert_cur, current, current['ts_end'])
            db.conn.commit()
            
    def _is_new_sequence(self, current: dict, subject_uuid: str, executable: str) -> bool:
        if current['subject_uuid'] is None:
            return False
        return subject_uuid != current['subject_uuid'] or executable != current['executable']
    
    def _reset_sequence(self, current:dict, subject_uuid: str, executable: str, ts: datetime) -> dict:
        return {
            'subject_uuid': subject_uuid,
            'executable': executable,
            'ts_begin': ts,
            'ts_end': None,
            'no': 0 if subject_uuid != current['subject_uuid'] else current['no'] + 1,
            'length': 0
        }
    
    def _save_sequence(self, cur, current: dict, ts: datetime):
        if current['subject_uuid'] is None or current['executable'] is None:
            print("Skipping empty sequence") 
            return
            
        
        insert_query = '''
        insert into sequence (subject_uuid, executable, instance_number, length, ts_begin, ts_end)
        values (%s, %s, %s, %s, %s, %s)
        on conflict (subject_uuid, executable, instance_number) do nothing
        RETURNING sequence_id;
        '''
                
        cur.execute(insert_query, (
            current['subject_uuid'],
            current['executable'],
            current['no'],
            current['length'],
            current['ts_begin'],
            ts
        ))
        
        self.stats['total_sequences'] += 1
        self.stats['total_length'] += current['length']
        
        if current['no'] == 0:
            self.stats['sequences_by_subject'] += 1
        else:
            self.stats['sequences_by_executable'] += 1