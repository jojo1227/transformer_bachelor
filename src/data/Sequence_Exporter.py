import numpy as np
import json
from typing import Tuple, List
from src.utils.Database_Connection import DatabaseConnection

class SequenceDataExporter:
    def __init__(self, db: DatabaseConnection):
        self.db = db

    def export_sequences(self, sequences_file: str, targets_file: str):
        query = """
        WITH ordered_events AS (
            SELECT 
                se.sequence_id,
                array_agg(se.event_type ORDER BY se.event_order) as events,
                se.executable
            FROM sequence_events se
            GROUP BY se.sequence_id, se.executable
            ORDER BY se.sequence_id
        )
        SELECT events, executable FROM ordered_events;
        """
        
        sequences = []
        targets = []
        
        with self.db.connect() as db:
            db.cur.execute(query)
            
            for row in db.cur:
                event_sequence, executable = row
                sequences.append(event_sequence)
                targets.append(executable)

        # Convert sequences to numpy array with object dtype to handle variable-length lists
        sequences_array = np.array(sequences, dtype=object)
        targets_array = np.array(targets)

        # Save using numpy
        np.save(sequences_file, sequences_array)
        np.save(targets_file, targets_array)
