import matplotlib.pyplot as plt

class SequenceEventVisualizer:
    def __init__(self, database_connection):
        """
        Initialisiert den Visualisierer mit Datenbankverbindungsobjekt
        
        Args:
            database_connection (DatabaseConnection): Datenbankverbindungsobjekt
        """
        self.db_connection = database_connection
    
    @staticmethod
    def group_bottom_percent(labels, sizes, threshold):
        """
        Gruppiert kleine Kategorien in einer 'andere'-Kategorie
        
        Args:
            labels (list): Liste von Beschriftungen
            sizes (list): Liste von Häufigkeiten
            threshold (float): Prozentschwelle für Gruppierung
        
        Returns:
            tuple: Gefilterte Labels und Größen
        """
        total = sum(sizes)
        labels_filtered = []
        sizes_filtered = []
        count = 0
        other_count = 0
        
        for label, size in zip(labels, sizes):
            if size / total < threshold:
                count += 1
                other_count += size
            else:
                labels_filtered.append(label)
                sizes_filtered.append(size)
        
        labels_filtered.append(f'andere ({count})')
        sizes_filtered.append(other_count)
        
        return labels_filtered, sizes_filtered
    
    def create_pie_chart(self, query, output_path: str = None, threshold: float = 0.01, figsize_cm: int = 20):
        """
        Erstellt ein Pie-Chart aus einer Datenbankabfrage
        
        Args:
            query (str): SQL-Abfrage
            output_path (str): Ausgabepfad für SVG
            threshold (float, optional): Gruppierungsschwelle. Defaults to 0.01.
            figsize_cm (int, optional): Figurengröße in Zentimetern. Defaults to 20.
        """
        # Verbindung und Cursor über Kontextmanager
        with self.db_connection.connect():
            cur = self.db_connection.getCursor()
            
            # Abfrage ausführen
            cur.execute(query)
            res = cur.fetchall()
        
        # Daten extrahieren
        labels = [r[0] for r in res]
        values = [r[1] for r in res]
        
        # Kleine Kategorien gruppieren
        labels, values = self.group_bottom_percent(labels, values, threshold)
        
        # Konvertierung Zentimeter zu Zoll
        figsize_inches = (figsize_cm / 2.54, figsize_cm / 2.54)
        
        # Figure und Plot erstellen
        fig, ax = plt.subplots(figsize=figsize_inches)
        fig.tight_layout()
        ax.set_aspect('equal')
        ax.pie(values, labels=labels, autopct='%1.1f%%')
        
        # Speichern und schließen
        plt.show()
        if output_path is not None:
            plt.savefig(output_path)
        plt.close()