import matplotlib.pyplot as plt

class SequenceVisualizer:
    def __init__(self, database_connection):
        """
        Initialisiert den Visualisierer mit existierender Datenbankverbindung
        
        Args:
            database_connection: Bestehende Datenbankverbindung
        """
        self.conn = database_connection
        
    def cm_to_inches(self, cm):
        """Konvertiert Zentimeter zu Zoll"""
        return cm / 2.54
    
    def group_bottom_percent(self, labels, sizes, threshold):
        """
        Gruppiert kleine Elemente in einer 'Andere'-Kategorie
        
        Args:
            labels (list): Liste von Beschriftungen
            sizes (list): Liste von Größen/Häufigkeiten
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
    
    def pie_chart_from_query(self, query, output_path: str = None, threshold: float = 0.01, figsize_cm: int = 20):
        """
        Erstellt ein Pie-Chart direkt aus einer SQL-Abfrage
        
        Args:
            query (str): SQL-Abfrage
            output_path (str): Ausgabepfad für SVG
            threshold (float, optional): Gruppierungsschwelle. Defaults to 0.01.
            figsize_cm (int, optional): Figurengröße in Zentimetern. Defaults to 20.
        """
        
        with self.conn.connect() as conn:
            
            # Cursor erstellen und Abfrage ausführen
            with self.conn.cur as cur:
                cur.execute(query)
                result = cur.fetchall()
            
            # Daten extrahieren
            labels = [row[0] for row in result]
            sizes = [row[1] for row in result]
            
            # Kleine Kategorien gruppieren
            labels, sizes = self.group_bottom_percent(labels, sizes, threshold)
            
            # Figure und Plot erstellen
            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', textprops={'size': 'medium'})
            ax.axis('equal')
            
            # Größe setzen
            fig.set_size_inches((self.cm_to_inches(figsize_cm), self.cm_to_inches(figsize_cm)))
            
            # Speichern und schließen
            plt.show()
            
            if(output_path is not None):
                plt.savefig(output_path)
            plt.close()