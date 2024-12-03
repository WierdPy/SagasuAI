from pymongo import MongoClient

# Verbindung zur MongoDB herstellen
uri = "mongodb+srv://htl3r:ILoveZAI123@statisticdata.i2ntz.mongodb.net/?retryWrites=true&w=majority&appName=StatisticData"
client = MongoClient(uri)

# Datenbank und Collection auswählen
db = client['Data']
collection = db['PeopleCount']

# Dokumente löschen, die nicht unique_id: 0 haben
result = collection.delete_many({"unique_id": {"$ne": 0}})

# Anzahl der gelöschten Dokumente anzeigen
print(f"Anzahl der gelöschten Dokumente: {result.deleted_count}")

# Verbindung schließen
client.close()
