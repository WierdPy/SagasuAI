import requests
from pymongo import MongoClient
from datetime import datetime, time
import pytz

# Funktion zum Abrufen der aktuellen Personenanzahl aus der MongoDB-Datenbank
def get_current_count():
    uri = "mongodb+srv://htl3r:ILoveZAI123@statisticdata.i2ntz.mongodb.net/?retryWrites=true&w=majority&appName=StatisticData"
    client = MongoClient(uri)
    db = client['Data']
    collection = db['PeopleCount']

    # Dokument mit unique_id: 0 abrufen
    document = collection.find_one({"unique_id": 0})

    client.close()

    if document:
        return document.get("currentCount", 0)
    return 0

# Funktion zum Senden einer Benachrichtigung
def send_notification(count, timestamp):
    url = "https://ntfy.sh/Sagasu"
    data = f"FIRE TEST: Current Count = {count} at {timestamp}"

    try:
        response = requests.post(url, data=data)
        response.raise_for_status()
        print(f"Notification sent: {data}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send notification: {e}")

# Hauptfunktion
# Überprüft die aktuelle Zeit und sendet eine Benachrichtigung, wenn Personen im Raum sind
# zu einer vorgegebenen Zeit.
def check_and_notify(scheduled_time):
    vienna_tz = pytz.timezone("Europe/Vienna")
    while True:
        # Aktuelle Uhrzeit in der Zeitzone von Wien abrufen
        current_time_vienna = datetime.now(vienna_tz)

        # Nur überprüfen, wenn die aktuelle Zeit mit der geplanten Zeit übereinstimmt
        if current_time_vienna.time() >= scheduled_time:
            # Personenanzahl abrufen
            current_count = get_current_count()

            if current_count > 0:
                # Benachrichtigung senden
                send_notification(current_count, current_time_vienna)

            break  # Nach Benachrichtigung die Schleife beenden

# Beispielhafte Verwendung der Funktion
if __name__ == "__main__":
    # Geplante Zeit (z. B. 14:00 Uhr)
    scheduled_time = time(11, 47, 0)  # 14:00:00

    check_and_notify(scheduled_time)