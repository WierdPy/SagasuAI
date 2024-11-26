import pymysql
import requests

apisend = True


def notify_person_event(event_type, timestamp, original_count, new_count):
    """
    Sends a notification to the API about a person entering or exiting and logs the event in the database.
    :param event_type: 'Enter' for entering the frame, 'Exit' for exiting the frame.
    :param timestamp: The timestamp of the event.
    :param original_count: The initial count of people before the event.
    :param new_count: The new count of people after the event.
    :param apisend: Boolean to control whether to send the notification to the API.
    """
    # 1. API-Benachrichtigung senden
    url = "https://ntfy.sh/Sagasu"
    data = event_type  # send 'enter' or 'exit' as data
    if apisend:
        try:
            response = requests.post(url, data=data)
            response.raise_for_status()
            #print(f"Notification sent: {event_type}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to send notification: {e}")
