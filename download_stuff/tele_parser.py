import csv
from datetime import datetime
from telethon.sync import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest
from telethon.tl.types import PeerChannel

from dotenv import dotenv_values

# Setting configuration values
api_id = dotenv_values('../../.env')['api_id']
api_hash = dotenv_values('../../.env')['api_hash']
username = dotenv_values('../../.env')['username']
chat = 'markettwits'

limit = 5000

offset_date = datetime(2025, 3, 13)


with TelegramClient(username, api_id, api_hash) as client:
    with open('markettwits.tsv', mode='w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(["ID", "Text", "Date"])
        for message in client.iter_messages(chat, limit=limit, offset_date=offset_date):
            writer.writerow([message.id, message.message.replace('\t', '\\t').replace('\n', '\\n'), message.date])
