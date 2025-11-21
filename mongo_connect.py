from pymongo import MongoClient
from urllib.parse import quote_plus

user = "admin"
password = quote_plus("admin")
uri = f"mongodb+srv://{user}:{password}@starkadvisor.aq4pvph.mongodb.net/starkadvisor?retryWrites=true&w=majority"

client = MongoClient(uri)
print(client.list_database_names())
