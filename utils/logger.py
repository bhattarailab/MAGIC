import os
import json
from datetime import timezone, datetime, timedelta

def get_current_time():
	"""
	returns current date and time in nepal time zone 
	"""
	tz = timezone(timedelta(hours=5, minutes=45))
	return datetime.now(tz).strftime("%d/%m/%Y, %H:%M:%S")

class Logger:
	def __init__(self, path):
		if os.path.exists("path"):
			raise FileExistsError(f"File already exists in path, {path}")
		
		self.path = path
		self.logs = {
			"timestamp": get_current_time(),
		}

	def save(self):
		with open(self.path, "w") as fp:
			json.dump(self.logs, fp, indent=2)

	def add(self, key, value):
		self.logs[key] = value
