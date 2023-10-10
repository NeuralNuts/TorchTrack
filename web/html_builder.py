import webbrowser
import os
 
f = open('GFG.html', 'w')
 
html_template = """
<html>
<head></head>
<body>
<p>Geeks For Geeks</p>
 
</body>
</html>
"""

#f.write(html_template)
#f.close()
# 1st method how to open html files in chrome using
 
filename = 'file:///'+os.getcwd()+'/' + 'torch_track_site.html'
webbrowser.open_new_tab(filename)

import json
import bson
from datetime import datetime
from bson.json_util import loads

# Your JSON data
json_data = """
{
  "name": "John Doe",
  "email": "john@example.com",
  "birthday": "1985-01-22",
  "subscribed": true,
  "preferences": ["email", "sms"],
  "contacts": {
    "home": "123-456-7890",
    "work": "098-765-4321"
  }
}
"""

# Convert JSON to BSON
bson_data = bson.BSON(loads(json_data))

# Write BSON to file
with open('data.bson', 'wb') as file:
    file.write(bson_data)

print("BSON data has been written to data.bson")

# Read BSON from file
with open('data.bson', 'rb') as file:
    bson_data_from_file = file.read()

# Convert BSON back to JSON
json_data_from_bson = dumps(bson.BSON.decode(bson_data_from_file))

print("The JSON data decoded from BSON is: ")
print(json_data_from_bson)

