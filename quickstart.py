from config import API_KEY, SPREADSHEET_ID, RANGE_NAME, SCOPES
from googleapiclient.discovery import build




service = build("sheets", "v4", developerKey=API_KEY)
sheet = service.spreadsheets()


sheet_read = sheet.values().get(spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME).execute()

values = sheet_read.get("values", [])