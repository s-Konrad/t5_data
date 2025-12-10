from typing import *
from config import API_KEY, SPREADSHEET_ID, RANGE_NAME, SCOPES
from googleapiclient.discovery import build
import pandas as pd
import streamlit as st


def values_from_spreadsheet() -> List[Any]:
    service = build("sheets", "v4", developerKey=API_KEY)
    sheet = service.spreadsheets()
    sheet_read = sheet.values().get(spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME).execute()
    values = sheet_read.get("values", [])
    return values


def generate_total_column(values: List[Any]):
    df = pd.DataFrame(values[1:], columns=values[0])
    df['Total'] = 0
    for col in df.columns[1:-1]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df['Total'] += df[col]
    return df

def main():
    values = values_from_spreadsheet()
    mines_df = generate_total_column(values)
