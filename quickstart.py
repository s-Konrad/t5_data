from typing import *
from config import API_KEY, SPREADSHEET_ID, RANGE_NAME, SCOPES
from googleapiclient.discovery import build
import pandas as pd
import streamlit as st


def values_from_spreadsheet() -> List[Any]:
    service = build("sheets", "v4", developerKey=API_KEY)
    sheet = service.spreadsheets()
    sheet_read = sheet.values().get(spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME).execute()
    values = sheet_read.get('values', [])
    return values


def generate_total_column(mines_df: pd.DataFrame) -> pd.DataFrame:
    total_df = mines_df[mines_df.columns[0]]
    total_df['Total'] = 0
    for col in mines_df.columns[1:]:
        mines_df[col] = pd.to_numeric(mines_df[col], errors='coerce')
        total_df['Total'] += mines_df[col]
    return total_df

def get_df_from_spreadsheet() -> pd.DataFrame:
    values = values_from_spreadsheet()
    df = pd.DataFrame(values[1:], columns=values[0])
    return df

def split_data_into_separate_df(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    date_column = df.columns[0]
    df_dict = {}
    for col in df.columns[1:]:
        sub_df = pd.DataFrame(df[date_column])
        sub_df[col] = df[col]
        df_dict[col] = sub_df
    return df_dict

def main():
    mines_df = get_df_from_spreadsheet()
    df_dict = split_data_into_separate_df(mines_df)
    df_dict['Total'] = generate_total_column(mines_df)

    means = mines_df.mean(numeric_only=True)
    stds = mines_df.std(numeric_only=True)
    medians = mines_df.median(numeric_only=True)
    qs1 = mines_df.quantile(q=0.25, axis='rows', numeric_only=True, interpolation='lower')
    qs3 = mines_df.quantile(q=0.75, axis='rows', numeric_only=True, interpolation='lower')
    iqrs = qs3 - qs1


if __name__ == "__main__":
    main()
