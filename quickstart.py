from typing import *
from googleapiclient.discovery import build
import pandas as pd
import streamlit as st
import numpy as np
from scipy import stats

def values_from_spreadsheet() -> List[Any]:
        # Access secrets directly from st.secrets dictionary
        api_key = st.secrets["API_KEY"]
        spreadsheet_id = st.secrets["SPREADSHEET_ID"]
        range_name = st.secrets["RANGE_NAME"]

        service = build("sheets", "v4", developerKey=api_key)
        sheet = service.spreadsheets()
        sheet_read = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
        values = sheet_read.get('values', [])
        return values


def generate_total_column(mines_df: pd.DataFrame) -> pd.DataFrame:
    total_df = pd.DataFrame(mines_df[mines_df.columns[0]])
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
        sub_df[col] = pd.to_numeric(df[col], errors='coerce')
        df_dict[col] = sub_df
    return df_dict

def basic_tests_per_df(df: pd.DataFrame) -> None:
    pass


def calculate_summary_stats(df_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    stats_list = []

    for name, df in df_dict.items():
        value_col = df.columns[1]
        series = df[value_col]

        mean = series.mean()
        median = series.median()
        std = series.std()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        stats_list.append({
            'Mine': name,
            'Mean': mean,
            'Median': median,
            'StdDev': std,
            'IQR': iqr,
            'Q1': q1, # Added this
            'Q3': q3  # Added this
        })

    summary_df = pd.DataFrame(stats_list)
    summary_df.set_index('Mine', inplace=True)
    return summary_df

def detect_anomalies(df: pd.DataFrame, stats_row: pd.Series) -> None:
    val_col = df.columns[1]

    mean = stats_row['Mean']
    std = stats_row['StdDev']
    q1 = stats_row['Q1']
    q3 = stats_row['Q3']
    iqr = stats_row['IQR']

    IQR_FACTOR = 1.5
    Z_THRESHOLD = 3.0
    MA_WINDOW = 5
    MA_THRESHOLD = 0.20
    GRUBBS_ALPHA = 0.05

    lower_bound = q1 - (IQR_FACTOR * iqr)
    upper_bound = q3 + (IQR_FACTOR * iqr)
    df['is_anomaly_IQR'] = (df[val_col] < lower_bound) | (df[val_col] > upper_bound)

    if std == 0:
        df['z_score'] = 0
    else:
        df['z_score'] = (df[val_col] - mean) / std
    df['is_anomaly_Zscore'] = df['z_score'].abs() > Z_THRESHOLD

    rolling_mean = df[val_col].rolling(window=MA_WINDOW, min_periods=1).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        pct_diff = np.abs((df[val_col] - rolling_mean) / rolling_mean)
    df['is_anomaly_MA'] = (pct_diff > MA_THRESHOLD) & (rolling_mean != 0)

    if std == 0:
        df['is_anomaly_Grubbs'] = False
    else:
        g_values = df['z_score'].abs()
        N = len(df)
        t_crit = stats.t.ppf(1 - GRUBBS_ALPHA / (2 * N), N - 2)
        g_critical = ((N - 1) * np.sqrt(np.square(t_crit))) / (np.sqrt(N) * np.sqrt(N - 2 + np.square(t_crit)))
        df['is_anomaly_Grubbs'] = g_values > g_critical

def apply_anomaly_tests(df_dict: Dict[str, pd.DataFrame], summary_df: pd.DataFrame) -> None:
    for name, df in df_dict.items():
        stats_row = summary_df.loc[name]

        detect_anomalies(df, stats_row)


def display_dashboard(df_dict: Dict[str, pd.DataFrame], summary_df: pd.DataFrame):
    tab_names = list(df_dict.keys())

    tabs = st.tabs(tab_names)

    for tab, mine_name in zip(tabs, tab_names):
        with tab:
            st.markdown(f"### 📊 Performance Overview: {mine_name}")

            stats = summary_df.loc[mine_name]

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Mean Output", f"{stats['Mean']:.2f}")
            with col2:
                st.metric("Median", f"{stats['Median']:.2f}")
            with col3:
                st.metric("Std Dev", f"{stats['StdDev']:.2f}")
            with col4:
                st.metric("IQR", f"{stats['IQR']:.2f}")

            st.divider()

            st.markdown("### 📋 Detailed Production Data")

            df = df_dict[mine_name]

            def style_anomalies(row):
                if row.get('is_anomaly_ANY', False):
                    return ['background-color: #ffe6e6; color: #990000'] * len(row)
                return [''] * len(row)

            display_cols = [df.columns[0], df.columns[1], 'is_anomaly_ANY']

            st.dataframe(
                df.style.apply(style_anomalies, axis=1),
                use_container_width=True,
                column_config={
                    "is_anomaly_ANY": st.column_config.CheckboxColumn(
                        "Anomaly Detected?",
                        help="True if any of the 4 statistical tests flagged this row."
                    )
                }
            )

            # Optional: Show a breakdown of which test failed for the anomalies
            # if df['is_anomaly_ANY'].any():
            #     with st.expander("⚠️ View Anomaly Details"):
            #         anomalies = df[df['is_anomaly_ANY']].copy()
            #         st.write(anomalies)

def main():
    mines_df = get_df_from_spreadsheet()
    df_dict = split_data_into_separate_df(mines_df)
    df_dict['Total'] = generate_total_column(mines_df)
    summary_df = calculate_summary_stats(df_dict)
    apply_anomaly_tests(df_dict, summary_df)
    display_dashboard(df_dict, summary_df)

if __name__ == "__main__":
    main()
