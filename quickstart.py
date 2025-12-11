from typing import *
from googleapiclient.discovery import build
import pandas as pd
import streamlit as st
import numpy as np
from scipy import stats
import plotly.express as px

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


def check_iqr(df: pd.DataFrame, stats: pd.Series, params: Dict[str, Any]) -> pd.Series:
    """Returns a Boolean Series indicating IQR anomalies."""
    val_col = df.columns[1]
    k = params['iqr_factor']

    lower_bound = stats['Q1'] - (k * stats['IQR'])
    upper_bound = stats['Q3'] + (k * stats['IQR'])

    return (df[val_col] < lower_bound) | (df[val_col] > upper_bound)


def check_zscore(df: pd.DataFrame, stats: pd.Series, params: Dict[str, Any]) -> Tuple[pd.Series, pd.Series]:
    """
    Returns two Series:
    1. The calculated Z-score values (float)
    2. The Boolean anomaly flags
    """
    val_col = df.columns[1]
    mean, std = stats['Mean'], stats['StdDev']

    # Calculate Z-Scores
    if std == 0:
        z_scores = pd.Series(0, index=df.index)
    else:
        z_scores = (df[val_col] - mean) / std

    is_anomaly = z_scores.abs() > params['z_threshold']
    return z_scores, is_anomaly


def check_moving_average(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Returns a Boolean Series indicating Moving Average anomalies."""
    val_col = df.columns[1]
    window = params['ma_window']
    threshold = params['ma_threshold']

    rolling_mean = df[val_col].rolling(window=window, min_periods=1).mean()

    with np.errstate(divide='ignore', invalid='ignore'):
        pct_diff = np.abs((df[val_col] - rolling_mean) / rolling_mean)

    return (pct_diff > threshold) & (rolling_mean != 0)


def check_grubbs(df: pd.DataFrame, z_scores: pd.Series) -> pd.Series:
    """Returns a Boolean Series for the single most extreme outlier."""
    N = len(df)
    alpha = 0.05

    # Calculate Critical G Value
    t_crit = stats.t.ppf(1 - alpha / (2 * N), N - 2)
    g_critical = ((N - 1) * np.sqrt(np.square(t_crit))) / (np.sqrt(N) * np.sqrt(N - 2 + np.square(t_crit)))

    return z_scores.abs() > g_critical

def detect_anomalies(df: pd.DataFrame, stats_row: pd.Series, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Orchestrates the anomaly detection by calling individual test functions.
    """
    # 1. Run IQR Test
    df['is_anomaly_IQR'] = check_iqr(df, stats_row, params)

    # 2. Run Z-Score Test (We get the column AND the flags back)
    df['z_score'], df['is_anomaly_Zscore'] = check_zscore(df, stats_row, params)

    # 3. Run Moving Average Test
    df['is_anomaly_MA'] = check_moving_average(df, params)

    # 4. Run Grubbs' Test (Re-uses the Z-scores we just calculated)
    # We guard against empty dataframes or zero variance
    if len(df) > 2 and stats_row['StdDev'] > 0:
        df['is_anomaly_Grubbs'] = check_grubbs(df, df['z_score'])
    else:
        df['is_anomaly_Grubbs'] = False

    # 5. Summary Column
    df['is_anomaly_ANY'] = (
        df['is_anomaly_IQR'] |
        df['is_anomaly_Zscore'] |
        df['is_anomaly_MA'] |
        df['is_anomaly_Grubbs']
    )

    return df

def apply_anomaly_tests(df_dict: Dict[str, pd.DataFrame], summary_df: pd.DataFrame, params) -> None:
    for name, df in df_dict.items():
        stats_row = summary_df.loc[name]

        detect_anomalies(df, stats_row, params)


def render_graph(df: pd.DataFrame, mine_name: str, graph_type: str):
    """
    Renders a Plotly graph based on the user's selection.
    Highlights anomalies in Red.
    """
    # Prepare data for plotting
    # We create a new 'Color' column to define the legend colors dynamically
    plot_df = df.copy()
    plot_df['Status'] = plot_df['is_anomaly_ANY'].apply(lambda x: 'Anomaly' if x else 'Normal')

    # Define color map: Normal = Blue/Grey, Anomaly = Red
    color_map = {'Normal': '#636EFA', 'Anomaly': '#EF553B'}

    # 1. Line Chart
    if graph_type == "Line Chart":
        # We use a scatter plot with lines to allow marker coloring for anomalies
        fig = px.line(plot_df, x=plot_df.columns[0], y=plot_df.columns[1], title=f"{mine_name} Production Trend")
        # Add red markers for anomalies
        anomalies = plot_df[plot_df['is_anomaly_ANY']]
        fig.add_scatter(x=anomalies.iloc[:, 0], y=anomalies.iloc[:, 1], mode='markers',
                        marker=dict(color='red', size=10), name='Anomaly')

    # 2. Bar Chart (Separate)
    elif graph_type == "Bar Chart":
        fig = px.bar(plot_df, x=plot_df.columns[0], y=plot_df.columns[1],
                     color='Status', color_discrete_map=color_map,
                     title=f"{mine_name} Daily Output")

    # 3. Bar Chart (Stacked) / Area
    # For a single mine, "Stacked" acts like an Area chart or a full bar.
    # We will use an Area Chart here as it is the closest "filled" equivalent
    # to a stacked chart for a single time series.
    else:
        fig = px.area(plot_df, x=plot_df.columns[0], y=plot_df.columns[1],
                      title=f"{mine_name} Cumulative View")
        # We can still highlight points if needed, but Area is usually for trends

    # Update layout for professional look
    fig.update_layout(xaxis_title="Date", yaxis_title="Production Output", hovermode="x unified")

    st.plotly_chart(fig, use_container_width=True)

def render_metrics(stats: pd.Series, mine_name: str):
    """Renders the professional statistics display using st.metric."""
    st.markdown(f"### 📊 Performance Overview: {mine_name}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Mean Output", f"{stats['Mean']:.2f}")
    with col2:
        st.metric("Median", f"{stats['Median']:.2f}")
    with col3:
        st.metric("Std Dev", f"{stats['StdDev']:.2f}")
    with col4:
        st.metric("IQR", f"{stats['IQR']:.2f}")


def render_chart_tab(df: pd.DataFrame, mine_name: str):
    """Renders the graph selection and the selected Plotly chart."""
    st.markdown("### 📈 Production Visualizations")

    # Graph Type Selector (Unique key for each tab)
    graph_type = st.radio(
        "Choose Graph Style:",
        ["Line Chart", "Bar Chart", "Area / Stacked"],
        horizontal=True,
        key=f"graph_select_{mine_name}"
    )

    # The render_graph function itself (from the previous step) is called here
    render_graph(df, mine_name, graph_type)


def render_data_table(df: pd.DataFrame):
    """Renders the detailed data table with conditional anomaly styling."""
    st.markdown("### 📋 Detailed Production Data")

    def style_anomalies(row):
        # Apply light red background if ANY anomaly is flagged
        if row.get('is_anomaly_ANY', False):
            return ['background-color: #ffe6e6; color: #990000'] * len(row)
        return [''] * len(row)

    st.dataframe(
        df.style.apply(style_anomalies, axis=1),
        use_container_width=True,
        column_config={
            "is_anomaly_ANY": st.column_config.CheckboxColumn(
                "Anomaly?",
                help="True if flagged by statistical tests."
            )
        }
    )

    def display_dashboard(df_dict: Dict[str, pd.DataFrame], summary_df: pd.DataFrame):
        """
        Orchestrates the entire dashboard display using Streamlit tabs and helper functions.
        """
        tab_names = list(df_dict.keys())
        tabs = st.tabs(tab_names)

        for tab, mine_name in zip(tabs, tab_names):
            with tab:
                # 1. Get the data for the current tab
                df = df_dict[mine_name]
                stats = summary_df.loc[mine_name]

                # 2. Render the professional metrics
                render_metrics(stats, mine_name)

                st.divider()

                # 3. Render the interactive chart
                render_chart_tab(df, mine_name)

                st.divider()

                # 4. Render the detailed data table
                render_data_table(df)

    # Note: The rest of your main function (loading, calculating, calling this function) remains the same.

    # Optional: Show a breakdown of which test failed for the anomalies
    if df['is_anomaly_ANY'].any():
        with st.expander("⚠️ View Anomaly Detection Details"):
            # Display only the anomalous rows with all diagnostic columns
            anomalies = df[df['is_anomaly_ANY']].copy()
            st.dataframe(anomalies)

def main():
    st.set_page_config(page_title="Mine Production Dashboard", layout="wide")

    # --- 1. Sidebar: Define the Parameters ---
    st.sidebar.header("⚙️ Anomaly Settings")
    st.sidebar.write("Adjust test sensitivity below:")

    # This dictionary packages all your settings to send to the functions
    params = {
        'iqr_factor': st.sidebar.slider(
            "IQR Factor (Boxplot)",
            min_value=0.5, max_value=3.0, value=1.5, step=0.1,
            help="Lower = More strict (flags more outliers). Standard is 1.5."
        ),
        'z_threshold': st.sidebar.slider(
            "Z-Score Threshold",
            min_value=1.0, max_value=5.0, value=3.0, step=0.1,
            help="Standard Deviation cutoff. Standard is 3.0."
        ),
        'ma_window': st.sidebar.number_input(
            "Moving Average Window (Days)",
            min_value=2, max_value=30, value=5,
            help="How many previous days to average."
        ),
        'ma_threshold': st.sidebar.slider(
            "Moving Average % Deviation",
            min_value=0.05, max_value=1.0, value=0.20, step=0.05,
            help="Flags if value deviates by X% from the moving average."
        )
    }
    mines_df = get_df_from_spreadsheet()
    df_dict = split_data_into_separate_df(mines_df)
    df_dict['Total'] = generate_total_column(mines_df)
    summary_df = calculate_summary_stats(df_dict)
    apply_anomaly_tests(df_dict, summary_df, params)
    display_dashboard(df_dict, summary_df)

if __name__ == "__main__":
    main()
