from typing import *
from googleapiclient.discovery import build
import pandas as pd
import streamlit as st
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import generatePDF

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

    t_crit = stats.t.ppf(1 - alpha / (2 * N), N - 2)
    g_critical = ((N - 1) * np.sqrt(np.square(t_crit))) / (np.sqrt(N) * np.sqrt(N - 2 + np.square(t_crit)))

    return z_scores.abs() > g_critical

def detect_anomalies(df: pd.DataFrame, stats_row: pd.Series, params: Dict[str, Any]) -> pd.DataFrame:
    df['is_anomaly_IQR'] = check_iqr(df, stats_row, params)

    df['z_score'], df['is_anomaly_Zscore'] = check_zscore(df, stats_row, params)

    df['is_anomaly_MA'] = check_moving_average(df, params)

    if len(df) > 2 and stats_row['StdDev'] > 0:
        df['is_anomaly_Grubbs'] = check_grubbs(df, df['z_score'])
    else:
        df['is_anomaly_Grubbs'] = False

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


def render_graph(df: pd.DataFrame, mine_name: str, graph_type: str, poly_degree: int):
    """
    Renders a Plotly graph with:
    1. User-selected chart type (Line, Bar, Area)
    2. Optional Polynomial Trendline (Degree 1-4)
    3. Anomalies highlighted in Red
    """
    fig = go.Figure()

    x_data = df[df.columns[0]]  # Dates
    y_data = df[df.columns[1]]  # Values

    if graph_type == "Line Chart":
        fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name='Production', line=dict(color='#636EFA')))
    elif graph_type == "Bar Chart":
        # We color the bars individually based on anomaly status
        colors = ['#EF553B' if is_anom else '#636EFA' for is_anom in df['is_anomaly_ANY']]
        fig.add_trace(go.Bar(x=x_data, y=y_data, name='Production', marker_color=colors))
    elif graph_type == "Area / Stacked":
        fig.add_trace(go.Scatter(x=x_data, y=y_data, fill='tozeroy', name='Production', line=dict(color='#636EFA')))

    if poly_degree is not None:
        clean_df = df.dropna(subset=[df.columns[1]])
        x_numeric = pd.to_datetime(clean_df[df.columns[0]]).map(pd.Timestamp.toordinal)
        y_numeric = clean_df[df.columns[1]]

        z = np.polyfit(x_numeric, y_numeric, poly_degree)
        p = np.poly1d(z)

        trend_y = p(x_numeric)

        fig.add_trace(go.Scatter(
            x=clean_df[df.columns[0]],
            y=trend_y,
            mode='lines',
            name=f'Trendline (Deg {poly_degree})',
            line=dict(color='orange', width=3, dash='dash')
        ))

    anomalies = df[df['is_anomaly_ANY']]
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies[df.columns[0]],
            y=anomalies[df.columns[1]],
            mode='markers',
            name='Anomaly',
            marker=dict(color='red', size=10, symbol='circle-open', line=dict(width=2))
        ))

    fig.update_layout(
        title=f"{mine_name} Production Analysis",
        xaxis_title="Date",
        yaxis_title="Output",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

def render_metrics(stats: pd.Series, mine_name: str):
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
    st.markdown("### 📈 Production Visualizations")

    # Create two columns for controls to keep UI compact
    col_ctrl1, col_ctrl2 = st.columns([1, 1])

    with col_ctrl1:
        # Chart Type Selector
        graph_type = st.selectbox(
            "Choose Graph Style:",
            ["Line Chart", "Bar Chart", "Area / Stacked"],
            key=f"graph_type_{mine_name}"
        )

    with col_ctrl2:
        poly_choice = st.selectbox(
            "Trendline Complexity:",
            ["None", "Linear (1)", "Quadratic (2)", "Cubic (3)", "Quartic (4)"],
            key=f"poly_{mine_name}"
        )

    degree_map = {
        "None": None,
        "Linear (1)": 1,
        "Quadratic (2)": 2,
        "Cubic (3)": 3,
        "Quartic (4)": 4
    }
    selected_degree = degree_map[poly_choice]

    render_graph(df, mine_name, graph_type, selected_degree)


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


def main():
    params = prepare_streamlit()
    mines_df = get_df_from_spreadsheet()
    df_dict = split_data_into_separate_df(mines_df)
    df_dict['Total'] = generate_total_column(mines_df)
    summary_df = calculate_summary_stats(df_dict)
    apply_anomaly_tests(df_dict, summary_df, params)
    display_dashboard(df_dict, summary_df)
    pdf_generator(df_dict, summary_df)

def pdf_generator(df_dict: Dict[str, pd.DataFrame], summary_df: pd.DataFrame):
    st.divider()
    st.header("📄 Download Report")

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Generate PDF Report"):
            with st.spinner("Generating detailed PDF report..."):
                # Call the function we just created
                pdf_bytes = generatePDF.create_pdf_report(df_dict, summary_df)

                st.download_button(
                    label="📥 Click to Download PDF",
                    data=pdf_bytes,
                    file_name="mine_production_report.pdf",
                    mime="application/pdf"
                )


def prepare_streamlit():
    st.set_page_config(page_title="Mine Production Dashboard", layout="wide")
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
            min_value=0.05, max_value=1.0, value=0.30, step=0.05,
            help="Flags if value deviates by X% from the moving average."
        )
    }
    return params


if __name__ == "__main__":
    main()
