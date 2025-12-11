from fpdf import FPDF
import matplotlib.pyplot as plt
import tempfile
import os
import pandas as pd
from typing import Dict


class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Mine Production Analysis Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')


def create_pdf_report(df_dict: Dict[str, pd.DataFrame], summary_df: pd.DataFrame):
    pdf = PDFReport()

    # 1. Summary Section
    pdf.add_page()
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '1. Statistical Summary', 0, 1)

    # Create Summary Table
    pdf.set_font('Arial', '', 10)

    # Define Columns
    cols = ['Mine', 'Mean', 'Median', 'StdDev', 'IQR']
    col_widths = [40, 35, 35, 35, 35]

    # Header Row
    for col, width in zip(cols, col_widths):
        pdf.cell(width, 10, col, 1, 0, 'C')
    pdf.ln()

    # Data Rows
    for mine_name, row in summary_df.iterrows():
        pdf.cell(col_widths[0], 10, str(mine_name), 1)
        pdf.cell(col_widths[1], 10, f"{row['Mean']:.2f}", 1)
        pdf.cell(col_widths[2], 10, f"{row['Median']:.2f}", 1)
        pdf.cell(col_widths[3], 10, f"{row['StdDev']:.2f}", 1)
        pdf.cell(col_widths[4], 10, f"{row['IQR']:.2f}", 1)
        pdf.ln()

    for mine_name, df in df_dict.items():
        if mine_name == 'Total': continue

        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, f'Analysis: {mine_name}', 0, 1)

        # --- Generate Static Chart ---
        plt.figure(figsize=(10, 5))
        plt.plot(df[df.columns[0]], df[df.columns[1]], label='Production', color='blue')

        if 'is_anomaly_ANY' in df.columns:
            anomalies = df[df['is_anomaly_ANY']]
            if not anomalies.empty:
                plt.scatter(anomalies[df.columns[0]], anomalies[df.columns[1]], color='red', label='Anomaly', zorder=5)

        plt.title(f"{mine_name} Production Trend")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45, fontsize=8)
        plt.tight_layout()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            plt.savefig(tmpfile.name)
            plt.close()
            pdf.image(tmpfile.name, x=10, y=None, w=190)
            os.unlink(tmpfile.name)  # Delete temp file

        pdf.ln(5)

        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Detected Anomalies', 0, 1)

        if 'is_anomaly_ANY' in df.columns and not df[df['is_anomaly_ANY']].empty:
            anomalies = df[df['is_anomaly_ANY']]

            pdf.set_font('Arial', 'B', 9)
            # Table Header
            pdf.cell(40, 8, 'Date', 1)
            pdf.cell(30, 8, 'Value', 1)
            pdf.cell(30, 8, 'Z-Score', 1)
            pdf.cell(30, 8, 'Type', 1)
            pdf.ln()

            pdf.set_font('Arial', '', 9)
            for idx, row in anomalies.iterrows():
                # Format Date
                date_val = row[df.columns[0]]
                date_str = str(date_val)[:10]

                # Format Value
                val = f"{row[df.columns[1]]:.2f}"

                # Format Z-Score (handle missing column safely)
                z_val = row.get('z_score', 0)
                z_score = f"{z_val:.2f}"

                # Determine Type
                mean_val = summary_df.loc[mine_name, 'Mean']
                anom_type = "Spike" if row[df.columns[1]] > mean_val else "Drop"

                pdf.cell(40, 8, date_str, 1)
                pdf.cell(30, 8, val, 1)
                pdf.cell(30, 8, z_score, 1)
                pdf.cell(30, 8, anom_type, 1)
                pdf.ln()
        else:
            pdf.set_font('Arial', 'I', 10)
            pdf.cell(0, 10, "No anomalies detected.", 0, 1)

    return bytes(pdf.output(dest='S'))