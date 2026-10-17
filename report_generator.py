"""
Antenna Design Report Generator
----------------------------------
Builds a formatted PDF report summarizing:
  - user input parameters
  - predicted antenna dimensions and performance
  - S11 vs frequency plot
  - notes on methodology (for academic/project submission use)

Uses reportlab for PDF layout and matplotlib to render the S11 chart as an embedded image.
"""

import io
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, KeepTogether
)


def make_s11_plot_image(s11_curve, fr_ghz):
    freqs = [p["frequency_GHz"] for p in s11_curve]
    values = [p["S11_dB"] for p in s11_curve]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(freqs, values, color="#1D9E75", linewidth=1.8)
    ax.axvline(fr_ghz, color="#999999", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("S11 (dB)")
    ax.set_title("Return Loss (S11) vs Frequency")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_report(inputs: dict, predictions: dict, s11_curve: list, model_name: str,
                     warnings: list = None) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], fontSize=20, spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle", parent=styles["Normal"], fontSize=10,
        textColor=colors.HexColor("#5F5E5A"), spaceAfter=18
    )
    section_style = ParagraphStyle(
        "SectionHeading", parent=styles["Heading2"], fontSize=13,
        spaceBefore=16, spaceAfter=8, textColor=colors.HexColor("#0F6E56")
    )
    body_style = styles["Normal"]
    note_style = ParagraphStyle(
        "Note", parent=styles["Normal"], fontSize=9,
        textColor=colors.HexColor("#5F5E5A"), spaceAfter=6
    )

    story = []

    story.append(Paragraph("Antenna Design Optimization Report", title_style))
    timestamp = datetime.datetime.now().strftime("%B %d, %Y at %H:%M")
    story.append(Paragraph(f"Generated on {timestamp} &middot; Model: {model_name}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#CCCCCC")))

    story.append(Paragraph("1. Design Inputs", section_style))
    input_table_data = [
        ["Parameter", "Value"],
        ["Operating frequency", f"{inputs['frequency_GHz']} GHz"],
        ["Substrate dielectric constant (\u03b5r)", f"{inputs['dielectric_constant']}"],
        ["Substrate height", f"{inputs['substrate_height_mm']} mm"],
    ]
    input_table = Table(input_table_data, colWidths=[3.2 * inch, 2.5 * inch])
    input_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F6E56")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
    ]))
    story.append(input_table)

    story.append(Paragraph("2. Predicted Antenna Design", section_style))
    pred_table_data = [
        ["Output", "Predicted Value"],
        ["Patch width (W)", f"{predictions['patch_width_mm']} mm"],
        ["Patch length (L)", f"{predictions['patch_length_mm']} mm"],
        ["Gain", f"{predictions['gain_dBi']} dBi"],
        ["Impedance bandwidth", f"{predictions['bandwidth_percent']} %"],
        ["S11 minimum (return loss)", f"{predictions['S11_min_dB']} dB"],
    ]
    pred_table = Table(pred_table_data, colWidths=[3.2 * inch, 2.5 * inch])
    pred_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F6E56")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
    ]))
    story.append(pred_table)

    if warnings:
        story.append(Spacer(1, 8))
        for w in warnings:
            story.append(Paragraph(f"&#9888; {w}", note_style))

    plot_buf = make_s11_plot_image(s11_curve, inputs["frequency_GHz"])
    s11_block = [
        Paragraph("3. Return Loss (S11) Characteristic", section_style),
        Image(plot_buf, width=6 * inch, height=3 * inch),
        Paragraph(
            "The curve above shows the predicted return loss around the target resonant "
            "frequency. The dip depth corresponds to the predicted S11 minimum; the "
            "curve width reflects the predicted impedance bandwidth.",
            note_style
        ),
    ]
    story.append(KeepTogether(s11_block))

    story.append(Paragraph("4. Methodology", section_style))
    story.append(Paragraph(
        "Predictions were generated by a machine learning model trained on a "
        "physics-informed synthetic dataset. The dataset was built using standard "
        "rectangular microstrip patch antenna design equations (transmission line "
        "model, Balanis) to compute patch dimensions, with empirical approximations "
        "used for gain, bandwidth, and S11. Controlled random noise was added to "
        "simulate realistic manufacturing and measurement variation. Several regression "
        f"models were trained and compared; {model_name} was selected as the best "
        "performing model based on validation R<super>2</super> score across all predicted outputs.",
        body_style
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Note: this tool provides a rapid, ML-based design estimate for initial "
        "sizing and exploration. Results should be verified against full-wave "
        "electromagnetic simulation (e.g. HFSS, CST, or MATLAB Antenna Toolbox) "
        "before fabrication.",
        note_style
    ))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()
