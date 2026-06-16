"""Generate financial report PDFs from report JSON."""

import io
from datetime import datetime, timezone
from typing import Any, Dict


def _flatten_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize nested report from generate_financial_report."""
    if report.get("status") == "error":
        return report
    flat: Dict[str, Any] = {"period": report.get("period", "")}
    income = report.get("income") or {}
    expenses = report.get("expenses") or {}
    flat["total_income"] = income.get("total")
    flat["total_expenses"] = expenses.get("total")
    flat["expense_by_category"] = expenses.get("by_category")
    flat["net_worth"] = report.get("net_worth")
    savings = report.get("savings") or {}
    flat["net_savings"] = savings.get("total")
    inv = report.get("investments") or {}
    flat["total_investments"] = inv.get("total_value")
    debt = report.get("debt") or {}
    flat["total_debt"] = debt.get("total")
    flat["savings_rate"] = report.get("savings_rate")
    return flat


def build_financial_report_pdf(report: Dict[str, Any]) -> bytes:
    """Build a simple PDF from generate_financial_report() output."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as exc:
        raise RuntimeError("Install reportlab: pip install reportlab") from exc

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    report = _flatten_report(report)
    period = report.get("period", "")
    title = "TrackExpensio Financial Report"
    if period:
        title += f" ({period})"
    story.append(Paragraph(title, styles["Title"]))
    story.append(
        Paragraph(
            f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 16))

    rows = [["Metric", "Value"]]
    for key, label in [
        ("total_income", "Total income"),
        ("total_expenses", "Total expenses"),
        ("net_savings", "Net savings"),
        ("total_investments", "Investments"),
        ("total_debt", "Total debt"),
        ("net_worth", "Net worth"),
    ]:
        val = report.get(key)
        if val is not None:
            rows.append([label, f"{float(val):,.2f}"])

    if len(rows) > 1:
        table = Table(rows, colWidths=[220, 200])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6366f1")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ]
            )
        )
        story.append(table)

    by_cat = report.get("expense_by_category") or report.get("expenses_by_category")
    if by_cat and isinstance(by_cat, dict):
        story.append(Spacer(1, 20))
        story.append(Paragraph("Expenses by category", styles["Heading2"]))
        cat_rows = [["Category", "Amount"]]
        for cat, amt in by_cat.items():
            cat_rows.append([str(cat), f"{float(amt):,.2f}"])
        ct = Table(cat_rows, colWidths=[220, 200])
        ct.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
        story.append(ct)

    doc.build(story)
    return buffer.getvalue()
