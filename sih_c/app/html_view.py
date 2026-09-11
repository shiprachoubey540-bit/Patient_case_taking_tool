"""
Renders a case summary dict as a styled HTML page — for demos/judges
instead of raw JSON. No template engine dependency; just an f-string,
kept simple on purpose.
"""

def _badge(confidence: float | None, needs_review: bool) -> str:
    if confidence is None:
        return ""
    color = "#dc2626" if needs_review else "#16a34a"
    label = "Needs review" if needs_review else "Looks good"
    return (
        f'<span style="background:{color};color:white;padding:4px 10px;'
        f'border-radius:12px;font-size:13px;font-weight:600;">'
        f'{label} · {confidence:.0f}% OCR confidence</span>'
    )


def _list_items(items: list[str]) -> str:
    if not items:
        return '<p style="color:#94a3b8;font-style:italic;">None recorded</p>'
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def _medicines_table(medicines: list[dict]) -> str:
    if not medicines:
        return '<p style="color:#94a3b8;font-style:italic;">None recorded</p>'
    rows = ""
    for m in medicines:
        source = m.get("source", "document")
        source_badge = (
            '<span style="background:#e0e7ff;color:#4338ca;padding:2px 8px;'
            'border-radius:8px;font-size:12px;">from document</span>'
            if source == "document"
            else '<span style="background:#fef3c7;color:#92400e;padding:2px 8px;'
                 'border-radius:8px;font-size:12px;">patient reported</span>'
        )
        rows += (
            f"<tr><td>{m.get('raw_line') or '—'}</td>"
            f"<td>{m.get('dosage') or '—'}</td>"
            f"<td>{m.get('frequency') or '—'}</td>"
            f"<td>{source_badge}</td></tr>"
        )
    return (
        '<table><thead><tr><th>Line</th><th>Dosage</th>'
        f"<th>Frequency</th><th>Source</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def _lab_values_table(labs: list[dict]) -> str:
    if not labs:
        return '<p style="color:#94a3b8;font-style:italic;">None recorded</p>'
    rows = "".join(
        f"<tr><td>{l.get('test')}</td><td>{l.get('value')}</td>"
        f"<td>{l.get('unit') or '—'}</td></tr>"
        for l in labs
    )
    return (
        "<table><thead><tr><th>Test</th><th>Value</th><th>Unit</th></tr>"
        f"</thead><tbody>{rows}</tbody></table>"
    )


def render_case_summary_html(summary: dict) -> str:
    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>MediKiosk — Case Summary</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif;
          background: #f8fafc; margin: 0; padding: 40px; color: #1e293b; }}
  .card {{ max-width: 760px; margin: 0 auto; background: white;
           border-radius: 16px; padding: 32px 40px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  .sub {{ color: #64748b; margin-bottom: 20px; }}
  h2 {{ font-size: 15px; text-transform: uppercase; letter-spacing: 0.05em;
        color: #64748b; margin-top: 28px; margin-bottom: 8px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 6px; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #e2e8f0; font-size: 14px; }}
  th {{ color: #64748b; font-weight: 600; font-size: 12px; text-transform: uppercase; }}
  ul {{ margin: 4px 0; padding-left: 20px; }}
  li {{ margin-bottom: 4px; }}
</style>
</head>
<body>
  <div class="card">
    <h1>Patient Case Summary</h1>
    <div class="sub">Source document: {summary.get('source_document') or '—'}</div>
    {_badge(summary.get('document_ocr_confidence'), summary.get('needs_human_review', False))}

    <h2>Chief Complaint</h2>
    <p>{summary.get('chief_complaint') or '<span style="color:#94a3b8;font-style:italic;">Not recorded</span>'}</p>

    <h2>Reported Symptoms {f"({summary.get('duration')})" if summary.get('duration') else ""}</h2>
    {_list_items(summary.get('reported_symptoms', []))}

    <h2>Diagnoses (from document)</h2>
    {_list_items(summary.get('diagnoses', []))}

    <h2>Medicines</h2>
    {_medicines_table(summary.get('medicines', []))}

    <h2>Lab Values</h2>
    {_lab_values_table(summary.get('lab_values', []))}

    <h2>Relevant Dates</h2>
    {_list_items(summary.get('relevant_dates', []))}
  </div>
</body>
</html>
"""
