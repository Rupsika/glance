import os
import re
import sys
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def format_inline_markdown(text):
    # Escape HTML tags first
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    
    # Bold: **text** -> <b>text</b>
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    
    # Italic: *text* -> <i>text</i>
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    
    # Inline code: `text` -> <font face="Courier">\1</font>
    text = re.sub(r'`(.*?)`', r'<font face="Courier">\1</font>', text)
    
    # Links: [text](url) -> <a href="url"><font color="#2B6CB0">\1</font></a>
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2"><font color="#2B6CB0">\1</font></a>', text)
    
    return text

def parse_markdown_to_flowables(md_content, styles):
    flowables = []
    lines = md_content.split("\n")
    in_code_block = False
    code_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # Code blocks
        if stripped.startswith("```"):
            if in_code_block:
                in_code_block = False
                code_text = "\n".join(code_lines)
                # Escape code text and format line breaks
                escaped_code = code_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
                flowables.append(Paragraph(escaped_code, styles['CodeStyle']))
                flowables.append(Spacer(1, 10))
                code_lines = []
            else:
                in_code_block = True
            continue
            
        if in_code_block:
            code_lines.append(line)
            continue
            
        # Headers
        if stripped.startswith("# "):
            title = stripped[2:]
            flowables.append(Paragraph(title, styles['Heading1']))
            flowables.append(Spacer(1, 15))
        elif stripped.startswith("## "):
            header = stripped[3:]
            flowables.append(Paragraph(header, styles['Heading2']))
            flowables.append(Spacer(1, 10))
        elif stripped.startswith("### "):
            header = stripped[4:]
            flowables.append(Paragraph(header, styles['Heading3']))
            flowables.append(Spacer(1, 8))
        elif stripped.startswith("---"):
            # Horizontal rule line
            t = Table([[""]], colWidths=[504], rowHeights=[1])
            t.setStyle(TableStyle([
                ('LINEABOVE', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E0")),
                ('TOPPADDING', (0,0), (-1,-1), 0),
                ('BOTTOMPADDING', (0,0), (-1,-1), 0),
            ]))
            flowables.append(t)
            flowables.append(Spacer(1, 12))
        elif stripped.startswith("- ") or stripped.startswith("* "):
            bullet_text = stripped[2:]
            bullet_text = format_inline_markdown(bullet_text)
            flowables.append(Paragraph(f"&bull; {bullet_text}", styles['BulletStyle']))
            flowables.append(Spacer(1, 4))
        elif stripped:
            para_text = format_inline_markdown(line)
            flowables.append(Paragraph(para_text, styles['NormalStyle']))
            flowables.append(Spacer(1, 8))
        else:
            flowables.append(Spacer(1, 4))
            
    return flowables

def add_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.HexColor("#718096"))
    canvas.drawString(54, 30, "Glance ML Internship Assignment | Multimodal Fashion & Context Retrieval")
    canvas.drawRightString(canvas._pagesize[0] - 54, 30, f"Page {doc.page}")
    canvas.restoreState()

def build_pdf(md_path, pdf_path):
    print(f"Reading markdown report from {md_path}...")
    if not os.path.exists(md_path):
        print(f"Error: {md_path} does not exist.")
        return
        
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=54, # 0.75 in (54 points)
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    # Custom styles
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1A365D"), # Dark blue
        spaceAfter=15
    )
    
    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#2B6CB0"), # Muted blue
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    
    h3_style = ParagraphStyle(
        'DocH3',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#4A5568"), # Charcoal
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#2D3748"), # Off-black
        spaceAfter=6
    )
    
    bullet_style = ParagraphStyle(
        'DocBullet',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )
    
    code_style = ParagraphStyle(
        'DocCode',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#2D3748"),
        backColor=colors.HexColor("#EDF2F7"),
        borderColor=colors.HexColor("#CBD5E0"),
        borderWidth=0.5,
        borderPadding=6,
        spaceBefore=4,
        spaceAfter=6
    )
    
    custom_styles = {
        'Heading1': title_style,
        'Heading2': h2_style,
        'Heading3': h3_style,
        'NormalStyle': body_style,
        'BulletStyle': bullet_style,
        'CodeStyle': code_style
    }
    
    flowables = parse_markdown_to_flowables(md_content, custom_styles)
    
    print(f"Generating PDF report to {pdf_path}...")
    doc.build(flowables, onFirstPage=add_footer, onLaterPages=add_footer)
    print("PDF compilation completed successfully!")

if __name__ == "__main__":
    build_pdf("report.md", "submission_report.pdf")
