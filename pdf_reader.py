import PyPDF2
from docx import Document


def read_pdf(file):
    text = ""

    pdf_reader = PyPDF2.PdfReader(file)

    for page in pdf_reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def read_docx(file):
    text = ""

    document = Document(file)

    for paragraph in document.paragraphs:
        text += paragraph.text + "\n"

    return text


def extract_text(uploaded_file):
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        return read_pdf(uploaded_file)

    elif filename.endswith(".docx"):
        return read_docx(uploaded_file)

    else:
        return ""