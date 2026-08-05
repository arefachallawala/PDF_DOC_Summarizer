import streamlit as st
import pandas as pd
import math
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize

from pdf_reader import extract_text
from preprocessing import preprocess_text
from summarizer import generate_summary
from qa import answer_question


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="DocSense AI",
    page_icon="📄",
    layout="wide"
)


# -----------------------------
# Load CSS
# -----------------------------

with open("style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)


# -----------------------------
# Header
# -----------------------------

st.markdown("""
<h1 class='main-title'>📄 DocSense AI</h1>
<p class='subtitle'>
Your Intelligent Document Assistant
</p>
""", unsafe_allow_html=True)


st.divider()


# -----------------------------
# Tabs
# -----------------------------

tab1, tab2, tab3 = st.tabs([
    "📂 Upload",
    "📝 Summary",
    "💬 Ask AI"
])


# -----------------------------
# Upload Tab
# -----------------------------

with tab1:

    st.markdown(
        "<h2 class='section-title'>Upload Document</h2>",
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload a PDF or DOCX file",
        type=["pdf", "docx"]
    )

    if uploaded_file:

        with st.spinner("Reading document..."):

            document_text = extract_text(uploaded_file)

            cleaned_text = preprocess_text(document_text)

            summary = generate_summary(document_text)

        st.success("Document uploaded successfully!")

        pages = 0

        if uploaded_file.name.endswith(".pdf"):
            import PyPDF2
            pdf = PyPDF2.PdfReader(uploaded_file)
            pages = len(pdf.pages)

        words = len(word_tokenize(document_text))

        sentences = len(sent_tokenize(document_text))

        reading_time = math.ceil(words / 200)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="stat-box">
            <div class="stat-number">{pages}</div>
            <div class="stat-label">Pages</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="stat-box">
            <div class="stat-number">{words}</div>
            <div class="stat-label">Words</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="stat-box">
            <div class="stat-number">{sentences}</div>
            <div class="stat-label">Sentences</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="stat-box">
            <div class="stat-number">{reading_time} min</div>
            <div class="stat-label">Reading Time</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        st.subheader("📖 Document Preview")

        preview = document_text[:1500]

        st.markdown(
            f"""
            <div class="card">
            {preview}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.info("Upload a PDF or DOCX file to begin.")

# -----------------------------
# Summary Tab
# -----------------------------

with tab2:

    st.markdown(
        "<h2 class='section-title'>AI Summary</h2>",
        unsafe_allow_html=True
    )

    if "uploaded_file" in locals() and uploaded_file:

        st.markdown(
            f"""
            <div class="summary-box">
            {summary}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            "<h2 class='section-title'>Top Keywords</h2>",
            unsafe_allow_html=True
        )

        words = word_tokenize(cleaned_text)

        word_frequency = Counter(words)

        common_words = word_frequency.most_common(10)

        keyword_html = ""

        for word, count in common_words:
            keyword_html += f"<span class='keyword'>{word}</span> "

        st.markdown(
            keyword_html,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            "<h2 class='section-title'>Download Summary</h2>",
            unsafe_allow_html=True
        )

        st.download_button(
            label="📥 Download Summary",
            data=summary,
            file_name="summary.txt",
            mime="text/plain"
        )

    else:

        st.info("Please upload a document first.")

# -----------------------------
# Ask AI Tab
# -----------------------------

with tab3:

    st.markdown(
        "<h2 class='section-title'>Ask AI</h2>",
        unsafe_allow_html=True
    )

    if "uploaded_file" in locals() and uploaded_file:

        question = st.text_input(
            "Ask a question about your document"
        )

        if st.button("🔍 Ask"):

            if question.strip() == "":
                st.warning("Please enter a question.")

            else:

                with st.spinner("Finding answer..."):

                    answer = answer_question(
                        document_text,
                        question
                    )

                st.markdown(
                    f"""
                    <div class="answer-box">
                    {answer}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    else:

        st.info("Please upload a document first.")


# -----------------------------
# Footer
# -----------------------------

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="footer">
        <b>DocSense AI</b><br>
        Developed for NLP Innovation Project<br><br>

        📄 Upload • 📝 Summarize • 💬 Ask • 📥 Download
    </div>
    """,
    unsafe_allow_html=True
)