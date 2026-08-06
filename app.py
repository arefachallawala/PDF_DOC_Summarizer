import math
from collections import Counter

import streamlit as st
from nltk.tokenize import sent_tokenize, word_tokenize

from pdf_reader import extract_text
from preprocessing import preprocess_text
from summarizer import generate_summary
from qa import answer_question


# ----------------------------------------
# Page Configuration
# ----------------------------------------

st.set_page_config(
    page_title="DocSense AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ----------------------------------------
# Load CSS
# ----------------------------------------

with open("style.css") as css:
    st.markdown(
        f"<style>{css.read()}</style>",
        unsafe_allow_html=True
    )


# ----------------------------------------
# Session State
# ----------------------------------------

default_state = {
    "uploaded": False,
    "document_text": "",
    "cleaned_text": "",
    "summary": "",
    "pages": 0,
    "words": 0,
    "sentences": 0,
    "reading_time": 0,
    "chat_history": []
}

for key, value in default_state.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ----------------------------------------
# Sidebar
# ----------------------------------------

st.sidebar.title("DocSense AI")

st.sidebar.markdown("---")

page = st.sidebar.radio(

    "Navigation",

    [

        "Dashboard",

        "Summary",

        "Chat",

        "Analytics"

    ]

)

st.sidebar.markdown("---")

st.sidebar.write("Application")

st.sidebar.caption(
    "Document Intelligence System"
)

st.sidebar.markdown("---")

st.sidebar.write("Supported Formats")

st.sidebar.write("• PDF")

st.sidebar.write("• DOCX")

st.sidebar.markdown("---")

st.sidebar.caption("NLP Project")


# ----------------------------------------
# Main Header
# ----------------------------------------

st.title("DocSense AI")

st.write(
    "A document intelligence platform for summarizing, "
    "analyzing and interacting with PDF and DOCX files."
)

st.divider()
# ----------------------------------------
# Dashboard
# ----------------------------------------

if page == "Dashboard":

    st.subheader("Dashboard")

    st.write(
        """
        Upload a PDF or DOCX document to generate
        summaries, extract keywords and interact
        with the document.
        """
    )

    uploaded_file = st.file_uploader(
        "Select Document",
        type=["pdf", "docx"]
    )

    if uploaded_file is not None:

        progress = st.progress(0)

        with st.spinner("Extracting document..."):

            progress.progress(20)

            document_text = extract_text(
                uploaded_file
            )

        with st.spinner("Cleaning text..."):

            progress.progress(50)

            cleaned_text = preprocess_text(
                document_text
            )

        with st.spinner("Generating summary..."):

            progress.progress(80)

            summary = generate_summary(
                document_text
            )

        progress.progress(100)

        st.session_state.uploaded = True

        st.session_state.document_text = document_text

        st.session_state.cleaned_text = cleaned_text

        st.session_state.summary = summary

        pages = 0

        if uploaded_file.name.lower().endswith(".pdf"):

            import PyPDF2

            pdf = PyPDF2.PdfReader(uploaded_file)

            pages = len(pdf.pages)

        words = len(
            word_tokenize(document_text)
        )

        sentences = len(
            sent_tokenize(document_text)
        )

        reading_time = math.ceil(
            words / 200
        )

        st.session_state.pages = pages

        st.session_state.words = words

        st.session_state.sentences = sentences

        st.session_state.reading_time = reading_time

        st.success("Document processed successfully.")

        st.divider()

        st.subheader("Document Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Pages",
                pages
            )

        with col2:

            st.metric(
                "Words",
                words
            )

        with col3:

            st.metric(
                "Sentences",
                sentences
            )

        with col4:

            st.metric(
                "Reading Time",
                f"{reading_time} min"
            )

        st.divider()

        st.subheader("Preview")

        st.text_area(

            label="",

            value=document_text[:2000],

            height=350,

            disabled=True

        )

    else:

        st.info(
            "Upload a document to begin."
        )
  # ----------------------------------------
# Summary
# ----------------------------------------

elif page == "Summary":

    st.subheader("Document Summary")

    if st.session_state.uploaded:

        summary_style = st.selectbox(
            "Summary Style",
            [
                "Short",
                "Medium",
                "Detailed",
                "Bullet Points"
            ]
        )

        summary = generate_summary(
            st.session_state.document_text,
            summary_style
        )

        st.markdown("### Generated Summary")

        if summary_style == "Bullet Points":

            for point in summary.split("\n\n"):

                if point.strip():

                    clean_point = point.replace(
                        "- ",
                        "",
                        1
                    )

                    st.markdown(
                        f"- {clean_point}"
                    )

        else:

            st.write(summary)

        st.divider()

        summary_words = len(
            word_tokenize(summary)
        )

        summary_sentences = len(
            sent_tokenize(summary)
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Summary Words",
                summary_words
            )

        with col2:

            st.metric(
                "Summary Sentences",
                summary_sentences
            )

        st.divider()

        st.subheader("Top Keywords")

        frequency = Counter(
            word_tokenize(
                st.session_state.cleaned_text
            )
        )

        keywords = frequency.most_common(10)

        keyword_cols = st.columns(5)

        for i, (word, count) in enumerate(keywords):

            keyword_cols[i % 5].metric(
                word,
                count
            )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            st.download_button(
                "Download Summary",
                summary,
                file_name="document_summary.txt",
                mime="text/plain",
                use_container_width=True
            )

        with col2:

            if st.button(
                "Read Summary",
                use_container_width=True
            ):

                st.info(
                    "Audio summary will be available here."
                )

    else:

        st.warning(
            "Upload a document from the Dashboard first."
        )      # ----------------------------------------
# Chat
# ----------------------------------------

elif page == "Chat":

    st.subheader("Document Assistant")

    if st.session_state.uploaded:

        if "messages" not in st.session_state:

            st.session_state.messages = []

        if len(st.session_state.messages) == 0:

            st.info(
                "Ask questions about your uploaded document."
            )

        for message in st.session_state.messages:

            with st.chat_message(message["role"]):

                st.markdown(message["content"])

        prompt = st.chat_input(
            "Ask something about your document..."
        )

        if prompt:

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": prompt
                }
            )

            with st.chat_message("user"):

                st.markdown(prompt)

            with st.chat_message("assistant"):

                with st.spinner("Thinking..."):

                    answer = answer_question(

                        st.session_state.document_text,

                        prompt

                    )

                    st.markdown(answer)

            st.session_state.messages.append(

                {

                    "role": "assistant",

                    "content": answer

                }

            )

        st.divider()

        st.subheader("Suggested Questions")

        col1, col2 = st.columns(2)

        with col1:

            if st.button("Summarize the document"):

                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": "Summarize the document."
                    }
                )

                reply = answer_question(
                    st.session_state.document_text,
                    "Summarize the document."
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": reply
                    }
                )

                st.rerun()

            if st.button("What is the main topic?"):

                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": "What is the main topic?"
                    }
                )

                reply = answer_question(
                    st.session_state.document_text,
                    "What is the main topic?"
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": reply
                    }
                )

                st.rerun()

        with col2:

            if st.button("Explain the conclusion"):

                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": "Explain the conclusion."
                    }
                )

                reply = answer_question(
                    st.session_state.document_text,
                    "Explain the conclusion."
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": reply
                    }
                )

                st.rerun()

            if st.button("List important keywords"):

                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": "List important keywords."
                    }
                )

                reply = answer_question(
                    st.session_state.document_text,
                    "List important keywords."
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": reply
                    }
                )

                st.rerun()

    else:

        st.warning(
            "Upload a document first."
        )