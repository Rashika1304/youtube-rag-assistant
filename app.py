
import streamlit as st
from rag_backend import process_video, answer_question

st.set_page_config(
    page_title="YouTube RAG Assistant",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 YouTube RAG Assistant")
st.write("Enter a YouTube URL, process the video, and ask questions about its content.")

# Keep the processed video and its FAISS index across Streamlit reruns
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "video_id" not in st.session_state:
    st.session_state.video_id = None


# Section 1: Process the video
youtube_url = st.text_input(
    "YouTube video URL",
    placeholder="https://www.youtube.com/watch?v=..."
)

if st.button("Process Video", type="primary"):
    if not youtube_url.strip():
        st.warning("Please enter a YouTube URL.")
    else:
        try:
            with st.spinner("Fetching transcript and building the search index..."):
                result = process_video(youtube_url)

            st.session_state.vector_store = result["vector_store"]
            st.session_state.video_id = result["video_id"]

            st.success("Video processed successfully! You can now ask questions.")

        except Exception as e:
            st.session_state.vector_store = None
            st.session_state.video_id = None
            st.error(f"Could not process this video: {e}")


# Section 2: Ask questions
st.divider()
st.subheader("💬 Ask a question")

question = st.text_input(
    "Your question",
    placeholder="What are the main points discussed in this video?"
)

if st.button("Ask Question"):
    if st.session_state.vector_store is None:
        st.warning("Please process a video first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Searching the transcript and generating an answer..."):
                answer, sources = answer_question(
                    st.session_state.vector_store,
                    question
                )

            st.subheader("🤖 Answer")
            st.write(answer)

            st.subheader("📚 Retrieved Sources")

            for i, doc in enumerate(sources, start=1):
                with st.expander(f"Source {i}"):
                    st.write(doc.page_content)
                    st.caption(f"Video ID: {doc.metadata.get('video_id', 'Unknown')}")
                    st.caption(f"Chunk: {doc.metadata.get('chunk_index', 'Unknown')}")

        except Exception as e:
            st.error(f"Could not generate an answer: {e}")
