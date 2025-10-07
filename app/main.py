import streamlit as st
import os
import sys
import tempfile
import logging

# Ensure imports from local app folder work inside Docker
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingest import SmartRAGAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize persistent objects in session_state
if "agent" not in st.session_state:
    # instantiate agent lazily — this initializes embeddings which can be heavy
    try:
        st.session_state.agent = SmartRAGAgent()
    except Exception as e:
        st.error(f"Failed to initialize agent: {e}")
        st.stop()

if "system" not in st.session_state:
    st.session_state.system = None
if "messages" not in st.session_state:
    st.session_state.messages = []


def main():
    st.set_page_config(page_title="PDF Q&A Assistant", layout="wide")
    st.title("🔍 PDF Q&A Assistant")
    st.markdown("Upload a PDF and ask questions about its content!")

    with st.sidebar:
        st.header("📄 Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", help="Upload a PDF document to analyze")

        if uploaded_file is not None:
            with st.spinner("Processing PDF..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                try:
                    # Use the SmartRAGAgent to process the file (returns a system dict)
                    system = st.session_state.agent.process_pdf(tmp_path)
                    if system:
                        st.session_state.system = system
                        st.success("✅ PDF processed successfully!")
                        st.info(f"Document: {uploaded_file.name}")

                        # Optionally persist vectorstore to /app/vectorstore (host-mounted volume)
                        try:
                            if "vectorstore" in system:
                                save_dir = "/app/vectorstore"
                                os.makedirs(save_dir, exist_ok=True)
                                system["vectorstore"].save_local(save_dir)
                                logger.info(f"Vectorstore saved to {save_dir}")
                        except Exception as e:
                            logger.warning(f"Could not save vectorstore: {e}")
                    else:
                        st.error("❌ Failed to process PDF. The PDF may be malformed or unsupported.")
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

    # Show instructions if no system loaded
    if not st.session_state.system:
        st.info("👆 Please upload a PDF document to start asking questions.")
        return

    # Show some basic document intelligence
    intel = st.session_state.system.get("intelligence", {})
    struct = st.session_state.system.get("structure", {})

    with st.expander("📑 Document summary", expanded=False):
        if intel:
            titles = intel.get("titles", [])
            if titles:
                st.write("**Detected titles (top candidates):**")
                for t, src in titles[:5]:
                    st.write(f"- {t}  _(source: {src})_")
            if intel.get("metadata"):
                st.write("**Metadata (sample):**")
                for k, v in intel.get("metadata", {}).items():
                    if v:
                        st.write(f"- **{k}**: {v[:3]}")
        if struct:
            st.write("**Structure**")
            st.write(f"- TOC sections: {len(struct.get('toc_sections', []))}")
            st.write(f"- Body sections: {len(struct.get('body_sections', []))}")
            if struct.get("structure_issues"):
                st.write("**Structure issues:**")
                for it in struct["structure_issues"]:
                    st.write(f"- {it}")

    # Show previous conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input("Ask a question about your PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = st.session_state.agent.query(prompt, st.session_state.system)
                    st.markdown(answer)

                    # Also show top source excerpts for transparency
                    try:
                        vs = st.session_state.system.get("vectorstore")
                        if vs:
                            top_docs = vs.similarity_search(prompt, k=3)
                            if top_docs:
                                with st.expander("📚 Top source excerpts"):
                                    for i, d in enumerate(top_docs, 1):
                                        page = d.metadata.get("page", "unknown")
                                        st.markdown(f"**Source {i}** (Page {page})")
                                        st.markdown((d.page_content or "")[:800] + ("..." if len((d.page_content or "")) > 800 else ""))
                    except Exception as e:
                        logger.warning(f"Could not fetch source excerpts: {e}")

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    err = f"Error generating response: {e}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})

    # Clear chat
    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()


if __name__ == "__main__":
    main()
