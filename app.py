import io
import os
import re
import tempfile
import uuid
import warnings
from datetime import datetime

import google.genai as genai_new
import markdown as md_lib
import streamlit as st
from dotenv import load_dotenv

# langchain-community deprecation: suppress BEFORE importing from it
warnings.filterwarnings(
    "ignore",
    message="`langchain-community` is being sunset",
    category=DeprecationWarning,
)

# Track: https://github.com/langchain-ai/langchain-community/issues/674
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf2image import convert_from_bytes
from PyPDF2 import PdfReader
from rank_bm25 import BM25Okapi

load_dotenv()

try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error(
            "GOOGLE_API_KEY not found in environment variables", icon=":material/key:"
        )
        st.stop()
except Exception as e:
    st.error(f"Error loading API key: {str(e)}", icon=":material/error:")
    st.stop()


@st.cache_resource
def get_genai_client():
    return genai_new.Client(api_key=os.getenv("GOOGLE_API_KEY"))


@st.cache_resource
def get_embeddings():
    """Get cached local embeddings model (no API key needed, zero quota cost)"""
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


@st.cache_resource
def get_llm_model():
    """Get cached LLM model"""
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"), temperature=0.3
    )


def init_session_state():
    """Initialize session state variables"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "generated_content" not in st.session_state:
        st.session_state.generated_content = {}
    if "processed_docs" not in st.session_state:
        st.session_state.processed_docs = {}  # {doc_id: {name, pages, size, chunk_count}}
    if "all_chunks" not in st.session_state:
        st.session_state.all_chunks = []
    if "all_metadatas" not in st.session_state:
        st.session_state.all_metadatas = []
    if "faiss_ready" not in st.session_state:
        st.session_state.faiss_ready = False
    if "selected_source" not in st.session_state:
        st.session_state.selected_source = "All Documents"
    if "bm25_index" not in st.session_state:
        st.session_state.bm25_index = None
    if "bm25_ready" not in st.session_state:
        st.session_state.bm25_ready = False
    if "messages" not in st.session_state:
        st.session_state.messages = []


def _ocr_page_with_gemini(image, model_name):
    client = get_genai_client()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        try:
            image.save(tmp.name, "PNG")
            uploaded = client.files.upload(file=tmp.name)
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    uploaded,
                    "Extract all text from this page accurately. Return the full text content.",
                ],
            )
            text = response.text if response and response.text else ""
            return text
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


def get_pdf_text(pdf_docs):
    try:
        if not pdf_docs:
            st.warning(
                "Please upload at least one PDF file", icon=":material/upload_file:"
            )
            return {}, {}

        docs_text = {}
        docs_info = {}
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

        for pdf in pdf_docs:
            try:
                # Phase 1: try PyPDF2 text extraction
                pdf.seek(0)
                pdf_reader = PdfReader(pdf)
                text = ""
                page_count = len(pdf_reader.pages)
                success_pages = 0
                ocr_used = False

                for _, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text += page_text + "\n\n"
                            success_pages += 1
                    except Exception:
                        continue

                # Phase 2: if text is very sparse, try OCR
                avg_chars = len(text.strip()) / max(page_count, 1)
                if avg_chars < 15 and page_count > 0:
                    st.info(
                        f"🔍 Detected scanned PDF: **{pdf.name}**. Running OCR via Gemini Vision..."
                    )
                    pdf.seek(0)
                    try:
                        images = convert_from_bytes(pdf.read(), dpi=200)
                        ocr_text = ""
                        success_pages = (
                            0  # reset — OCR replaces PyPDF2 extraction entirely
                        )
                        ocr_progress = st.progress(0, text=f"OCR: {pdf.name}")
                        for i, img in enumerate(images):
                            ocr_progress.progress(
                                (i + 1) / len(images),
                                text=f"OCR page {i + 1}/{len(images)}",
                            )
                            page_text = _ocr_page_with_gemini(img, model_name)
                            if page_text.strip():
                                ocr_text += page_text + "\n\n"
                                success_pages += 1
                        ocr_progress.empty()

                        if ocr_text.strip():
                            text = ocr_text
                            ocr_used = True
                            st.toast(
                                f"OCR completed for {pdf.name} — {len(images)} pages"
                            )
                        else:
                            st.warning(
                                f"OCR produced no text for **{pdf.name}**",
                                icon=":material/warning:",
                            )
                    except Exception as ocr_e:
                        st.warning(
                            f"OCR failed for **{pdf.name}**: {str(ocr_e)}",
                            icon=":material/warning:",
                        )

                if text.strip():
                    if pdf.name in docs_text:
                        st.warning(
                            f"Multiple files named '{pdf.name}' uploaded — only the last one will be processed",
                            icon=":material/warning:",
                        )
                    docs_text[pdf.name] = text
                    docs_info[pdf.name] = {
                        "total_pages": page_count,
                        "extracted_pages": success_pages,
                        "size_kb": round(len(text) / 1024, 1),
                        "ocr": ocr_used,
                    }
                else:
                    st.warning(f"⚠️ No text could be extracted from **{pdf.name}**")

            except Exception as e:
                st.error(f"❌ Error reading PDF {pdf.name}: {str(e)}")
                continue

        if not docs_text:
            st.error(
                "No text could be extracted from the uploaded PDFs",
                icon=":material/error:",
            )
            return {}, {}

        return docs_text, docs_info
    except Exception as e:
        st.error(f"❌ Error processing PDFs: {str(e)}")
        return {}, {}


def get_text_chunks_with_metadata(docs_text):
    """Split per-document text into chunks with source metadata"""
    try:
        if not docs_text:
            return [], []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000, chunk_overlap=1000
        )

        all_chunks = []
        all_metadatas = []

        for source_name, text in docs_text.items():
            chunks = text_splitter.split_text(text)
            doc_id = str(uuid.uuid4())[:8]
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadatas.append({"source": source_name, "doc_id": doc_id})

        if not all_chunks:
            st.error(
                "Could not create text chunks from the extracted text",
                icon=":material/error:",
            )
            return [], []

        return all_chunks, all_metadatas
    except Exception as e:
        st.error(f"❌ Error creating text chunks: {str(e)}")
        return [], []


def get_vector_store(text_chunks, metadatas):
    """Create FAISS vector store with metadata"""
    try:
        if not text_chunks:
            st.error(
                "No text chunks available to create vector store",
                icon=":material/error:",
            )
            return None

        embeddings = get_embeddings()
        vector_store = FAISS.from_texts(
            text_chunks, embedding=embeddings, metadatas=metadatas
        )
        return vector_store
    except Exception as e:
        st.error(f"❌ Error creating vector store: {str(e)}")
        return None


def rebuild_faiss_index():
    """Rebuild FAISS index and BM25 index from session state"""
    try:
        if not st.session_state.all_chunks:
            st.session_state.faiss_ready = False
            st.session_state.bm25_ready = False
            return False

        vstore = get_vector_store(
            st.session_state.all_chunks, st.session_state.all_metadatas
        )
        if vstore:
            vstore.save_local("faiss_index")
            st.session_state.faiss_ready = True
            if not _build_bm25_index():
                st.warning(
                    "Keyword search index failed (hybrid search degraded to vector-only)",
                    icon=":material/warning:",
                )
            return True
        st.session_state.faiss_ready = False
        return False
    except Exception as e:
        st.error(f"❌ Error rebuilding index: {str(e)}")
        st.session_state.faiss_ready = False
        return False


def remove_document(doc_name):
    """Remove a document and its chunks from the index"""
    try:
        remaining_chunks = []
        remaining_metas = []
        removed_count = 0

        for chunk, meta in zip(
            st.session_state.all_chunks, st.session_state.all_metadatas
        ):
            if meta["source"] != doc_name:
                remaining_chunks.append(chunk)
                remaining_metas.append(meta)
            else:
                removed_count += 1

        st.session_state.all_chunks = remaining_chunks
        st.session_state.all_metadatas = remaining_metas

        if doc_name in st.session_state.processed_docs:
            del st.session_state.processed_docs[doc_name]

        if st.session_state.all_chunks:
            rebuild_faiss_index()
        else:
            st.session_state.faiss_ready = False
            st.session_state.bm25_ready = False
            st.session_state.bm25_index = None
            for f in ["faiss_index/index.faiss", "faiss_index/index.pkl"]:
                if os.path.exists(f):
                    os.remove(f)

        return removed_count
    except Exception as e:
        st.error(f"❌ Error removing document: {str(e)}")
        return 0


def get_conversational_chain():
    """Create conversational chain with error handling and memory support"""
    try:
        prompt_template = """
        You are a helpful assistant answering questions based on PDF documents.
        Use the provided context to answer the question as accurately as possible.
        If the answer is not in the provided context, say "Answer is not available in the context".
        Do not provide wrong answers.

        {conversation_history}

        Context from the PDFs:
        {context}

        Question: {question}

        Answer:
        """

        model = get_llm_model()
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["conversation_history", "context", "question"],
        )

        return model, prompt
    except Exception as e:
        st.error(f"❌ Error creating conversational chain: {str(e)}")
        return None, None


def _extract_text(content):
    """Normalize response content to string (handles list/str from newer LLM SDKs)"""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def create_download_link(content, filename, label):
    """Create a download button for content"""
    text = _extract_text(content)
    buffer = io.BytesIO()
    buffer.write(text.encode())
    buffer.seek(0)

    st.download_button(label=label, data=buffer, file_name=filename, mime="text/plain")


def _load_vector_store():
    """Load FAISS index or return None"""
    try:
        if not os.path.exists("faiss_index"):
            return None
        embeddings = get_embeddings()
        return FAISS.load_local(
            "faiss_index", embeddings, allow_dangerous_deserialization=True
        )
    except Exception:
        return None


def _get_source_filter(selected_doc):
    """Build filter dict for similarity search based on selected document"""
    if selected_doc and selected_doc != "All Documents":
        return {"source": selected_doc}
    return None


def _build_bm25_index():
    """Build BM25 keyword index from current chunks"""
    try:
        if not st.session_state.all_chunks:
            st.session_state.bm25_index = None
            st.session_state.bm25_ready = False
            return False

        tokenized_corpus = [chunk.split() for chunk in st.session_state.all_chunks]
        st.session_state.bm25_index = BM25Okapi(tokenized_corpus)
        st.session_state.bm25_ready = True
        return True
    except Exception as e:
        st.warning(
            f"⚠️ Keyword index build failed (falling back to vector-only): {str(e)}"
        )
        st.session_state.bm25_index = None
        st.session_state.bm25_ready = False
        return False


def _hybrid_search(query, filter=None, k=5):
    """Hybrid search: BM25 keyword + FAISS vector with RRF fusion.
    Returns combined results ranked by reciprocal rank fusion.
    """
    try:
        new_db = _load_vector_store()
        if not new_db:
            return None

        # 1. FAISS vector search (fetch more for reranking)
        vector_docs = new_db.similarity_search(query, k=k * 2, filter=filter)

        # 2. BM25 keyword search
        bm25_results = []
        if st.session_state.bm25_index and st.session_state.all_chunks:
            tokenized_query = query.split()
            scores = st.session_state.bm25_index.get_scores(tokenized_query)

            # Apply source filter for BM25
            if filter and "source" in filter:
                # Zero out scores from other documents
                for i, meta in enumerate(st.session_state.all_metadatas):
                    if i < len(scores) and meta.get("source") != filter["source"]:
                        scores[i] = -1.0

            # Get top BM25 results
            top_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[: k * 2]
            top_indices = [i for i in top_indices if scores[i] > 0]

            if top_indices:
                from langchain_core.documents import Document

                for idx in top_indices:
                    meta = (
                        st.session_state.all_metadatas[idx]
                        if idx < len(st.session_state.all_metadatas)
                        else {}
                    )
                    bm25_results.append(
                        Document(
                            page_content=st.session_state.all_chunks[idx], metadata=meta
                        )
                    )

        # 3. RRF Fusion (k=60 is standard constant)
        rrf_constant = 60
        fused = {}

        for rank, doc in enumerate(vector_docs):
            content_key = doc.page_content[:150]  # prefix as dedup key
            rank_score = 1.0 / (rrf_constant + rank)
            fused[content_key] = (doc, fused.get(content_key, (doc, 0))[1] + rank_score)

        for rank, doc in enumerate(bm25_results):
            content_key = doc.page_content[:150]
            rank_score = 1.0 / (rrf_constant + rank)
            fused[content_key] = (doc, fused.get(content_key, (doc, 0))[1] + rank_score)

        sorted_results = sorted(fused.values(), key=lambda x: x[1], reverse=True)
        docs = [doc for doc, _ in sorted_results[: k * 2]]

        if not docs:
            return new_db.similarity_search(query, k=k, filter=filter)

        return _rerank_docs(query, docs, k=k)

    except Exception as e:
        st.warning(f"⚠️ Hybrid search error, falling back to vector: {str(e)}")
        try:
            new_db = _load_vector_store()
            if new_db:
                return new_db.similarity_search(query, k=k, filter=filter)
        except Exception:
            pass
        return None


def _rerank_docs(query, docs, k=5):
    if not docs or len(docs) <= k:
        return docs[:k]
    try:
        client = get_genai_client()
        scored = []
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        for doc in docs:
            p = f"""Rate relevance 1-5. Respond with only a number.

Question: {query}
Document: {doc.page_content[:800]}
Score:"""
            try:
                resp = client.models.generate_content(model=model_name, contents=p)
                raw = resp.text.strip() if resp and resp.text else "3"
                score = max(
                    1, min(5, int("".join(c for c in raw if c.isdigit()) or "3"))
                )
            except Exception:
                score = 3
            scored.append((doc, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [d for d, _ in scored[:k]]
    except Exception as e:
        st.warning(f"⚠️ Reranking unavailable, using hybrid order: {str(e)}")
        return docs[:k]


def _md_to_html(text):
    """Convert markdown text to HTML with MathJax support."""
    math_map = {}

    def _protect(m):
        k = f"\x00M{len(math_map)}\x00"
        math_map[k] = m.group(1)
        return k

    def _restore(t):
        for k, v in math_map.items():
            s = v.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            t = t.replace(k, f'<span class="math inline">\\({s}\\)</span>')
        return t

    protected = re.sub(r"\$(.+?)\$", _protect, text)
    html = md_lib.markdown(protected, extensions=["extra"])
    return _restore(html)


# Shared base style for all streamed/inline content boxes
_STYLE_BASE = (
    "background:var(--st-secondary-background-color);"
    "border:1px solid var(--st-border-color);"
    "border-radius:var(--st-base-radius);"
    "white-space:pre-wrap;"
    "font-size:1.1rem;"
)


def _render_cached(text, horizontal=False):
    """Render cached content in the standard styled output box (no streaming)."""
    if not text:
        return
    html = _md_to_html(text)
    hs = "column-count:2;column-gap:32px;" if horizontal else ""
    st.html(
        f'<div style="{hs}{_STYLE_BASE}padding:1.25rem;line-height:1.7;overflow-wrap:break-word;">{html}</div>'
    )


def _stream_into_component(model, prompt, horizontal=False):
    """Stream model response into a single in-place DOM element."""
    try:
        stream = model.stream(prompt)
        full_text = ""
        hs = "column-count:2;column-gap:32px;" if horizontal else ""
        box = st.empty()

        # Shared CSS for both thinking indicator and stream cursor
        stream_css = (
            "<style>"
            ".thinking-dot{display:inline-block;width:8px;height:8px;border-radius:50%;"
            "background-color:#CDEA12;margin:0 3px;animation:think-dots 1.4s infinite ease-in-out}"
            ".thinking-dot:nth-child(1){animation-delay:0s}"
            ".thinking-dot:nth-child(2){animation-delay:0.2s}"
            ".thinking-dot:nth-child(3){animation-delay:0.4s}"
            "@keyframes think-dots{0%,55%,100%{opacity:0.2;transform:translateY(0)}"
            "28%{opacity:1;transform:translateY(-4px)}}"
            ".stream-cursor{animation:blink-cursor 0.8s step-end infinite;"
            "color:var(--st-text-color);opacity:0.7}"
            "@keyframes blink-cursor{50%{opacity:0}}"
            "</style>"
        )

        # Show thinking indicator immediately while waiting for first word
        got_word = False
        box.markdown(
            f"{stream_css}"
            f'<div style="{hs}text-align:center;padding:2.5rem 2rem;color:var(--st-text-color);'
            f'font-size:1.3rem;font-weight:500;">'
            '<span class="thinking-dot"></span>'
            '<span class="thinking-dot"></span>'
            '<span class="thinking-dot"></span>'
            " Thinking</div>",
            unsafe_allow_html=True,
        )

        for chunk in stream:
            if chunk and hasattr(chunk, "content") and chunk.content:
                ct = (
                    _extract_text(chunk.content)
                    if isinstance(chunk.content, list)
                    else str(chunk.content)
                )
                full_text += ct
                if got_word or full_text.strip():
                    if not got_word:
                        got_word = True
                    html = _md_to_html(full_text + "\n\n\u200b")
                    box.markdown(
                        f'<div style="{hs}{_STYLE_BASE}padding:1.25rem;line-height:1.7;overflow-wrap:break-word;">'
                        f'{html}<span class="stream-cursor"></span></div>',
                        unsafe_allow_html=True,
                    )

        if full_text:
            html = _md_to_html(full_text)
            box.markdown(
                f'<div style="{hs}{_STYLE_BASE}padding:1.25rem;line-height:1.7;overflow-wrap:break-word;">{html}</div>',
                unsafe_allow_html=True,
            )
        else:
            box.markdown(
                f'<div style="{_STYLE_BASE}padding:1.25rem;color:var(--st-text-color);">*No response generated*</div>',
                unsafe_allow_html=True,
            )

        return full_text
    except Exception as e:
        st.error(f"❌ Streaming error: {str(e)}")
        return None


def _get_conversation_history(n=3):
    """Format last n Q&A exchanges as conversation context string."""
    if not st.session_state.chat_history:
        return ""
    recent = st.session_state.chat_history[-n:]
    lines = ["### Previous conversation context:"]
    for h in recent:
        lines.append(f"**User:** {h['question']}")
        lines.append(f"**Assistant:** {h['answer']}")
    return "\n".join(lines) + "\n\n"


def _run_feature(
    source_filter,
    search_query,
    search_k,
    heading,
    content_key,
    toast_msg,
    prompt_template,
    pre_stream=None,
    post_stream=None,
):
    """DRY helper for feature functions: load → search → model → prompt → stream → save."""
    new_db = _load_vector_store()
    if not new_db:
        st.error(
            "Please upload and process PDF files first", icon=":material/description:"
        )
        return None

    filter_param = _get_source_filter(source_filter)
    with st.spinner(":material/search: Searching your documents..."):
        docs = _hybrid_search(search_query, k=search_k, filter=filter_param)
        if not docs:
            label = (
                f" in **{source_filter}**"
                if source_filter and source_filter != "All Documents"
                else ""
            )
            st.warning(f"⚠️ No content found{label}")
            return None
        model = get_llm_model()
        if not model:
            return None
        sources = set(d.metadata["source"] for d in docs if hasattr(d, "metadata"))
        context = "\n\n".join([doc.page_content for doc in docs])

    prompt = prompt_template.format(context=context)

    st.toast(toast_msg)
    if sources:
        st.caption(f"Based on: {', '.join(sorted(sources))}")
    if pre_stream:
        pre_stream()
    elif heading:
        st.markdown(heading)

    response = _stream_into_component(model, prompt)
    if response:
        if content_key:
            st.session_state.generated_content[content_key] = response
        if post_stream:
            post_stream(response, sources, source_filter)
        return response
    st.error("Could not generate content", icon=":material/error:")
    return None


def summarize_pdf(source_filter=None):
    """Generate PDF summary"""
    try:
        _run_feature(
            source_filter=source_filter,
            search_query="summary main points key information",
            search_k=10,
            heading="### :material/summarize: Document Summary",
            content_key="summary",
            toast_msg="Generating summary…",
            prompt_template=(
                "Please provide a comprehensive summary of the following document content.\n"
                "Include the main points, key findings, and important information:\n\n"
                "{context}\n\nSummary:"
            ),
        )
    except Exception as e:
        st.error(f"❌ Error generating summary: {str(e)}")


def generate_questions(source_filter=None):
    """Generate questions from PDF content"""
    try:

        def _after(response, sources, sf):
            qs = [
                q.strip()
                for q in response.split("\n")
                if q.strip() and ("?" in q or q.strip().endswith("."))
            ]
            with st.container(horizontal=True, gap="small"):
                for i, q in enumerate(qs[:10], 1):
                    q = q.lstrip("0123456789.- ")
                    st.button(
                        f":material/answer: Q{i}",
                        key=f"answer_{i}",
                        on_click=answer_question,
                        args=(q, sf),
                        use_container_width=True,
                    )
            st.caption("Scroll down to download your questions")

        _run_feature(
            source_filter=source_filter,
            search_query="main topics concepts important information",
            search_k=8,
            heading="### ❓ Practice Questions",
            content_key="questions",
            toast_msg="Generating questions…",
            prompt_template=(
                "Based on the following document content, generate 8-10 thoughtful questions "
                "that would help someone understand the key concepts and important information.\n"
                "Make the questions clear and specific:\n\n{context}\n\nQuestions:"
            ),
            post_stream=_after,
        )
    except Exception as e:
        st.error(f"❌ Error generating questions: {str(e)}")


def generate_mcqs(source_filter=None):
    """Generate multiple choice questions"""
    try:
        _run_feature(
            source_filter=source_filter,
            search_query="key concepts important facts definitions",
            search_k=6,
            heading="### :material/quiz: Multiple Choice Questions",
            content_key="mcqs",
            toast_msg="Generating MCQs…",
            prompt_template=(
                "Based on the following document content, create 5 multiple choice questions "
                "with 4 options each (A, B, C, D).\n"
                "Include the correct answer at the end. Format each question clearly:\n\n"
                "{context}\n\nMCQs:"
            ),
            post_stream=lambda r, s, sf: st.caption(
                "Scroll down to download your MCQs"
            ),
        )
    except Exception as e:
        st.error(f"❌ Error generating MCQs: {str(e)}")


def generate_notes(source_filter=None):
    """Generate short notes from PDF"""
    try:

        def _notes_heading():
            with st.container(border=True):
                st.markdown(
                    "### :material/note_stack: Study Notes", text_alignment="center"
                )

        _run_feature(
            source_filter=source_filter,
            search_query="main concepts key points important information",
            search_k=8,
            heading=None,
            pre_stream=_notes_heading,
            content_key="notes",
            toast_msg="Generating notes…",
            prompt_template=(
                "Create concise, well-organized study notes from the following content.\n"
                "Use bullet points, headings, and clear structure. Focus on key concepts and important information:\n\n"
                "{context}\n\nStudy Notes:"
            ),
            post_stream=lambda r, s, sf: st.caption(
                "Scroll down to download your notes"
            ),
        )
    except Exception as e:
        st.error(f"❌ Error generating notes: {str(e)}")


def answer_question(question, source_filter=None):
    """Answer a specific question with hybrid search and streaming"""
    try:
        if not os.path.exists("faiss_index"):
            st.error(
                "Please upload and process PDF files first",
                icon=":material/description:",
            )
            return

        filter_param = _get_source_filter(source_filter)

        with st.spinner(":material/search: Finding answer..."):
            docs = _hybrid_search(question, k=5, filter=filter_param)

            if not docs:
                st.warning("No relevant information found", icon=":material/warning:")
                return

            model, prompt = get_conversational_chain()
            if not model or not prompt:
                return

            sources = set(d.metadata["source"] for d in docs if hasattr(d, "metadata"))
            context = "\n\n".join([doc.page_content for doc in docs])

            memory = _get_conversation_history(3)
            formatted_prompt = prompt.format(
                conversation_history=memory, context=context, question=question
            )

        st.toast("Answer found!", icon=":material/check_circle:")
        response_text = _stream_into_component(model, formatted_prompt)

        if response_text:
            if sources:
                st.caption(f"📄 Source: {', '.join(sorted(sources))}")

            st.session_state.chat_history.append(
                {
                    "question": question,
                    "answer": response_text,
                    "source": ", ".join(sorted(sources)) if sources else "Unknown",
                    "timestamp": datetime.now(),
                }
            )
        else:
            st.error("Could not generate answer", icon=":material/error:")

    except Exception as e:
        st.error(f"❌ Error answering question: {str(e)}")


def main():
    st.set_page_config(
        page_title="PDF-GPT | Chat With Your PDFs",
        page_icon=":material/picture_as_pdf:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # MathJax for LaTeX math rendering in streaming output
    st.html(
        '<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" async></script>'
    )

    init_session_state()

    st.title("PDF-GPT", text_alignment="center")
    st.caption(
        "Chat with your PDFs using AI — generate summaries, notes, MCQs, and more",
        text_alignment="center",
    )
    st.markdown(
        "### :material/auto_awesome: AI-powered features", text_alignment="center"
    )

    cols = st.columns(4, vertical_alignment="center")
    card_data = [
        (
            "school",
            "Study Smarter",
            "Generate summaries and notes",
            ":material/summarize: Summarize",
        ),
        (
            "search_insights",
            "Research Faster",
            "Ask questions instantly",
            ":material/help: Generate questions",
        ),
        (
            "quiz",
            "Practice Better",
            "Generate questions and MCQs",
            ":material/quiz: Create MCQs",
        ),
        (
            "timer",
            "Save Time",
            "Extract key information quickly",
            ":material/note_stack: Generate notes",
        ),
    ]
    btn = {}
    for col, (icon, title, desc, label) in zip(cols, card_data):
        with col:
            with st.container(border=True, height="stretch"):
                st.markdown(f":material/{icon}: **{title}**")
                st.caption(desc)
                btn[icon] = st.button(label, use_container_width=True)

    st.space()

    # --- Per-document source selector ---
    if st.session_state.processed_docs:
        doc_options = ["All Documents"] + list(st.session_state.processed_docs.keys())
        st.session_state.selected_source = st.selectbox(
            ":material/target: Target document",
            options=doc_options,
            index=doc_options.index(st.session_state.selected_source)
            if st.session_state.selected_source in doc_options
            else 0,
            help="Select a specific document to analyze, or 'All Documents' for cross-document analysis",
        )
    else:
        st.session_state.selected_source = "All Documents"
        st.caption("Upload PDFs and click **Process documents** to select a source")

    just_generated = None
    sf = st.session_state.selected_source
    if btn.get("school"):
        just_generated = "summary"
        summarize_pdf(source_filter=sf)
    if btn.get("search_insights"):
        just_generated = "questions"
        generate_questions(source_filter=sf)
    if btn.get("quiz"):
        just_generated = "mcqs"
        generate_mcqs(source_filter=sf)
    if btn.get("timer"):
        just_generated = "notes"
        generate_notes(source_filter=sf)

    # ── Persistent feature output — re-renders from cache on every rerun ──
    feature_order = [
        ("summary", "### :material/summarize: Document Summary", False),
        ("mcqs", "### :material/quiz: Multiple Choice Questions", False),
        ("notes", "### :material/note_stack: Study Notes", False),
    ]
    for key, heading, horiz in feature_order:
        if key in st.session_state.generated_content and key != just_generated:
            st.markdown(heading)
            _render_cached(st.session_state.generated_content[key], horizontal=horiz)

    if (
        "questions" in st.session_state.generated_content
        and just_generated != "questions"
    ):
        st.markdown("### ❓ Practice Questions")
        _render_cached(st.session_state.generated_content["questions"])
        qs = [
            q.strip()
            for q in st.session_state.generated_content["questions"].split("\n")
            if q.strip() and ("?" in q or q.strip().endswith("."))
        ]
        with st.container(horizontal=True, gap="small"):
            for i, q in enumerate(qs[:10], 1):
                q = q.lstrip("0123456789.- ")
                st.button(
                    f":material/answer: Q{i}",
                    key=f"q_persist_{i}",
                    on_click=answer_question,
                    args=(q, sf),
                    use_container_width=True,
                )

    st.space()

    st.markdown("### :material/chat: Ask questions about your PDFs")

    with st.container(border=True):
        for msg in st.session_state.messages:
            avatar = msg.get(
                "avatar",
                ":material/robot:"
                if msg["role"] == "assistant"
                else ":material/person:",
            )
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

        prompt = st.chat_input("Ask a question about your PDFs...")

    if prompt:
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(prompt)
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "avatar": ":material/person:"}
        )

        filter_param = _get_source_filter(st.session_state.selected_source)

        model = None
        chain_prompt = None
        formatted_prompt = None
        sources = set()
        docs = []

        with st.spinner(":material/search: Searching your documents..."):
            docs = _hybrid_search(prompt, k=5, filter=filter_param)

            if docs:
                model, chain_prompt = get_conversational_chain()
                if model and chain_prompt:
                    sources = set(
                        d.metadata["source"] for d in docs if hasattr(d, "metadata")
                    )
                    context = "\n\n".join([doc.page_content for doc in docs])
                    memory = _get_conversation_history(3)
                    formatted_prompt = chain_prompt.format(
                        conversation_history=memory, context=context, question=prompt
                    )

        if docs and model and chain_prompt:
            with st.chat_message("assistant", avatar=":material/auto_awesome:"):
                response_text = _stream_into_component(model, formatted_prompt)

            if response_text:
                if sources:
                    st.caption(
                        f":material/description: Source: {', '.join(sorted(sources))}"
                    )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response_text,
                        "avatar": ":material/auto_awesome:",
                    }
                )
                st.session_state.chat_history.append(
                    {
                        "question": prompt,
                        "answer": response_text,
                        "source": ", ".join(sorted(sources)) if sources else "Unknown",
                        "timestamp": datetime.now(),
                    }
                )
        else:
            with st.chat_message("assistant", avatar=":material/auto_awesome:"):
                st.error(
                    "No relevant information found in your documents.",
                    icon=":material/search_off:",
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "No relevant information found in your documents.",
                        "avatar": ":material/auto_awesome:",
                    }
                )

    if st.session_state.generated_content:
        st.space()
        st.markdown("### :material/download: Download generated content")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        downloads = [
            ("summary", ":material/summarize: Download Summary", f"summary_{ts}.txt"),
            ("questions", ":material/help: Download Questions", f"questions_{ts}.txt"),
            ("mcqs", ":material/quiz: Download MCQs", f"mcqs_{ts}.txt"),
            ("notes", ":material/note_stack: Download Notes", f"notes_{ts}.txt"),
        ]
        with st.container(
            horizontal=True, gap="small", horizontal_alignment="distribute"
        ):
            for key, label, fname in downloads:
                if key in st.session_state.generated_content:
                    create_download_link(
                        st.session_state.generated_content[key], fname, label
                    )

    # ── Sidebar ──
    with st.sidebar:
        st.subheader(":material/upload_file: Document upload")

        with st.container(border=True):
            st.markdown("**Quick start**")
            st.markdown("1. Upload your PDF files")
            st.markdown("2. Click Process documents")
            st.markdown("3. Use AI features or ask questions")

        pdf_docs = st.file_uploader(
            "Choose PDF files",
            accept_multiple_files=True,
            type=["pdf"],
            help="Upload one or more PDF files to analyze",
        )

        if pdf_docs:
            st.toast(f"Uploaded {len(pdf_docs)} file(s)", icon=":material/upload_file:")
            upload_cols = st.columns(2)
            for idx, pdf in enumerate(pdf_docs):
                with upload_cols[idx % 2]:
                    st.markdown(f":material/description: **{pdf.name}**")

        if st.button(":material/settings: Process documents", type="secondary"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file", icon=":material/error:")
            else:
                with st.status("Processing documents...", expanded=True) as status:
                    progress_bar = st.progress(0)

                    progress_bar.progress(15)
                    status.update(label=":shimmer[Extracting text from PDFs...]")
                    docs_text, docs_info = get_pdf_text(pdf_docs)

                    if docs_text:
                        progress_bar.progress(35)

                        # Filter out already-processed docs
                        new_docs = {}
                        new_info = {}
                        for name in docs_text:
                            if name not in st.session_state.processed_docs:
                                new_docs[name] = docs_text[name]
                                new_info[name] = docs_info[name]
                            else:
                                st.info(f"{name} already processed, skipping")

                        if not new_docs:
                            st.warning("All uploaded files have already been processed")
                            progress_bar.empty()
                        else:
                            progress_bar.progress(50)
                            status.update(label="Creating text chunks...")
                            chunks, metadatas = get_text_chunks_with_metadata(new_docs)

                            if chunks:
                                progress_bar.progress(70)
                                status.update(label="Building search index...")

                                # Stage: temporarily extend session state for index build
                                old_chunks_len = len(st.session_state.all_chunks)
                                old_metas_len = len(st.session_state.all_metadatas)
                                st.session_state.all_chunks.extend(chunks)
                                st.session_state.all_metadatas.extend(metadatas)

                                progress_bar.progress(85)
                                success = rebuild_faiss_index()

                                if success:
                                    # Commit: add processed_docs tracking
                                    for name, info in new_info.items():
                                        st.session_state.processed_docs[name] = info
                                    for meta in metadatas:
                                        doc_name = meta["source"]
                                        if (
                                            "chunk_count"
                                            not in st.session_state.processed_docs[
                                                doc_name
                                            ]
                                        ):
                                            st.session_state.processed_docs[doc_name][
                                                "chunk_count"
                                            ] = 0
                                        st.session_state.processed_docs[doc_name][
                                            "chunk_count"
                                        ] += 1
                                else:
                                    # Rollback: remove staged chunks, keep state clean
                                    del st.session_state.all_chunks[old_chunks_len:]
                                    del st.session_state.all_metadatas[old_metas_len:]

                                progress_bar.progress(100)

                                if success:
                                    status.update(
                                        label=f"Processed {len(new_docs)} document(s)!",
                                        state="complete",
                                    )
                                    st.toast(
                                        f"Processed {len(new_docs)} document(s)!",
                                        icon=":material/check_circle:",
                                    )
                                    st.snow()
                                else:
                                    status.update(
                                        label="Failed to create vector store",
                                        state="error",
                                    )
                                    st.error(
                                        "Failed to create vector store",
                                        icon=":material/error:",
                                    )
                            else:
                                status.update(
                                    label="Failed to create text chunks", state="error"
                                )
                                st.error(
                                    "Failed to create text chunks",
                                    icon=":material/error:",
                                )
                    else:
                        status.update(
                            label="Failed to extract text from PDFs", state="error"
                        )
                        st.error(
                            "Failed to extract text from PDFs", icon=":material/error:"
                        )

                    progress_bar.empty()

        if st.session_state.processed_docs:
            st.subheader(":material/library_books: Processed documents")
            st.caption(
                f"{len(st.session_state.processed_docs)} document(s) · {len(st.session_state.all_chunks)} chunks"
            )

            for doc_name, info in list(st.session_state.processed_docs.items()):
                pages = info.get("total_pages", "?")
                chunks = info.get("chunk_count", "?")
                size = info.get("size_kb", "?")
                ocr = info.get("ocr", False)

                with st.container(border=True):
                    cols = st.columns([4, 1])
                    with cols[0]:
                        st.markdown(f"**{doc_name}**")
                        if ocr:
                            st.markdown('<span style="background:transparent;border:1px solid var(--st-primary-color,#CDEA12);color:var(--st-primary-color,#CDEA12);border-radius:4px;padding:1px 6px;font-size:0.7rem;">OCR</span>', unsafe_allow_html=True)
                        st.caption(f"{pages} pages · {chunks} chunks · {size} KB")
                    with cols[1]:
                        if st.button(
                            ":material/delete:",
                            key=f"rm_{doc_name}",
                            help=f"Remove {doc_name}",
                        ):
                            removed = remove_document(doc_name)
                            if removed > 0:
                                st.toast(
                                    f"Removed {doc_name}", icon=":material/delete:"
                                )
                                st.rerun()

        st.divider()
        st.subheader(":material/info: About")
        st.markdown(
            "AI-powered PDF analysis tool. Ask questions, generate summaries, and create study materials from your documents."
        )
        st.markdown('<span style="background:transparent;border:1px solid var(--st-secondary-text-color,#888);color:var(--st-secondary-text-color,#888);border-radius:4px;padding:1px 6px;font-size:0.7rem;">v2.2</span>', unsafe_allow_html=True)
        st.markdown(
            ":material/person: Krishna · :material/code: [GitHub](https://github.com/krishnak2c/PDF-GPT) · :material/bug_report: [Issues](https://github.com/krishnak2c/PDF-GPT/issues)"
        )

        with st.expander("What's new in v2.2", icon=":material/new_releases:"):
            st.markdown("• OCR for scanned PDFs (Gemini Vision)")
            st.markdown("• AI reranker for better search quality")
            st.markdown("• Hybrid search (keyword + vector)")
            st.markdown("• Streaming responses")
            st.markdown("• Conversation memory")
            st.markdown("• Redesigned UI")

        st.caption(
            ":material/psychology: Gemini 3.5 Flash :material/lock: Local processing"
        )


if __name__ == "__main__":
    main()
