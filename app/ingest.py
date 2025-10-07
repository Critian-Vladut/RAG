from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os
import logging
import re
from typing import List, Optional, Dict, Any
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImprovedTOCExtractor:
    """Enhanced Table of Contents extraction with better pattern recognition."""
    
    def __init__(self):
        # Expanded TOC detection keywords
        self.toc_keywords = [
            "table of contents", "contents", "table of content",
            "index", "outline", "chapter", "sections",
            # Also look for structural indicators
            "page", "chapter 1", "introduction", "executive summary"
        ]
        
        # Multiple TOC entry patterns
        self.toc_patterns = [
            # Standard: "1.2 Title .... 15"
            re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.+?)\s*\.{2,}\s*(\d+)\s*$"),
            # Without dots: "1.2 Title 15"  
            re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.+?)\s+(\d+)\s*$"),
            # No page numbers: "1.2 Title"
            re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.+?)\s*$"),
            # With dashes: "1.2 - Title"
            re.compile(r"^\s*(\d+(?:\.\d+)*)\s*[-–—]\s*(.+?)\s*(?:\.{2,}\s*(\d+))?\s*$"),
            # Roman numerals: "I. Title"
            re.compile(r"^\s*([IVX]+\.?)\s+(.+?)\s*(?:\.{2,}\s*(\d+))?\s*$"),
            # Bullet points: "• Title"
            re.compile(r"^\s*[•·▪▫]\s+(.+?)\s*(?:\.{2,}\s*(\d+))?\s*$"),
            # Chapter format: "Chapter 1: Title"
            re.compile(r"^\s*(?:Chapter|Section)\s+(\d+(?:\.\d+)*)[:\s]+(.+?)\s*(?:\.{2,}\s*(\d+))?\s*$", re.IGNORECASE),
        ]
        
        # Patterns to exclude (not TOC entries)
        self.exclude_patterns = [
            re.compile(r"^\s*(?:figure|table|appendix)\s+\d+", re.IGNORECASE),
            re.compile(r"^\s*(?:page|total|subtotal)", re.IGNORECASE),
            re.compile(r"^\s*\$\d+", re.IGNORECASE),  # Money amounts
            re.compile(r"^\s*\d{4}-\d{2}-\d{2}", re.IGNORECASE),  # Dates
        ]

    def find_toc_pages(self, documents: List) -> List[int]:
        """Find pages that likely contain table of contents."""
        toc_pages = []
        
        for i, doc in enumerate(documents[:10]):  # Check first 10 pages
            content = (doc.page_content or "").lower()
            
            # Look for TOC indicators
            if any(keyword in content for keyword in self.toc_keywords):
                toc_pages.append(i)
                continue
            
            # Look for structural patterns (multiple numbered entries)
            lines = content.split('\n')
            numbered_lines = 0
            for line in lines:
                if re.match(r'^\s*\d+(?:\.\d+)*\s+\w+', line.strip()):
                    numbered_lines += 1
            
            # If page has many numbered entries, likely a TOC
            if numbered_lines >= 5:
                toc_pages.append(i)
        
        return toc_pages

    def extract_toc_entries(self, documents: List) -> List[Dict[str, Any]]:
        """Extract TOC entries with improved pattern matching."""
        toc_entries = []
        toc_pages = self.find_toc_pages(documents)
        
        if not toc_pages:
            return toc_entries
        
        for page_idx in toc_pages:
            if page_idx >= len(documents):
                continue
                
            content = documents[page_idx].page_content or ""
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines):
                line = line.strip()
                if not line or len(line) < 3:
                    continue
                
                # Skip if matches exclusion patterns
                if any(pattern.match(line) for pattern in self.exclude_patterns):
                    continue
                
                # Try each TOC pattern
                for pattern_idx, pattern in enumerate(self.toc_patterns):
                    match = pattern.match(line)
                    if match:
                        entry = self._create_toc_entry(match, pattern_idx, page_idx, line_num)
                        if entry and self._is_valid_toc_entry(entry):
                            toc_entries.append(entry)
                        break  # Only match first pattern
        
        return self._deduplicate_toc_entries(toc_entries)

    def _create_toc_entry(self, match, pattern_idx: int, page_idx: int, line_num: int) -> Dict[str, Any]:
        """Create TOC entry from regex match."""
        groups = match.groups()
        
        if pattern_idx <= 3:  # Standard numbered patterns
            section_num = groups[0] if groups[0] else ""
            title = groups[1].strip() if len(groups) > 1 and groups[1] else ""
            page_num = groups[2] if len(groups) > 2 and groups[2] else None
        elif pattern_idx == 4:  # Roman numerals
            section_num = groups[0] if groups[0] else ""
            title = groups[1].strip() if len(groups) > 1 and groups[1] else ""
            page_num = groups[2] if len(groups) > 2 and groups[2] else None
        elif pattern_idx == 5:  # Bullet points
            section_num = ""
            title = groups[0].strip() if groups[0] else ""
            page_num = groups[1] if len(groups) > 1 and groups[1] else None
        elif pattern_idx == 6:  # Chapter format
            section_num = groups[0] if groups[0] else ""
            title = groups[1].strip() if len(groups) > 1 and groups[1] else ""
            page_num = groups[2] if len(groups) > 2 and groups[2] else None
        else:
            return None
        
        return {
            "number": section_num,
            "title": title,
            "page": page_num,
            "toc_page": page_idx,
            "line": line_num,
            "confidence": self._calculate_confidence(section_num, title, page_num)
        }

    def _is_valid_toc_entry(self, entry: Dict[str, Any]) -> bool:
        """Validate TOC entry quality."""
        title = entry.get("title", "")
        
        # Filter out low-quality entries
        if len(title) < 3 or len(title) > 200:
            return False
        
        # Skip entries that are mostly numbers or symbols
        if len(re.sub(r'[^\w\s]', '', title)) < 3:
            return False
        
        # Skip common non-TOC lines
        title_lower = title.lower()
        if any(skip in title_lower for skip in ['page', 'continued', 'see page', 'figure', 'table']):
            return False
        
        return entry.get("confidence", 0) > 0.5

    def _calculate_confidence(self, section_num: str, title: str, page_num: str) -> float:
        """Calculate confidence score for TOC entry."""
        confidence = 0.5
        
        # Boost confidence for good patterns
        if section_num and re.match(r'^\d+(\.\d+)*$', section_num):
            confidence += 0.3
        
        if title and 5 <= len(title) <= 100:
            confidence += 0.2
        
        if page_num and page_num.isdigit():
            confidence += 0.2
        
        # Penalize for suspicious patterns
        if title and title.isupper() and len(title) > 50:
            confidence -= 0.2
        
        return min(1.0, max(0.0, confidence))

    def _deduplicate_toc_entries(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate TOC entries."""
        seen = set()
        unique_entries = []
        
        for entry in sorted(entries, key=lambda x: x.get("confidence", 0), reverse=True):
            key = (entry.get("number", ""), entry.get("title", "").lower())
            if key not in seen and key[1]:  # Must have title
                seen.add(key)
                unique_entries.append(entry)
        
        return unique_entries


class SmartRAGAgent:
    """
    Smart RAG agent: extracts document intelligence, structure (TOC vs body),
    creates chunks and a FAISS vectorstore, and answers queries with structure-aware logic.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1200,
        chunk_overlap: int = 300,
        device: str = "cpu",
        use_improved_intelligence: bool = True,
    ):
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.device = device

        if use_improved_intelligence:
            try:
                from improved_intelligence import ImprovedDocumentIntelligence
                self.intelligence_extractor = ImprovedDocumentIntelligence()
                self.use_improved_intelligence = True
                logger.info("Using improved document intelligence extraction")
            except ImportError:
                logger.warning("Could not import ImprovedDocumentIntelligence, falling back to original")
                self.use_improved_intelligence = False
        else:
            self.use_improved_intelligence = False

        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise

        # title/metadata heuristics
        self.title_patterns = [
            r"^Project\s+Title[:\s]*(.+)$",
            r"^Title[:\s]*(.+)$",
            r"^[A-Z][A-Za-z\s\-:]{10,100}$",
            r"^\d+\.\s+[A-Z][A-Za-z\s\-:]{5,80}$",
            r"^[A-Z\s]{5,50}$",
        ]

        self.metadata_patterns = {
            "author": [r"Author[:\s]*(.+)$", r"By[:\s]*(.+)$"],
            "date": [r"Date[:\s]*(.+)$", r"\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}"],
            "project": [r"Project[:\s]*(.+)$", r"Project\s+Title[:\s]*(.+)$"],
            "version": [r"Version[:\s]*(.+)$", r"v\d+\.\d+"],
            "company": [r"Company[:\s]*(.+)$", r"Organization[:\s]*(.+)$"],
        }

        # query type detection: includes structure_check AND table_of_contents
        self.query_types = {
            "title": ["title", "what is this", "name of document", "project title"],
            "summary": ["summary", "about", "main topic", "overview"],
            "author": ["author", "who wrote", "written by"],
            "date": ["date", "when", "published"],
            "specific_fact": ["what is", "how much", "how many", "where", "when"],
            "list": ["list", "enumerate", "what are", "types of"],
            "explanation": ["explain", "how", "why", "describe"],
            "structure_check": [
                "structure",
                "sections match",
                "document structure",
                "contents match",
            ],
            "table_of_contents": [
                "table of contents",
                "toc",
                "list contents", 
                "show contents",
                "document outline", 
                "section list", 
                "chapter list",
                "contents",
                "outline"
            ],
        }

    def extract_document_intelligence(self, documents: List) -> Dict[str, Any]:
        """Extract document intelligence using improved or original method."""
        if hasattr(self, 'use_improved_intelligence') and self.use_improved_intelligence:
            return self.intelligence_extractor.extract_document_intelligence(documents)
        else:
            # Original implementation as fallback
            return self._original_extract_document_intelligence(documents)

    def _original_extract_document_intelligence(self, documents: List) -> Dict[str, Any]:
        """Original implementation kept for compatibility."""
        intelligence = {
            "titles": [],
            "headings": [],
            "metadata": defaultdict(list),
            "first_page_content": "",
            "key_terms": set(),
        }

        for i, doc in enumerate(documents):
            content = doc.page_content or ""
            if i == 0:
                intelligence["first_page_content"] = content[:2000]

            lines = content.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                for pattern in self.title_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        title = match.group(1).strip() if match.groups() else line
                        source = "explicit" if "Project Title" in pattern or "Title:" in pattern else "heuristic"
                        intelligence["titles"].append((title, source))

                for meta_type, patterns in self.metadata_patterns.items():
                    for pattern in patterns:
                        match = re.search(pattern, line, re.IGNORECASE)
                        if match:
                            meta_value = match.group(1).strip() if match.groups() else match.group(0).strip()
                            intelligence["metadata"][meta_type].append(meta_value)

                # simple heading heuristics
                if len(line) < 100 and (
                    line.isupper()
                    or re.match(r"^\d+\.", line)
                    or (re.match(r"^[A-Z][a-z]", line) and ":" not in line)
                ):
                    intelligence["headings"].append(line)

                # collect capitalized words as key terms
                key_terms = re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b", line)
                intelligence["key_terms"].update(key_terms)

        # Convert to serializable format for compatibility
        intelligence["metadata"] = dict(intelligence["metadata"])
        intelligence["key_terms"] = list(intelligence["key_terms"])
        
        return intelligence
        
    def extract_document_structure(self, documents: List) -> Dict[str, Any]:
        """Enhanced document structure extraction with better TOC detection."""
        
        structure = {
            "toc_sections": [],
            "body_sections": [],
            "missing_in_body": [],
            "missing_in_toc": [],
            "structure_issues": [],
            "toc_found": False,
            "toc_pages": [],
            "extraction_method": "enhanced"
        }
        
        # Use improved TOC extractor
        toc_extractor = ImprovedTOCExtractor()
        toc_entries = toc_extractor.extract_toc_entries(documents)
        
        if toc_entries:
            structure["toc_found"] = True
            structure["toc_sections"] = toc_entries
            structure["toc_pages"] = toc_extractor.find_toc_pages(documents)
        
        # Extract body sections (keep existing logic but improve it)
        for idx, doc in enumerate(documents):
            page_num = doc.metadata.get("page", idx)
            content = doc.page_content or ""
            
            # Look for section headings in body
            lines = content.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Enhanced heading detection
                heading_patterns = [
                    r"^\s*(\d+(?:\.\d+)*)\s+([A-Z][^.!?]{5,100})\s*$",  # "1.2 Section Title"
                    r"^\s*(?:Chapter|Section)\s+(\d+(?:\.\d+)*)[:\s]+(.{5,100})\s*$",  # "Chapter 1: Title"
                    r"^\s*([A-Z][A-Z\s]{10,60}[A-Z])\s*$",  # "SECTION TITLE"
                ]
                
                for pattern in heading_patterns:
                    match = re.match(pattern, line, re.IGNORECASE)
                    if match:
                        if len(match.groups()) >= 2:
                            section_num = match.group(1)
                            section_title = re.sub(r"\s+", " ", match.group(2)).rstrip(".")
                        else:
                            section_num = ""
                            section_title = match.group(1)
                        
                        structure["body_sections"].append({
                            "number": section_num,
                            "title": section_title,
                            "page": page_num
                        })
                        break
        
        # Compare TOC and body sections
        if structure["toc_sections"] and structure["body_sections"]:
            toc_numbers = {s["number"] for s in structure["toc_sections"] if s.get("number")}
            body_numbers = {s["number"] for s in structure["body_sections"] if s.get("number")}
            
            structure["missing_in_body"] = [
                s for s in structure["toc_sections"] 
                if s.get("number") and s["number"] not in body_numbers
            ]
            structure["missing_in_toc"] = [
                s for s in structure["body_sections"] 
                if s.get("number") and s["number"] not in toc_numbers
            ]
            
            if structure["missing_in_body"]:
                structure["structure_issues"].append(
                    f"TOC lists {len(structure['missing_in_body'])} sections not found in body"
                )
            if structure["missing_in_toc"]:
                structure["structure_issues"].append(
                    f"Body has {len(structure['missing_in_toc'])} sections not in TOC"
                )
        
        return structure

    def classify_query(self, query: str) -> str:
        q = (query or "").lower()
        for qtype, keys in self.query_types.items():
            for k in keys:
                if k in q:
                    return qtype
        return "general"

    def smart_chunk_documents(self, documents: List, intelligence: Dict[str, Any]) -> List:
        """
        Create prioritized chunks: metadata/first-page chunks smaller, rest larger.
        Returns langchain Document objects (split_documents returns these).
        """
        important_docs = []
        regular_docs = []

        top_titles = [t[0] for t in intelligence.get("titles", [])[:5]]

        for idx, doc in enumerate(documents):
            content = doc.page_content or ""
            has_meta = any(
                any(re.search(pat, content, re.IGNORECASE) for pat in pats)
                for pats in self.metadata_patterns.values()
            )
            is_first = doc.metadata.get("page", idx) == 0
            has_title = any(title and title in content for title in top_titles)

            if has_meta or is_first or has_title:
                important_docs.append(doc)
            else:
                regular_docs.append(doc)

        chunks = []

        if important_docs:
            small_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400, chunk_overlap=50, length_function=len, separators=["\n\n", "\n", ". ", " ", ""]
            )
            small_chunks = small_splitter.split_documents(important_docs)
            for c in small_chunks:
                c.metadata = c.metadata or {}
                c.metadata["importance"] = "high"
                c.metadata["chunk_type"] = "metadata"
            chunks.extend(small_chunks)

        if regular_docs:
            regular_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
            )
            reg_chunks = regular_splitter.split_documents(regular_docs)
            for c in reg_chunks:
                c.metadata = c.metadata or {}
                c.metadata["importance"] = "normal"
                c.metadata["chunk_type"] = "content"
            chunks.extend(reg_chunks)

        return chunks

    def intelligent_retrieval(self, query: str, vectorstore, intelligence: Dict[str, Any], k: int = 8):
        """
        Return a list of Documents relevant to the query. Behavior varies by query type.
        """
        qtype = self.classify_query(query)

        if qtype == "title":
            explicit_titles = [t for t, src in intelligence.get("titles", []) if src == "explicit"]
            if explicit_titles:
                # return small synthetic docs describing explicit titles
                return [{"page_content": t, "metadata": {"type": "explicit_title"}} for t in explicit_titles[:k]]

            # fallback: search the vectorstore for likely title-like short chunks
            docs = vectorstore.similarity_search(query, k=k * 2)
            title_candidates = []
            for d in docs:
                text = (d.page_content or "").strip()
                if len(text) < 200 and any(re.search(p, text, re.IGNORECASE) for p in self.title_patterns):
                    title_candidates.append(d)
            return title_candidates[:k] if title_candidates else docs[:k]

        # for author/date/summary prefer high importance chunks
        if qtype in ("author", "date", "summary"):
            docs = vectorstore.similarity_search(query, k=k * 2)
            high = [d for d in docs if d.metadata.get("importance") == "high"]
            normal = [d for d in docs if d.metadata.get("importance") != "high"]
            return (high + normal)[:k]

        # default
        return vectorstore.similarity_search(query, k=k)

    def format_context_for_llama(self, query: str, docs: List, intelligence: Dict[str, Any], structure: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a short context string to feed LLM (or return to the user).
        """
        parts = []

        # If this is a title query, list candidate titles first
        if self.classify_query(query) == "title":
            explicit = [t for t, s in intelligence.get("titles", []) if s == "explicit"]
            if explicit:
                parts.append("Explicit titles found:")
                for i, t in enumerate(explicit[:5], 1):
                    parts.append(f"{i}. {t}")
            else:
                parts.append("Title candidates from the document:")

        # Add a few source excerpts
        for i, d in enumerate(docs[:5], 1):
            page = d.metadata.get("page", "unknown")
            importance = d.metadata.get("importance", "")
            header = f"[Source {i}] Page {page} {('[Important]' if importance == 'high' else '')}"
            parts.append(header)
            parts.append((d.page_content or "")[:1000].strip() + ("..." if len((d.page_content or "")) > 1000 else ""))

        # Add structure summary if relevant
        if structure:
            parts.append("\nDocument structure (TOC vs Body) summary:")
            parts.append(f"TOC sections: {len(structure.get('toc_sections', []))}")
            parts.append(f"Body sections: {len(structure.get('body_sections', []))}")
            if structure.get("structure_issues"):
                parts.append("Structure issues detected:")
                for issue in structure["structure_issues"][:5]:
                    parts.append(f"- {issue}")

        parts.append(f"\nQuestion: {query}")
        parts.append("Please answer concisely using the above sources.")

        return "\n\n".join(parts)

    def handle_toc_query(self, system: Dict[str, Any]) -> str:
        """Handle table of contents specific queries."""
        structure = system.get("structure", {})
        
        if not structure or not structure.get("toc_found", False):
            return "No table of contents was found in this document. The document may not have a traditional TOC structure, or it may be formatted in a way that wasn't detected."
        
        toc_sections = structure.get("toc_sections", [])
        if not toc_sections:
            return "A table of contents section was detected but no entries could be extracted."
        
        # Format TOC nicely
        response = ["Table of Contents:"]
        response.append("=" * 50)
        
        for i, section in enumerate(toc_sections[:25], 1):  # Limit to first 25
            number = section.get("number", "")
            title = section.get("title", "")
            page = section.get("page", "")
            
            if number and page:
                response.append(f"{number}. {title} ... {page}")
            elif number:
                response.append(f"{number}. {title}")
            else:
                response.append(f"• {title}")
        
        if len(toc_sections) > 25:
            response.append(f"... and {len(toc_sections) - 25} more sections")
        
        response.append(f"\nTotal sections found: {len(toc_sections)}")
        if structure.get("toc_pages"):
            response.append(f"TOC found on pages: {', '.join(map(str, structure['toc_pages']))}")
        
        return "\n".join(response)

    def process_pdf(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        """
        Process PDF and return system dict with vectorstore, intelligence, structure, chunks.
        """
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()
            if not documents:
                logger.error("No pages loaded from PDF.")
                return None

            # ensure page metadata present
            for idx, d in enumerate(documents):
                if "page" not in d.metadata:
                    d.metadata["page"] = idx

            intelligence = self.extract_document_intelligence(documents)
            structure = self.extract_document_structure(documents)
            chunks = self.smart_chunk_documents(documents, intelligence)

            if not chunks:
                # fallback chunk: split whole document text if splitter failed
                all_text = "\n\n".join((d.page_content or "") for d in documents)
                splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
                fallback_docs = splitter.create_documents([all_text])
                for fd in fallback_docs:
                    fd.metadata = fd.metadata or {}
                    fd.metadata["importance"] = "normal"
                chunks = fallback_docs

            vectorstore = FAISS.from_documents(chunks, embedding=self.embeddings)
            system = {"vectorstore": vectorstore, "intelligence": intelligence, "structure": structure, "chunks": chunks}
            logger.info("PDF processed successfully.")
            return system
        except Exception as e:
            logger.error(f"Error processing PDF: {e}", exc_info=True)
            return None

    def query(self, question: str, system: Dict[str, Any]) -> str:
        """
        Answer a question using the system dict returned by process_pdf.
        For structure_check queries, return a special structured response.
        Otherwise return a context string built from top documents.
        """
        try:
            if not system or "vectorstore" not in system:
                return "No processed document available. Please upload and process a PDF first."

            qtype = self.classify_query(question)
            
            # DEBUG: Print what query type was detected
            print(f"DEBUG: Query '{question}' classified as: {qtype}")
            
            if qtype == "structure_check":
                s = system.get("structure", {})
                if not s:
                    return "No structure information extracted from the document."
                # human-readable report
                report = []
                report.append(f"TOC sections: {len(s.get('toc_sections', []))}")
                report.append(f"Body sections: {len(s.get('body_sections', []))}")
                report.append(f"Missing in body (TOC entries not found): {len(s.get('missing_in_body', []))}")
                report.append(f"Missing in TOC (body entries not listed): {len(s.get('missing_in_toc', []))}")
                if s.get("structure_issues"):
                    report.append("Structure issues:")
                    report.extend([f"- {it}" for it in s["structure_issues"]])
                else:
                    report.append("No major structure issues detected.")
                return "\n".join(report)

            # Handle TOC queries
            if qtype == "table_of_contents":
                print("DEBUG: Routing to TOC handler")
                return self.handle_toc_query(system)

            # otherwise do semantic retrieval + format
            print("DEBUG: Using semantic search")
            docs = self.intelligent_retrieval(question, system["vectorstore"], system["intelligence"], k=6)
            return self.format_context_for_llama(question, docs, system["intelligence"], structure=system.get("structure"))
        except Exception as e:
            logger.error(f"Query error: {e}", exc_info=True)
            return f"Error answering query: {question}"


# Utility wrappers used by older main.py variants (keeps compatibility)
def process_pdf(pdf_path: str):
    agent = SmartRAGAgent()
    result = agent.process_pdf(pdf_path)
    if result:
        return result["vectorstore"]
    return None


def save_vectorstore(vectorstore, save_path="./vectorstore"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    vectorstore.save_local(save_path)


def load_vectorstore(load_path="./vectorstore"):
    if not os.path.exists(load_path):
        return None
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True}
    )
    return FAISS.load_local(load_path, embeddings)