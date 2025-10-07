# improved_intelligence.py - Separate improved class for easy testing
import re
from collections import defaultdict
from typing import List, Dict, Any, Set, Tuple
import logging

class ImprovedDocumentIntelligence:
    """
    Improved document intelligence extraction with better accuracy, performance, and maintainability.
    Can be used as a drop-in replacement for the extract_document_intelligence method.
    """
    
    def __init__(self):
        # Compiled regex patterns for better performance
        self.title_patterns = [
            (re.compile(r"^(?:project\s+)?title:\s*(.+)$", re.IGNORECASE), "explicit"),
            (re.compile(r"^title:\s*(.+)$", re.IGNORECASE), "explicit"),
            (re.compile(r"^[A-Z][A-Za-z\s\-:]{10,100}$", re.IGNORECASE), "heuristic"),
            (re.compile(r"^\d+\.\s+[A-Z][A-Za-z\s\-:]{5,80}$", re.IGNORECASE), "heuristic"),
            (re.compile(r"^[A-Z\s]{5,50}$", re.IGNORECASE), "heuristic"),
        ]
        
        self.metadata_patterns = {
            "author": [
                re.compile(r"author[:\s]*(.+)$", re.IGNORECASE),
                re.compile(r"by[:\s]*(.+)$", re.IGNORECASE)
            ],
            "date": [
                re.compile(r"date[:\s]*(.+)$", re.IGNORECASE),
                re.compile(r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b"),
                re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),
            ],
            "project": [
                re.compile(r"project[:\s]*(.+)$", re.IGNORECASE),
                re.compile(r"project\s+title[:\s]*(.+)$", re.IGNORECASE)
            ],
            "version": [
                re.compile(r"version[:\s]*(.+)$", re.IGNORECASE),
                re.compile(r"\b(v\d+\.\d+(?:\.\d+)?)\b", re.IGNORECASE)
            ],
            "company": [
                re.compile(r"company[:\s]*(.+)$", re.IGNORECASE),
                re.compile(r"organization[:\s]*(.+)$", re.IGNORECASE)
            ],
        }
        
        # Improved heading patterns
        self.heading_patterns = [
            re.compile(r"^(\d+(?:\.\d+)*\.?\s+.+)$"),  # Numbered: 1.2.3 Title
            re.compile(r"^([A-Z][A-Z\s]{2,49}[A-Z])$"),  # ALL CAPS (not single words)
            re.compile(r"^([A-Z][a-z].*[^.!?:])$"),  # Title Case without ending punctuation
        ]
        
        # Common words to filter out from key terms
        self.common_words = {
            'The', 'This', 'That', 'These', 'Those', 'And', 'But', 'For', 
            'Not', 'With', 'From', 'They', 'Been', 'Have', 'Has', 'Had', 
            'Will', 'Would', 'Could', 'Should', 'May', 'Might', 'Can',
            'Project', 'Document', 'Page', 'Section', 'Chapter', 'Part'
        }
        
        self.logger = logging.getLogger(__name__)

    def extract_document_intelligence(self, documents: List) -> Dict[str, Any]:
        """
        Extract structured intelligence from documents with improved accuracy and performance.
        Returns dictionary in same format as original method for compatibility.
        """
        intelligence = {
            "titles": [],
            "headings": [],
            "metadata": defaultdict(list),
            "first_page_content": "",
            "key_terms": set(),
        }

        try:
            for i, doc in enumerate(documents):
                if not hasattr(doc, 'page_content'):
                    self.logger.warning(f"Document {i} missing page_content attribute")
                    continue
                    
                content = doc.page_content or ""
                if not content.strip():
                    continue
                    
                # Extract first page content more intelligently
                if i == 0:
                    intelligence["first_page_content"] = self._extract_first_page_content(content)

                self._process_document_content(content, intelligence, line_limit=i < 10)

        except Exception as e:
            self.logger.error(f"Error processing documents: {e}")
            
        # Convert to serializable format and clean up
        return self._finalize_intelligence(intelligence)

    def _extract_first_page_content(self, content: str) -> str:
        """Extract first page content at natural breakpoints."""
        if len(content) <= 2000:
            return content
            
        # Try to break at paragraph or sentence boundaries
        truncated = content[:2000]
        
        # Find last complete sentence
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')
        
        breakpoint = max(last_period, last_newline)
        if breakpoint > 1500:  # Reasonable minimum
            return content[:breakpoint + 1]
        
        return truncated + "..."

    def _process_document_content(self, content: str, intelligence: Dict[str, Any], line_limit: bool = False) -> None:
        """Process document content line by line with improved efficiency."""
        lines = content.split("\n")
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Extract titles (only from first few lines for efficiency)
            if line_limit and line_num < 15:
                self._extract_titles(line, intelligence)
            
            # Extract metadata
            self._extract_metadata(line, intelligence)
            
            # Extract headings
            self._extract_headings(line, intelligence)
            
            # Extract key terms
            self._extract_key_terms(line, intelligence)

    def _extract_titles(self, line: str, intelligence: Dict[str, Any]) -> None:
        """Extract titles using improved pattern matching."""
        for pattern, source in self.title_patterns:
            match = pattern.search(line)
            if match:
                if match.groups():
                    title = match.group(1).strip()
                else:
                    title = line.strip()
                    
                if title and len(title) > 3:  # Filter very short titles
                    intelligence["titles"].append((title, source))
                    break  # Only take first match per line

    def _extract_metadata(self, line: str, intelligence: Dict[str, Any]) -> None:
        """Extract metadata with better normalization."""
        for meta_type, patterns in self.metadata_patterns.items():
            for pattern in patterns:
                match = pattern.search(line)
                if match:
                    if match.groups():
                        meta_value = match.group(1).strip()
                    else:
                        meta_value = match.group(0).strip()
                    
                    # Normalize and deduplicate
                    meta_value = self._normalize_metadata_value(meta_value, meta_type)
                    if meta_value and meta_value not in intelligence["metadata"][meta_type]:
                        intelligence["metadata"][meta_type].append(meta_value)

    def _normalize_metadata_value(self, value: str, meta_type: str) -> str:
        """Normalize metadata values based on type."""
        if meta_type == "date":
            value = re.sub(r'\s+', ' ', value)  # Normalize whitespace
        elif meta_type == "author":
            value = re.sub(r'^(?:by\s+|author:\s*)', '', value, flags=re.IGNORECASE)
            
        return value.strip()

    def _extract_headings(self, line: str, intelligence: Dict[str, Any]) -> None:
        """Extract headings using improved heuristics."""
        if len(line) > 100:  # Skip very long lines
            return
            
        for pattern in self.heading_patterns:
            if pattern.match(line):
                if not self._is_likely_heading(line):
                    continue
                    
                if line not in intelligence["headings"]:  # Deduplicate
                    intelligence["headings"].append(line)
                break

    def _is_likely_heading(self, line: str) -> bool:
        """Additional heuristics to filter out false positive headings."""
        # Skip if it looks like a sentence (ends with punctuation)
        if line.endswith(('.', '!', '?')):
            return False
            
        # Skip if it contains too many common words
        words = line.split()
        if len(words) > 2:
            common_count = sum(1 for word in words if word in self.common_words)
            if common_count > len(words) * 0.5:
                return False
                
        return True

    def _extract_key_terms(self, line: str, intelligence: Dict[str, Any]) -> None:
        """Extract key terms with better filtering."""
        # Improved regex for capitalized terms and acronyms
        capitalized_terms = re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b", line)
        acronyms = re.findall(r"\b[A-Z]{2,}\b", line)
        
        all_terms = capitalized_terms + acronyms
        
        for term in all_terms:
            # Filter out common words and very short terms
            if (len(term) > 2 and 
                term not in self.common_words and 
                not term.isdigit()):
                intelligence["key_terms"].add(term)

    def _finalize_intelligence(self, intelligence: Dict[str, Any]) -> Dict[str, Any]:
        """Convert to serializable format and clean up data."""
        # Convert defaultdict to regular dict
        intelligence["metadata"] = dict(intelligence["metadata"])
        
        # Convert set to sorted list for consistent output
        intelligence["key_terms"] = sorted(list(intelligence["key_terms"]))
        
        # Deduplicate and prioritize titles
        intelligence["titles"] = self._deduplicate_titles(intelligence["titles"])
        
        # Limit results to prevent excessive data
        intelligence["headings"] = intelligence["headings"][:50]
        intelligence["key_terms"] = intelligence["key_terms"][:100]
        
        return intelligence

    def _deduplicate_titles(self, titles: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Deduplicate titles and prioritize explicit ones."""
        seen = set()
        explicit_titles = []
        heuristic_titles = []
        
        for title, source in titles:
            if title.lower() not in seen:
                seen.add(title.lower())
                if source == "explicit":
                    explicit_titles.append((title, source))
                else:
                    heuristic_titles.append((title, source))
        
        return explicit_titles + heuristic_titles[:3]


# Enhanced SmartRAGAgent with integrated improvements
class EnhancedSmartRAGAgent:
    """
    Enhanced version of SmartRAGAgent with improved document intelligence extraction.
    Drop-in replacement for your existing SmartRAGAgent class.
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
        self.use_improved_intelligence = use_improved_intelligence

        try:
            from sentence_transformers import SentenceTransformer
            from langchain_community.embeddings import HuggingFaceEmbeddings
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to initialize embeddings: {e}")
            raise

        # Initialize improved intelligence extractor if requested
        if self.use_improved_intelligence:
            self.intelligence_extractor = ImprovedDocumentIntelligence()
        
        # Keep original patterns for backwards compatibility
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

        # query type detection: includes structure_check
        self.query_types = {
            "title": ["title", "what is this", "name of document", "project title"],
            "summary": ["summary", "about", "main topic", "overview"],
            "author": ["author", "who wrote", "written by"],
            "date": ["date", "when", "published"],
            "specific_fact": ["what is", "how much", "how many", "where", "when"],
            "list": ["list", "enumerate", "what are", "types of"],
            "explanation": ["explain", "how", "why", "describe"],
            "structure_check": [
                "table of contents",
                "toc",
                "structure",
                "sections match",
                "document structure",
                "contents match",
            ],
        }

    def extract_document_intelligence(self, documents: List) -> Dict[str, Any]:
        """
        Extract document intelligence using improved or original method.
        """
        if self.use_improved_intelligence:
            return self.intelligence_extractor.extract_document_intelligence(documents)
        else:
            # Original implementation (unchanged)
            return self._extract_document_intelligence_original(documents)
    
    def _extract_document_intelligence_original(self, documents: List) -> Dict[str, Any]:
        """Original implementation kept for comparison and fallback."""
        from collections import defaultdict
        import re
        
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

        return intelligence

    # All other methods remain the same as your original SmartRAGAgent
    def extract_document_structure(self, documents: List) -> Dict[str, Any]:
        """Keep original structure extraction method."""
        structure = {
            "toc_sections": [],
            "body_sections": [],
            "missing_in_body": [],
            "missing_in_toc": [],
            "structure_issues": [],
        }

        toc_found = False
        toc_content = ""

        # find a page that looks like TOC (first 5 pages)
        for doc in documents[:5]:
            content_lower = (doc.page_content or "").lower()
            if any(kw in content_lower for kw in ["table of contents", "contents", "table of content"]):
                toc_found = True
                toc_content = doc.page_content or ""
                break

        if toc_found:
            # parse TOC lines for patterns like "2.1   Title .... 12"
            import re
            for line in (toc_content or "").split("\n"):
                line = line.strip()
                if not line:
                    continue
                m = re.match(r"^\s*(\d+(?:\.\d+)*)\s+(.+?)\s*(?:\.*\s*(\d+))?\s*$", line)
                if m:
                    section_num = m.group(1)
                    section_title = m.group(2).strip()
                    page_num = m.group(3)
                    structure["toc_sections"].append({"number": section_num, "title": section_title, "page": page_num})

        # parse body headings from all pages
        import re
        for idx, doc in enumerate(documents):
            page_num = doc.metadata.get("page", idx)
            for line in (doc.page_content or "").split("\n"):
                line = line.strip()
                if not line:
                    continue
                # match headings like "2.1 Experimental beam tests"
                m = re.match(r"^\s*(\d+(?:\.\d+)*)\s+(.{1,200})\s*$", line)
                if m:
                    section_num = m.group(1)
                    section_title = re.sub(r"\s+", " ", m.group(2)).rstrip(".")
                    structure["body_sections"].append({"number": section_num, "title": section_title, "page": page_num})

        # compare
        if structure["toc_sections"] and structure["body_sections"]:
            toc_numbers = {s["number"] for s in structure["toc_sections"]}
            body_numbers = {s["number"] for s in structure["body_sections"]}

            structure["missing_in_body"] = [s for s in structure["toc_sections"] if s["number"] not in body_numbers]
            structure["missing_in_toc"] = [s for s in structure["body_sections"] if s["number"] not in toc_numbers]

            if structure["missing_in_body"]:
                structure["structure_issues"].append(
                    f"TOC lists {len(structure['missing_in_body'])} sections not found in body"
                )
            if structure["missing_in_toc"]:
                structure["structure_issues"].append(
                    f"Body has {len(structure['missing_in_toc'])} sections not in TOC"
                )

        return structure

    # All other methods (classify_query, smart_chunk_documents, etc.) remain exactly the same
    # ... (keeping your original implementations)


# Usage example and testing utilities
def compare_intelligence_extraction(documents: List, verbose: bool = True):
    """
    Compare original vs improved intelligence extraction for testing.
    """
    # Original method
    original_agent = EnhancedSmartRAGAgent(use_improved_intelligence=False)
    original_result = original_agent.extract_document_intelligence(documents)
    
    # Improved method
    improved_agent = EnhancedSmartRAGAgent(use_improved_intelligence=True)
    improved_result = improved_agent.extract_document_intelligence(documents)
    
    if verbose:
        print("=== COMPARISON RESULTS ===")
        print(f"Original titles: {len(original_result['titles'])}")
        print(f"Improved titles: {len(improved_result['titles'])}")
        print(f"Original headings: {len(original_result['headings'])}")
        print(f"Improved headings: {len(improved_result['headings'])}")
        print(f"Original key terms: {len(original_result['key_terms'])}")
        print(f"Improved key terms: {len(improved_result['key_terms'])}")
        
        print("\n=== SAMPLE IMPROVEMENTS ===")
        print("Improved titles:", improved_result['titles'][:3])
        print("Improved key terms:", improved_result['key_terms'][:10])
    
    return {
        "original": original_result,
        "improved": improved_result,
        "improvements": {
            "titles": len(improved_result['titles']) - len(original_result['titles']),
            "headings": len(improved_result['headings']) - len(original_result['headings']),
            "key_terms": len(improved_result['key_terms']) - len(original_result['key_terms']),
        }
    }