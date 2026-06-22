'''
1. PyMuPDF (fitz) - The "Swiss Army Knife"
    Best for: General-purpose text extraction, speed, and reliability.
    ✅ Fastest pure Python extractor for most PDFs
    ✅ Handles both text-based and some complex layouts
    ✅ Can extract images, metadata, and annotations
    ❌ Struggles with heavily formatted tables
    Use when: You need a reliable, fast default for 90% of PDFs.

2. pypdfium2 - The "Speed Demon"
    Best for: Processing thousands of PDFs or very large files.
    ✅ Extremely fast (uses C++ backend)
    ✅ Low memory footprint
    ❌ Less accurate with complex formatting
    ❌ No table extraction
    Use when: Speed is critical and layout precision is secondary.

3. pdfplumber - The "Layout Specialist"
    Best for: Financial reports, invoices, and documents with tables.
    ✅ Best-in-class table extraction
    ✅ Preserves spatial layout (x,y coordinates)
    ✅ Great for structured data extraction
    ❌ Slower than PyMuPDF
    Use when: You need to extract data from tables or preserve exact positioning.

4. pymupdf4llm - The "RAG Optimizer"
    Best for: Preparing documents for LLMs and RAG systems.
    ✅ Converts PDFs directly to Markdown
    ✅ Preserves headers, lists, and code blocks
    ✅ Built on top of PyMuPDF but optimized for AI
    ❌ Newer, so fewer community examples
    Use when: You're building a RAG pipeline and want structure-aware chunks.

5. Unstructured / LlamaParse - The "Heavy Lifters"
    Best for: Production-grade, messy, or scanned documents.
    ✅ OCR built-in for scanned PDFs
    ✅ Handles mixed media (images + text)
    ✅ Auto-detects document type (invoice, article, etc.)
    ❌ Heavy dependencies, slower, often requires API keys
    Use when: You have no control over PDF quality or need enterprise-grade reliability.

6. pdf2image + pytesseract - The "OCR Last Resort"
    Best for: Scanned PDFs where other libraries fail.
    ✅ Works on any image-based PDF
    ❌ Very slow
    ❌ Requires installing Tesseract OCR on Windows
    ❌ Lower accuracy than cloud-based OCR
    Use when: The PDF is purely images (scanned books, old archives)
'''
