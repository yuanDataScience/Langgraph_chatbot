from pypdf import PdfReader

# this is not an asynchronous function and can't run
# in an asynchronous event loop since it will block the event loop!!
def pdf_text_extractor(filepath: str) -> None:
    """
    This function read pdf file and convert it to plain text
    """
    content = ""
    pdf_reader = PdfReader(filepath, strict=True)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            content += f"{page_text}\n\n"
    with open(filepath.replace("pdf", "txt"), "w", encoding="utf-8") as file:
        file.write(content)