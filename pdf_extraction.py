from langchain_community.document_loaders import PyPDFLoader

def extract_pdf(pdf_path):

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    text = ""

    for page in pages:
        text += page.page_content

    return text


text = extract_pdf("ads_cookbook.pdf")

print(len(text))