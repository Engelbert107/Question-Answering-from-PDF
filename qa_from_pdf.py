import re
import io
import fitz  # PyMuPDF
import pdfplumber 
import pytesseract 
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


# Extract text using pdfplumber
def extract_text_with_pdfplumber(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract text from each page
            page_text = page.extract_text()
            if page_text:  # Ensure text is not None
                text += page_text + "\n"
    return text

# Extract images using PyMuPDF (for OCR fallback)
def extract_images(pdf_path):
    images = []
    pdf_document = fitz.open(pdf_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)
    return images

# Apply OCR to images using Tesseract
def ocr_images(images):
    ocr_text = ""
    for image in images:
        text = pytesseract.image_to_string(image, config='--psm 6')  # PSM 6 for block of text
        ocr_text += text + "\n"
    return ocr_text

# Clean and preprocess text
def clean_text(text):
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters, but keep punctuation and numbers
    text = re.sub(r'[^a-zA-Z0-9\s.,;!?]', '', text)
    return text

# Main workflow
def process_pdf(pdf_path):
    # Try extracting text using pdfplumber
    text = extract_text_with_pdfplumber(pdf_path)

    # If no text is found or text extraction is poor, fall back to OCR
    if not text or len(text.strip()) < 10:  # Check if text is empty or too short
        print("No text found using pdfplumber. Falling back to OCR...")
        images = extract_images(pdf_path)
        ocr_text = ocr_images(images)
        text = ocr_text

    # Clean and preprocess text
    cleaned_text = clean_text(text)
    return cleaned_text




if __name__ == "__main__":
    # Path to your PDF
    pdf_path = "data/cancerQA.pdf"
    extracted_text = process_pdf(pdf_path)

    # Load the model and tokenizer
    model_name = "deepset/bert-base-cased-squad2"

    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
    
    # Define the question and context
    # question = 'What is the main subject of this paper?'
    question = 'What large model is used in this paper?'
    context = extracted_text

    # Get the answer
    res = nlp(question=question, context=context)

    # Print results
    print("\nQuestion:", question)
    print("Answer:", res['answer']) 
    print("Answer with score:", res,"\n") 
