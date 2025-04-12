import os
import cv2
import numpy as np
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import shutil
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("document_classifier.log"),
        logging.StreamHandler()
    ]
)

# Path configuration
INPUT_FOLDER = "input_docs"
OUTPUT_FOLDER = "output_docs"
CATEGORIES = ["invoice", "tax-related", "agreements"]

# Configure Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image_path):
    """Preprocess image for better OCR results"""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            logging.error(f"Failed to read image: {image_path}")
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get black and white image
        _, threshold = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Apply noise reduction
        denoised = cv2.GaussianBlur(threshold, (5, 5), 0)
        
        return denoised
    except Exception as e:
        logging.error(f"Error preprocessing image {image_path}: {e}")
        return None

def extract_text(image):
    """Extract text from preprocessed image using OCR"""
    try:
        if image is None:
            return ""
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        logging.error(f"OCR error: {e}")
        return ""

def train_classifier():
    """Train a document classifier with sample data"""
    # In a real application, you would use actual labeled data
    # This is a simplified example with synthetic data
    
    # Sample texts for each category
    sample_texts = {
        "invoice": [
            "Invoice #12345 Total Amount: $500.00 Payment Due: 30 days",
            "INVOICE Bill To: Customer Name Items: Product A Quantity: 2 Amount: $150",
            "Invoice Date: 2023-01-15 Terms: Net 15 Please pay the amount of $250.50",
            "BILLING STATEMENT Customer ID: ABC123 Previous Balance: $0 Current Charges: $75.99",
            "RECEIPT Order #98765 Payment Method: Credit Card Total: $199.99"
        ],
        "tax-related": [
            "Form W-2 Wage and Tax Statement Employer Identification Number: 12-3456789",
            "ANNUAL TAX SUMMARY Tax Year: 2022 Total Income: $85,000 Tax Withheld: $12,750",
            "1099-MISC Miscellaneous Income Payer's TIN: 98-7654321 Amount: $2,500.00",
            "TAX DEDUCTION RECEIPT Charitable Contribution: $500 Date: December 15, 2022",
            "PROPERTY TAX STATEMENT Parcel ID: 123-45-678 Assessment Value: $320,000"
        ],
        "agreements": [
            "EMPLOYMENT CONTRACT Between Company XYZ and John Doe Effective Date: March 1, 2023",
            "RENTAL AGREEMENT Landlord: Property Management Inc. Tenant: Jane Smith Term: 12 months",
            "SERVICE AGREEMENT This contract outlines the terms and conditions between the service provider and client",
            "SOFTWARE LICENSE AGREEMENT The licensee agrees to the following terms of use for the software product",
            "NON-DISCLOSURE AGREEMENT This confidentiality agreement is made between the undersigned parties"
        ]
    }
    
    # Prepare training data
    texts = []
    labels = []
    
    for category, samples in sample_texts.items():
        for sample in samples:
            texts.append(sample)
            labels.append(category)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    
    # Vectorize the text
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train classifier
    classifier = SVC(kernel='linear')
    classifier.fit(X_train_vec, y_train)
    
    # Evaluate (in a real application, you'd want to report metrics)
    accuracy = classifier.score(X_test_vec, y_test)
    logging.info(f"Classifier trained with accuracy: {accuracy:.2f}")
    
    return vectorizer, classifier

def classify_document(text, vectorizer, classifier):
    """Classify document based on its text content"""
    if not text.strip():
        logging.warning("Empty text provided for classification")
        return "unclassified"
    
    # Transform text using the same vectorizer
    text_vec = vectorizer.transform([text])
    
    # Predict category
    category = classifier.predict(text_vec)[0]
    return category

def setup_folders():
    """Create necessary folders if they don't exist"""
    # Create input folder
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    
    # Create output folder and category subfolders
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    for category in CATEGORIES + ["unclassified"]:
        os.makedirs(os.path.join(OUTPUT_FOLDER, category), exist_ok=True)
    
    logging.info(f"Created folder structure: {INPUT_FOLDER} and {OUTPUT_FOLDER} with category subfolders")

def move_file(file_path, category):
    """Move file to the appropriate category folder"""
    file_name = os.path.basename(file_path)
    destination = os.path.join(OUTPUT_FOLDER, category, file_name)
    
    try:
        shutil.copy2(file_path, destination)
        logging.info(f"Moved {file_name} to {category} folder")
        return True
    except Exception as e:
        logging.error(f"Error moving file {file_name}: {e}")
        return False

def process_documents():
    """Main function to process all documents in the input folder"""
    logging.info("Document classification process started")
    
    # Setup folder structure
    setup_folders()
    
    # Train the classifier
    vectorizer, classifier = train_classifier()
    
    # Get list of files in input folder
    input_path = Path(INPUT_FOLDER)
    files = list(input_path.glob('*.*'))
    
    if not files:
        logging.warning(f"No files found in {INPUT_FOLDER}")
        return
    
    logging.info(f"Found {len(files)} files to process")
    
    # Process each file
    for file_path in files:
        file_str = str(file_path)
        file_ext = file_path.suffix.lower()
        
        # Check if it's an image file
        image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.pdf']
        if file_ext in image_extensions:
            logging.info(f"Processing {file_path.name}")
            
            # Preprocess image
            processed_image = preprocess_image(file_str)
            
            # Extract text
            document_text = extract_text(processed_image)
            
            # Classify document
            category = classify_document(document_text, vectorizer, classifier)
            
            # Move file to appropriate folder
            move_file(file_str, category)
        else:
            logging.warning(f"Skipping unsupported file: {file_path.name}")
    
    logging.info("Document classification completed")

if __name__ == "__main__":
    process_documents()
    