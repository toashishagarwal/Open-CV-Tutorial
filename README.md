<b>About</b> <br>
This project showcases the classification of input documents (/input_docs) into separate folders (tax-related, invoice, agreements, unclassified) based on the contents of the input document. The documents that a fed into the app are image formats of the text.

The app uses a Support Vector Machine (SVC) classifier & transforms document text using TF-IDF vectorization to predict the document category

Techology Stack
* Python
* Terreseract
* OpenCV

<b>Setup Instructions </b>
1. Create a python virtual env called venv
python -m venv venv

2. Activate the virtual env (for windows)
.\venv\Scripts\Activate.ps1 

    (If you get permission error, you may have to run the below command)
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

3. Install the dependencies
pip install -r requirements.txt

4. Install Tesseract OCR for Windows
Download the installer from Tesseract GitHub releases (https://github.com/UB-Mannheim/tesseract/wiki)
Run the installer and remember the installation path (default is usually C:\Program Files\Tesseract-OCR)

<b>Run Instructions</b> <br>
1. Place the image documents in the input_docs folder
2. Run the application
    python document_classifier.main
3. check the output_docs folder to see your classified documents
