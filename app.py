from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from functools import wraps
from werkzeug.utils import secure_filename
import os
import io
from zipfile import ZipFile
import logging
from pdf_operations.pdf_separator import run_process  # Ensure this imports the correct main function
from pdf_operations.pdf_merger import merge_pdfs_main  # Import the merge_pdfs function
from pdf_operations.pdf_splitter import split_pdf  # Import the split_pdf function

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.secret_key = os.environ.get('SECRET_KEY')  # Use environment variable for secret key, with a default for testing
PASSCODE = os.environ.get('PASSCODE')  # Use environment variable for passcode, with a default for testing
print(f'Secret Key: {app.secret_key}')
print(f'Passcode: {PASSCODE}')

logging.basicConfig(level=logging.INFO)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('authenticated'):
            return redirect(url_for('passcode'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/passcode', methods=['GET', 'POST'])
def passcode():
    if request.method == 'POST':
        entered_passcode = request.form['passcode']
        if entered_passcode == PASSCODE:
            session['authenticated'] = True
            logging.info("Authentication successful.")
            return redirect(url_for('index'))
        else:
            logging.warning("Authentication failed.")
            flash('Invalid passcode', 'danger')
    return render_template('passcode.html')

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/separator', methods=['GET', 'POST'])
@login_required
def separator():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file and uploaded_file.filename.endswith('.pdf'):
            filename = secure_filename(uploaded_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(file_path)
            try:
                output_files = run_process.RunProcess(file_path).output_files
                zip_buffer = io.BytesIO()
                with ZipFile(zip_buffer, 'w') as zip_file:
                    for file in output_files:
                        zip_file.write(file, os.path.basename(file))
                zip_buffer.seek(0)

                # Delete split PDF folder
                if output_files:
                    directory_path = os.path.dirname(output_files[0])

                    # Delete the folder and its contents
                    for file in os.listdir(directory_path):
                        os.remove(os.path.join(directory_path, file))
                    os.rmdir(directory_path)

                # Delete the uploaded PDF
                os.remove(file_path)

                return send_file(zip_buffer, as_attachment=True, download_name='separated_pdfs.zip', mimetype='application/zip')
            except Exception as e:
                logging.error(f'Error separating PDF: {str(e)}')
                flash(f'Error separating PDF: {str(e)}', 'danger')
        else:
            flash('Please select a valid PDF file.', 'danger')
    return render_template('separator.html')

@app.route('/merger', methods=['GET', 'POST'])
@login_required
def merger():
    if request.method == 'POST':
        uploaded_files = request.files.getlist('files')
        file_paths = []

        # Check if any files are uploaded
        if not uploaded_files or all(file.filename == '' for file in uploaded_files):
            flash('Please upload at least one PDF file.', 'danger')
            return redirect(request.url)  # Redirect to the same page to show the error message

        for uploaded_file in uploaded_files:
            if uploaded_file and uploaded_file.filename.endswith('.pdf'):
                filename = secure_filename(uploaded_file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                uploaded_file.save(file_path)
                file_paths.append(file_path)
                
        try:
            merged_file_path = merge_pdfs_main.merge_pdfs(file_paths)
            response = send_file(merged_file_path, as_attachment=True, download_name='merged.pdf', mimetype='application/pdf')

            # Delete the merged PDF
            os.remove(merged_file_path)

            # Delete the uploaded PDFs
            for file_path in file_paths:
                os.remove(file_path)

            return response
        except Exception as e:
            logging.error(f'Error merging PDFs: {str(e)}')
            flash(f'Error merging PDFs: {str(e)}', 'danger')
            # Delete the uploaded PDFs in case of error
            for file_path in file_paths:
                os.remove(file_path)

    return render_template('merger.html')

@app.route('/splitter', methods=['GET', 'POST'])
@login_required
def splitter():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        split_points = request.form['split_points']  # Get the split points from the form

        if uploaded_file and uploaded_file.filename.endswith('.pdf'):
            filename = secure_filename(uploaded_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(file_path)

            try:
                split_points = list(map(int, split_points.split(',')))  # Convert to a list of integers
                split_pdfs = split_pdf(file_path, split_points)

                zip_buffer = io.BytesIO()
                with ZipFile(zip_buffer, 'w') as zip_file:
                    for idx, file in enumerate(split_pdfs):
                        split_file_name = f'split_{idx + 1}.pdf'
                        zip_file.write(file, split_file_name)
                
                zip_buffer.seek(0)

                for file in split_pdfs:
                    os.remove(file)

                os.remove(file_path)

                return send_file(zip_buffer, as_attachment=True, download_name='split_pdfs.zip', mimetype='application/zip')
            except Exception as e:
                logging.error(f'Error splitting PDF: {str(e)}')
                flash(f'Error splitting PDF: {str(e)}', 'danger')
        else:
            flash('Please select a valid PDF file.', 'danger')
    
    return render_template('splitter.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
