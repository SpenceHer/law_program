# pdf_operations/pdf_separator/main.py
from .run_process import RunProcess

def main(file_path):
    print(f"Processing file: {file_path}")
    run_process = RunProcess(file_path)
    print(f"Output files: {run_process.output_files}")
    return run_process.output_files

if __name__ == "__main__":
    import sys
    main(sys.argv[1])