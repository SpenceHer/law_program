# Import necessary modules
from gui import GUI

def main():
    # Create an instance of the main application GUI class
    app = GUI()
    # Start the application
    app.run()

# Check if the script is running directly (not imported)
if __name__ == "__main__":
    main()