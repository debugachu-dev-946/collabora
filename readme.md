# Medical Form Processing Application

A Flask-based web application that processes medical forms using Google's Gemini AI for data extraction and presents the results in a structured format.

## Features

- Upload and process different types of medical forms (Form A-E)
- Image processing using Pillow (PIL)
- Data extraction using Google Gemini AI
- Responsive web interface with Bootstrap
- Tabular data presentation
- Support for multiple form types:
  - Form A: Outreach Clinic Data and Financial Information
  - Form B: ART Treatment and Services
  - Form C: Other Medical Services
  - Form D: HIV Counselling and Testing
  - Form E: Family Planning and Maternal Health

## Prerequisites

- Python 3.x
- Flask
- Pillow
- pandas
- numpy
- google-generativeai
- Internet connection for Bootstrap and FontAwesome CDN

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
```

2. Install the required packages:
```bash
pip install flask Pillow pandas numpy google-generativeai
```

3. Set up your environment:
```bash
# Create a directory for uploads
mkdir -p static/uploads
```

## Project Structure

├── app.py # Main Flask application
├── process_image.py # Image processing functions
├── static/
│ └── uploads/ # Directory for uploaded images
├── templates/
│ ├── index.html # Upload form template
│ └── results.html # Results display template
└── README.md


## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open a web browser and navigate to `http://localhost:5000`

3. Select the appropriate form type and upload an image

4. View the processed results in a tabulated format

## Form Types and Data Processing

- **Form A**: Processes outreach clinic data and financial information
- **Form B**: Handles ART treatment data and service availability
- **Form C**: Processes other medical services data
- **Form D**: Manages HIV counselling and testing information
- **Form E**: Handles family planning and maternal health data

## Contributing

Please feel free to submit issues and pull requests.

## Acknowledgments

- Google Gemini AI for image processing
- Bootstrap for UI components
- FontAwesome for icons