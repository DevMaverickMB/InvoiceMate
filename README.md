# InvoiceMate - AI Invoice Analyzer

![🌐LIVE]([https://invoicemate.streamlit.app])

InvoiceMate is an AI-powered tool built with Streamlit and Google Gemini API to analyze and extract key details from invoice images. It processes single or multiple invoices, provides structured data extraction, and offers insights through an intuitive dashboard. Users can upload invoice images, ask questions about the data, and export results in CSV, PDF, or JSON formats.


![screencapture-localhost-8501-2025-03-11-13_15_51](https://github.com/user-attachments/assets/24e6f841-a826-44eb-8f1a-12f0f2851ec4)

![screencapture-localhost-8501-2025-03-11-13_17_20](https://github.com/user-attachments/assets/d1935935-f677-447d-b9db-5e27826e6a93)




## Features

- **Invoice Data Extraction**: Extracts key details such as invoice number, date, vendor name, total amount, and tax amount from uploaded images.
- **Single & Batch Processing**: Handles both individual invoices and multiple invoices with aggregate insights.
- **Interactive Dashboard**: Displays metrics like total amount, total tax, and number of vendors.
- **Custom Queries**: Allows users to ask specific questions about the invoices (e.g., "What's the total amount across all invoices?").
- **Export Options**: Generate reports in CSV, PDF, or JSON formats.
- **User-Friendly UI**: Built with Streamlit, featuring a clean and responsive design with custom CSS.

## Tech Stack

- **Python**: Core programming language.
- **Streamlit**: Web application framework for the UI.
- **Google Gemini API**: AI model for invoice analysis and data extraction (`gemini-2.0-flash`).
- **Pandas**: Data manipulation and CSV export.
- **Plotly**: Data visualization (optional, not fully utilized in current code).
- **ReportLab**: PDF report generation.
- **PIL (Pillow)**: Image processing.
- **dotenv**: Environment variable management.

## Usage
- **Upload Invoices**:
Go to the "Upload & Process" tab.
Upload one or more invoice images (JPG, JPEG, PNG).
Optionally, enter a query (e.g., "What’s the total tax amount?").
- **Process Invoices**:
Click "Process Invoices" to extract data and get AI-generated responses.
View extracted data and any query responses.
- **Analyze Insights**:
Switch to the "Analysis & Insights" tab to see a dashboard with metrics.
Ask follow-up questions about the processed invoices.
- **Export Results**:
Go to the "Export & Reports" tab.
Choose CSV, PDF, or JSON format and download the results.

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key (sign up at [Google AI Studio](https://aistudio.google.com/) to obtain one)

