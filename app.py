# Set page config must be the first Streamlit command
import streamlit as st
st.set_page_config(
    page_title="InvoiceMate - AI Invoice Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

from dotenv import load_dotenv
import os
from PIL import Image
import google.generativeai as genai
import pandas as pd
import time
import plotly.express as px
from datetime import datetime
import io
import base64

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Gemini model with caching
@st.cache_resource
def load_model():
    return genai.GenerativeModel('gemini-2.0-flash')

model = load_model()

# Process single invoice
def get_gemini_response(input_prompt, image, user_query):
    try:
        response = model.generate_content([input_prompt, image[0], user_query])
        return response.text
    except Exception as e:
        return f"Error processing invoice: {str(e)}"

# Process multiple invoices and combine results
def process_multiple_invoices(input_prompt, images, user_query):
    all_invoice_data = []
    
    with st.status("Processing invoices...", expanded=True) as status:
        total_invoices = len(images)
        
        for i, image in enumerate(images):
            status.update(label=f"Processing invoice {i+1}/{total_invoices}")
            
            # Extract structured data from invoice
            extraction_query = "Extract the following information in JSON format: invoice_number, date, vendor_name, total_amount (as a numeric value), tax_amount (as a numeric value)"
            invoice_json_prompt = f"{input_prompt}\n{extraction_query}"
            
            try:
                response = model.generate_content([invoice_json_prompt, image])
                all_invoice_data.append(response.text)
                time.sleep(0.5)  # Small delay to prevent rate limiting
            except Exception as e:
                st.error(f"Error processing invoice {i+1}: {str(e)}")
        
        status.update(label="Processing complete!", state="complete")
    
    # If user has a specific query about all invoices
    if user_query:
        combined_query = f"{input_prompt}\nBased on all {len(images)} invoices, please answer: {user_query}"
        try:
            # We pass all images together with the query for multi-invoice analysis
            response = model.generate_content([combined_query] + images)
            return response.text, all_invoice_data
        except Exception as e:
            return f"Error analyzing multiple invoices: {str(e)}", all_invoice_data
    
    return "All invoices processed successfully. You can ask questions about them now.", all_invoice_data

# Process image for API
def input_image_details(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No File Uploaded")

# Extract key metrics from invoice data for dashboard
def extract_metrics_from_invoices(invoice_data_list):
    try:
        import json
        import re
        
        # Initialize metrics
        total_amount = 0
        total_tax = 0
        vendors = set()
        dates = []
        invoice_numbers = []
        
        # Regular expressions to extract data if JSON parsing fails
        amount_pattern = r'"total_amount":\s*([\d\.]+)'
        tax_pattern = r'"tax_amount":\s*([\d\.]+)'
        vendor_pattern = r'"vendor_name":\s*"([^"]+)"'
        date_pattern = r'"date":\s*"([^"]+)"'
        invoice_pattern = r'"invoice_number":\s*"([^"]+)"'
        
        for invoice_data in invoice_data_list:
            try:
                # Try to extract JSON data
                # Find JSON in the text
                json_match = re.search(r'\{.*\}', invoice_data, re.DOTALL)
                if json_match:
                    invoice_json = json.loads(json_match.group())
                    
                    # Extract data from JSON
                    if 'total_amount' in invoice_json:
                        total_amount += float(invoice_json['total_amount'])
                    if 'tax_amount' in invoice_json:
                        total_tax += float(invoice_json['tax_amount'])
                    if 'vendor_name' in invoice_json:
                        vendors.add(invoice_json['vendor_name'])
                    if 'date' in invoice_json:
                        dates.append(invoice_json['date'])
                    if 'invoice_number' in invoice_json:
                        invoice_numbers.append(invoice_json['invoice_number'])
                else:
                    # If JSON parsing fails, use regex
                    amount_match = re.search(amount_pattern, invoice_data)
                    if amount_match:
                        total_amount += float(amount_match.group(1))
                    
                    tax_match = re.search(tax_pattern, invoice_data)
                    if tax_match:
                        total_tax += float(tax_match.group(1))
                    
                    vendor_match = re.search(vendor_pattern, invoice_data)
                    if vendor_match:
                        vendors.add(vendor_match.group(1))
                    
                    date_match = re.search(date_pattern, invoice_data)
                    if date_match:
                        dates.append(date_match.group(1))
                    
                    invoice_match = re.search(invoice_pattern, invoice_data)
                    if invoice_match:
                        invoice_numbers.append(invoice_match.group(1))
                        
            except json.JSONDecodeError:
                # If JSON parsing fails, use regex
                amount_match = re.search(amount_pattern, invoice_data)
                if amount_match:
                    total_amount += float(amount_match.group(1))
                
                tax_match = re.search(tax_pattern, invoice_data)
                if tax_match:
                    total_tax += float(tax_match.group(1))
                
                vendor_match = re.search(vendor_pattern, invoice_data)
                if vendor_match:
                    vendors.add(vendor_match.group(1))
                
                date_match = re.search(date_pattern, invoice_data)
                if date_match:
                    dates.append(date_match.group(1))
                
                invoice_match = re.search(invoice_pattern, invoice_data)
                if invoice_match:
                    invoice_numbers.append(invoice_match.group(1))
        
        return {
            "total_amount": round(total_amount, 2),
            "total_tax": round(total_tax, 2),
            "num_vendors": len(vendors),
            "vendors": list(vendors),
            "dates": dates,
            "invoice_numbers": invoice_numbers
        }
    except Exception as e:
        st.error(f"Error extracting metrics: {str(e)}")
        return {
            "total_amount": 0,
            "total_tax": 0,
            "num_vendors": 0,
            "vendors": [],
            "dates": [],
            "invoice_numbers": []
        }

# CSV export function
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# PDF report generation function
def generate_invoice_report(metrics, invoice_data_list):
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from io import BytesIO
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Add title
    title_style = styles["Heading1"]
    elements.append(Paragraph("Invoice Analysis Report", title_style))
    elements.append(Spacer(1, 12))
    
    # Add date
    date_style = styles["Normal"]
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", date_style))
    elements.append(Spacer(1, 12))
    
    # Add summary section
    elements.append(Paragraph("Summary", styles["Heading2"]))
    elements.append(Spacer(1, 6))
    
    summary_data = [
        ["Total Invoices", str(len(invoice_data_list))],
        ["Total Amount", f"${metrics['total_amount']}"],
        ["Total Tax", f"${metrics['total_tax']}"],
        ["Number of Vendors", str(metrics['num_vendors'])],
        ["Vendors", ", ".join(metrics['vendors'])]
    ]
    
    summary_table = Table(summary_data, colWidths=[200, 300])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 12))
    
    # Add invoice details
    elements.append(Paragraph("Invoice Details", styles["Heading2"]))
    elements.append(Spacer(1, 6))
    
    # Create a table for invoice details
    if metrics['invoice_numbers'] and metrics['dates']:
        invoice_details = [["Invoice Number", "Date"]]
        for i in range(min(len(metrics['invoice_numbers']), len(metrics['dates']))):
            invoice_details.append([metrics['invoice_numbers'][i], metrics['dates'][i]])
        
        invoice_table = Table(invoice_details, colWidths=[250, 250])
        invoice_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(invoice_table)
    
    # Build the PDF
    doc.build(elements)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data

# Custom CSS for better UI
def apply_custom_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    .stat-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 1rem;
        color: #616161;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-weight: 500;
        padding: 10px 20px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar UI components
def create_sidebar():
    with st.sidebar:
        # st.markdown('<div class="sub-header">Configuration</div>', unsafe_allow_html=True)
        
        # # API key input
        # api_key = st.text_input("Google Gemini API Key", value=os.getenv("GOOGLE_API_KEY"), type="password")
        # if api_key:
        #     os.environ["GOOGLE_API_KEY"] = api_key
        #     genai.configure(api_key=api_key)
        
        # # Model selection
        # model_option = st.selectbox(
        #     "Select Gemini Model",
        #     ["gemini-2.0-flash", "gemini-2.0-pro", "gemini-2.0-vision"],
        #     index=0
        # )
        
        # # Advanced options
        # st.markdown('<div class="sub-header">Advanced Options</div>', unsafe_allow_html=True)
        # enable_batch_processing = st.checkbox("Enable Batch Processing", value=True)
        
        # max_tokens = st.slider("Max Output Tokens", 100, 2048, 1024)
        # temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
        
        # st.markdown("---")
        st.markdown("### About InvoiceMate")
        st.markdown("InvoiceMate is an AI-powered tool that extracts and organizes key invoice details with precision. Simply upload an invoice image, and our system will analyze due dates, amounts, and payment terms, presenting them in a clear, structured format. Designed for simplicity and efficiency, it helps businesses and individuals manage invoices effortlessly. 🚀")

# Upload & Process Tab UI
def upload_process_tab():
    st.markdown('<div class="sub-header">Upload Invoices</div>', unsafe_allow_html=True)
    
    # File uploader
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Choose Invoice Images (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Show uploaded images in a grid
        cols = st.columns(min(3, len(uploaded_files)))
        for i, uploaded_file in enumerate(uploaded_files):
            with cols[i % 3]:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Invoice {i+1}", use_container_width=True, output_format="JPEG")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Query input
    user_query = st.text_input("What would you like to know about these invoices?", 
                               placeholder="e.g., What's the total amount across all invoices?")
    
    # Process button
    col1, col2 = st.columns([1, 1])
    with col1:
        process_button = st.button("Process Invoices", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("Clear All", type="secondary", use_container_width=True)
    
    # Input prompt template
    input_prompt = """
    You are an AI specialized in understanding, analyzing, and extracting data from invoices. 
    Your task is to accurately interpret the details provided in invoice images, including:
    - Vendor information (name, address, contact)
    - Invoice number
    - Invoice date
    - Line items with descriptions
    - Prices, quantities, and subtotals
    - Total amount
    - Tax information
    - Payment terms and methods
    
    Provide structured, accurate data extraction and answer questions about the invoices precisely.
    If handling multiple invoices, analyze relationships between them and provide aggregate insights.
    """
    
    # Process invoices
    if process_button and uploaded_files:
        # Clear previous results
        if clear_button:
            st.session_state.processed_invoices = []
            st.session_state.invoice_images = []
            st.session_state.invoice_data = []
            st.session_state.metrics = None
        
        # Prepare images for processing
        all_images = []
        for uploaded_file in uploaded_files:
            try:
                image_data = input_image_details(uploaded_file)
                all_images.append(image_data[0])
                
                # Store original images for display
                if uploaded_file not in st.session_state.invoice_images:
                    st.session_state.invoice_images.append(uploaded_file)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
        
        if all_images:
            # Start processing
            with st.spinner("Processing invoices..."):
                if len(all_images) == 1:
                    # Process single invoice
                    response = get_gemini_response(input_prompt, [all_images[0]], user_query)
                    st.session_state.processed_invoices.append(response)
                else:
                    # Process multiple invoices
                    response, invoice_data = process_multiple_invoices(input_prompt, all_images, user_query)
                    st.session_state.processed_invoices.append(response)
                    st.session_state.invoice_data = invoice_data
                    
                    # Extract metrics for dashboard
                    st.session_state.metrics = extract_metrics_from_invoices(invoice_data)
            
            # Show success message
            st.success(f"Successfully processed {len(all_images)} invoice(s)!")
            
            # Display response
            st.markdown("### AI Analysis")
            st.markdown(response)
    
    # Clear all data if clear button is pressed
    if clear_button:
        st.session_state.processed_invoices = []
        st.session_state.invoice_images = []
        st.session_state.invoice_data = []
        st.session_state.metrics = None
        st.success("All data cleared!")

# Analysis & Insights Tab UI
def analysis_insights_tab():
    if st.session_state.metrics:
        st.markdown('<div class="sub-header">Invoice Dashboard</div>', unsafe_allow_html=True)
        
        # Display metrics in a dashboard
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.markdown(f'<p class="metric-value">${st.session_state.metrics["total_amount"]}</p>', unsafe_allow_html=True)
            st.markdown('<p class="metric-label">Total Amount</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.markdown(f'<p class="metric-value">${st.session_state.metrics["total_tax"]}</p>', unsafe_allow_html=True)
            st.markdown('<p class="metric-label">Total Tax</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.markdown(f'<p class="metric-value">{st.session_state.metrics["num_vendors"]}</p>', unsafe_allow_html=True)
            st.markdown('<p class="metric-label">Vendors</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Create a dataframe from the extracted invoice data
        if st.session_state.metrics["invoice_numbers"] and st.session_state.metrics["dates"]:
            st.markdown("### Invoice Summary")
            
            data = []
            for i in range(min(len(st.session_state.metrics["invoice_numbers"]), len(st.session_state.metrics["dates"]))):
                data.append({
                    "Invoice Number": st.session_state.metrics["invoice_numbers"][i],
                    "Date": st.session_state.metrics["dates"][i],
                })
            
            if data:
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
        
        # Additional analysis section
        st.markdown("### Ask More Questions")
        follow_up_query = st.text_input("Ask another question about the invoices", 
                                      placeholder="e.g., Which vendor has the highest invoice amount?")
        
        if follow_up_query:
            if st.button("Get Answer", type="primary"):
                with st.spinner("Analyzing..."):
                    # Prepare all images again
                    all_images = []
                    for uploaded_file in st.session_state.invoice_images:
                        try:
                            image_data = input_image_details(uploaded_file)
                            all_images.append(image_data[0])
                        except Exception as e:
                            st.error(f"Error processing image: {str(e)}")
                    
                    if all_images:
                        # Use the model to answer follow-up questions
                        follow_up_prompt = f"{input_prompt}\nBased on all invoices, please answer: {follow_up_query}"
                        response = model.generate_content([follow_up_prompt] + all_images)
                        st.markdown("### Answer")
                        st.markdown(response.text)
    else:
        st.info("Upload and process invoices to see analysis and insights")

# Export & Reports Tab UI
def export_reports_tab():
    st.markdown('<div class="sub-header">Export Options</div>', unsafe_allow_html=True)
    
    if st.session_state.metrics and st.session_state.invoice_data:
        # Create export options
        export_option = st.radio(
            "Select export format:",
            ["CSV Summary", "PDF Report", "JSON Data"]
        )
        
        if export_option == "CSV Summary":
            # Create a dataframe for CSV export
            if st.session_state.metrics["invoice_numbers"] and st.session_state.metrics["dates"]:
                data = []
                for i in range(min(len(st.session_state.metrics["invoice_numbers"]), len(st.session_state.metrics["dates"]))):
                    data.append({
                        "Invoice Number": st.session_state.metrics["invoice_numbers"][i],
                        "Date": st.session_state.metrics["dates"][i],
                        "Vendor": st.session_state.metrics["vendors"][i % len(st.session_state.metrics["vendors"])] if i < len(st.session_state.metrics["vendors"]) else "Unknown"
                    })
                
                if data:
                    df = pd.DataFrame(data)
                    csv = convert_df_to_csv(df)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="invoice_summary.csv",
                        mime="text/csv",
                    )
        
        elif export_option == "PDF Report":
            if st.button("Generate PDF Report"):
                with st.spinner("Generating PDF..."):
                    pdf_data = generate_invoice_report(st.session_state.metrics, st.session_state.invoice_data)
                    
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_data,
                        file_name="invoice_report.pdf",
                        mime="application/pdf",
                    )
        
        elif export_option == "JSON Data":
            import json
            
            # Prepare JSON data
            json_data = {
                "summary": st.session_state.metrics,
                "invoice_data": st.session_state.invoice_data
            }
            
            # Convert to JSON string
            json_str = json.dumps(json_data, indent=2)
            
            st.download_button(
                label="Download JSON Data",
                data=json_str,
                file_name="invoice_data.json",
                mime="application/json",
            )
    else:
        st.info("Upload and process invoices to enable export options")

# Define input prompt template - moved outside of functions to be accessible
input_prompt = """
You are an AI specialized in understanding, analyzing, and extracting data from invoices. 
Your task is to accurately interpret the details provided in invoice images, including:
- Vendor information (name, address, contact)
- Invoice number
- Invoice date
- Line items with descriptions
- Prices, quantities, and subtotals
- Total amount
- Tax information
- Payment terms and methods

Provide structured, accurate data extraction and answer questions about the invoices precisely.
If handling multiple invoices, analyze relationships between them and provide aggregate insights.
"""

# Main function - now only organizing the app flow
def main():
    # Initialize session state for storing invoice data
    if 'processed_invoices' not in st.session_state:
        st.session_state.processed_invoices = []
    if 'invoice_images' not in st.session_state:
        st.session_state.invoice_images = []
    if 'invoice_data' not in st.session_state:
        st.session_state.invoice_data = []
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None
    
    # Apply custom CSS
    apply_custom_css()
    
    # Main content area
    st.markdown('<div class="main-header">📊 InvoiceMate - AI Invoice Analyzer</div>', unsafe_allow_html=True)
    
    # Create sidebar
    create_sidebar()
    
    # Create tabs
    tabs = st.tabs(["Upload & Process", "Analysis & Insights", "Export & Reports"])
    
    # Upload & Process Tab
    with tabs[0]:
        upload_process_tab()
    
    # Analysis & Insights Tab
    with tabs[1]:
        analysis_insights_tab()
    
    # Export & Reports Tab
    with tabs[2]:
        export_reports_tab()

# Run main app
if __name__ == "__main__":
    main()