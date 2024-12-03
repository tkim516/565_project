import streamlit as st
import pandas as pd

st.set_page_config(page_title="Upload Data", 
                   page_icon="📚",
                   layout="wide")


st.markdown("<h1 style='text-align: center; color: #57cfff;'>Upload your shopper data</h1>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center;'>Unlock insights about shopper buying behaviors</h3>", unsafe_allow_html=True)

# Sidebar information
with st.sidebar:
    st.header("📂 Upload Page Instructions")
    st.write("""
    On this page, you can:
    1. Upload your shopper **CSV file** exported from your data source.
    2. Ensure the file contains valid data with columns like:
       - `Month`
       - `OperatingSystems`
       - `Browser`
       - `Region`
       - `TrafficType`
       - `VisitorType`
       - `Weekend`
       - `Administrative`
       - `Administrative_Duration`
       - `Informational`
       - `Informational_Duration`
       - `ProductRelated`
       - `ProductRelated_Duration`
       - `BounceRates`
       - `ExitRates`
       - `PageValues`
       - `SpecialDay`
       - `Revenue`
    3. Once uploaded, your data will be used across different analysis pages.
    """)
    st.info("📋 Tip: Ensure your CSV file has the same column names and data types as mentioned above.")

# Add an expander for upload instructions
with st.expander("**How to prepare your data?** 📤"):
    st.write("""
    1. Export your data from your data source.
    2. Ensure your CSV file contains the columns mentioned in the sidebar.
    3. Upload the file here to proceed.
    """)

# File uploader
uploaded_file = st.file_uploader(
    "",
    type=["csv"],
    help="Ensure the file is in CSV format with the required columns."
)

# Check if a file is uploaded
if uploaded_file is not None:
    # Read and store the uploaded data
    df = pd.read_csv(uploaded_file)
    st.session_state["user_data"] = df  # Save data to session state

    # Display success message and preview
    st.success("File uploaded successfully!")
    st.write("Here's a preview of your data:")
    st.dataframe(df.head(), use_container_width=True)

    # Calculate statistics
    total_rows = df.shape[0]
    total_revenue = df["Revenue"].sum()
    avg_page_values = df["PageValues"].mean().round(2)
    most_used_browser = df["Browser"].mode()[0]

    # Display quick summary
    st.markdown("### Quick Stats:")
    st.write(f"📊 **Total Rows:** {total_rows}")
    st.write(f"💰 **Total Revenue Generated:** {total_revenue}")
    st.write(f"📖 **Average Page Values:** {avg_page_values}")
    st.write(f"🌐 **Most Used Browser:** {most_used_browser}")

# Check if data already exists in session_state
elif "user_data" in st.session_state:
    st.info("Using previously uploaded data.")
    df = st.session_state["user_data"]

    # Display existing data preview
    st.write("Here's a preview of your existing data:")
    st.dataframe(df.head(), use_container_width=True)

    # Recalculate and display stats for existing data
    total_rows = df.shape[0]
    total_revenue = df["Revenue"].sum()
    avg_page_values = df["PageValues"].mean().round(2)
    most_used_browser = df["Browser"].mode()[0]

    st.markdown("### Quick Stats:")
    st.write(f"📊 **Total Rows:** {total_rows}")
    st.write(f"💰 **Total Revenue Generated:** {total_revenue}")
    st.write(f"📖 **Average Page Values:** {avg_page_values}")
    st.write(f"🌐 **Most Used Browser:** {most_used_browser}")

