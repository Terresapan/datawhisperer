import gspread
import json
from oauth2client.service_account import ServiceAccountCredentials
import streamlit as st
import hmac
import re

# Function to save feedback to a file
def save_feedback(feedback_text):
    # Load the credentials from the secrets
    credentials_data = st.secrets["gcp"]["service_account_json"]
    # print(credentials_data)
    creds = json.loads(credentials_data, strict=False)

    # Set up the Google Sheets API credentials
    scope = ["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/spreadsheets"]
    credentials = ServiceAccountCredentials.from_json_keyfile_dict(creds, scope)
    client = gspread.authorize(credentials)

    # Open the Google Sheet
    sheet_id = '1qnFzZZ7YI-9pXj3iAXafjRmC_EIQyK9gA98AjMv29DM'
    sheet = client.open_by_key(sheet_id).worksheet("datawhisperer")
    sheet.append_row([feedback_text])
       
    
# Password checking function
def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


