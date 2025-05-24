import streamlit as st
import os

# Must come FIRST!
# st.set_page_config("Chat PDF", layout="wide")
st.set_page_config(page_title="📄 RAG Document Uploader", layout="wide")

import os
from ChatWithPDF.pdf_Chat import pdf_chatbot  # ✅ Import only the needed function
from GeminiPro_main.Document_Q_A_app import *
from RAGModel_Base.RAG_Streamlit_driver import *




# ---------- Define App Pages ----------
def main_dashboard():
    st.title("Application Dashboard")
    st.write("Choose an app to launch:")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Run App 1"):
            st.session_state.page = "app1"
        if st.button("Run App 3"):
            st.session_state.page = "app3"
        if st.button("Run App 5"):
            st.session_state.page = "app5"
        if st.button("Run App 7"):
            st.session_state.page = "app7"
        if st.button("Run App 9"):
            st.session_state.page = "app9"

    with col2:
        if st.button("Run App 2"):
            st.session_state.page = "app2"
        if st.button("Run App 4"):
            st.session_state.page = "app4"
        if st.button("Run App 6"):
            st.session_state.page = "app6"
        if st.button("Run App 8"):
            st.session_state.page = "app8"
        if st.button("Run App 10"):
            st.session_state.page = "app10"

    st.markdown("---")
    if st.button("🛑 Stop App"):
        st.success("App stopped. You may now close the tab or refresh to restart.")
        st.stop()

    if st.button("🛑 Force Close App"):
        st.warning("Shutting down the Streamlit app...")
        os._exit(0)

def return_to_dashboard():
    if st.button("🔙 Return to Main Page"):
        st.session_state.page = "main"

def app_template(app_number):
    st.subheader(f"App {app_number}: Demo")
    st.write(f"This is App {app_number}. Add your functionality here.")
    return_to_dashboard()

# ---------- Routing ----------
def app_router():
    if "page" not in st.session_state:
        st.session_state.page = "main"

    page = st.session_state.page

    if page == "main":
        main_dashboard()
    elif page == "app1":
        pdf_chatbot()  # Call the imported function

    elif page == "app2":
        DocumentQAApp().run()

    elif page == "app3":
        RAGUploaderApp().run()
        if "go_home" in st.session_state and st.session_state.go_home:
            st.session_state.page = "main"
            # st.rerun()

        # RAGUploaderApp().run()

    elif page == "app4":
        app_template(4)
    elif page == "app5":
        app_template(5)
    elif page == "app6":
        app_template(6)
    elif page == "app7":
        app_template(7)
    elif page == "app8":
        app_template(8)
    elif page == "app9":
        app_template(9)
    elif page == "app10":
        app_template(10)
    else:
        st.error("Unknown page!")

# ---------- Run the App ----------
if __name__ == "__main__":
    app_router()
