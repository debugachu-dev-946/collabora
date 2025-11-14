from flask import Flask, request, render_template
from PIL import Image
from process_image import process_image_gemini_2_A, process_image_gemini_2_B, process_image_gemini_2_C, process_image_gemini_2_D, process_image_gemini_2_E
import google.generativeai as genai
import base64
import pandas as pd
import numpy as np
import os


app = Flask(__name__)

def process_image(selected_model, image, file_name):
    if selected_model == 'Form_E':
        return process_image_gemini_2_E(image, file_name)
    elif selected_model == 'Form_D':
        return process_image_gemini_2_D(image, file_name)
    elif selected_model == 'Form_C':
        return process_image_gemini_2_C(image, file_name)
    elif selected_model == 'Form_B':
        return process_image_gemini_2_B(image, file_name)
    elif selected_model == 'Form_A':
        return process_image_gemini_2_A(image, file_name)

# def clean_data(raw_data):
#     """
#     Cleans the raw data by removing unwanted prefixes, newlines, and extra spaces.

#     Args:
#         raw_data: A string containing raw data.

#     Returns:
#         A cleaned and formatted string.
#     """
#     # Remove excessive newline characters and strip leading/trailing whitespace
#     clean_text = raw_data.replace("\n", "").strip()
#     return clean_text

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        uploaded_file = request.files["file"]
        selected_model = request.form.get("form_selection")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(file_path)
        #print(selected_model)
        if not selected_model:
            return render_template("index.html", error="Please select a form.")

        if uploaded_file:
            file_name = uploaded_file.filename
            image = Image.open(uploaded_file)

            if selected_model == "Form_A":
                df1, df2, df3, df4, df5 = process_image(selected_model, image, file_name)
                #print(df1, df2, df3, df4, df5)

                # Convert DataFrames to HTML
                df_html = {
                    "OUTREACH CLINIC DATA": df1.to_html(classes="table table-striped", index=False),
                    "OUTREACH CLINIC INFORMATION": df2.to_html(classes="table table-striped", index=False),
                    "Total Patients": df3.to_html(classes="table table-striped", index=False),
                    "Money Collected": df4.to_html(classes="table table-striped", index=False),
                    "Money Distributed": df5.to_html(classes="table table-striped", index=False),
                }
                print(df_html)

                # Render the results page
                return render_template("results.html", dataframes=df_html, form=selected_model, file_url=file_path)
            
            if selected_model == "Form_B":
                df1, df2, df3, df4, df5, df6, df7, df8, df9 = process_image(selected_model, image, file_name)
                #print(df1, df2, df3, df4, df5, df6)

                # Convert DataFrames to HTML
                df_html = {
                    "OUTREACH CLINIC DATA": df1.to_html(classes="table table-striped", index=False),
                    "OUTREACH CLINIC INFORMATION": df2.to_html(classes="table table-striped", index=False),
                    "Service Offered": df3.to_html(classes="table table-striped", index=False),
                    "What Services were Impacted or Unavailable due to stockouts?": df4.to_html(classes="table table-striped", index=False),
                    "ANTI-RETROVIRAL TREATMENT (ART)": None,
                    "1. TB": df5.to_html(classes="table table-striped", index=False),
                    "2. Number of Patients given each ART duration": df6.to_html(classes="table table-striped", index=False),
                    "3. Patients Receiving ART": df7.to_html(classes="table table-striped", index=False),
                    "4. Patients Newly Linked to Core": df8.to_html(classes="table table-striped", index=False),
                    "5. Viral Load Blood Samples Taken": df9.to_html(classes="table table-striped", index=False),
                }
                # print(df_html)

                # Render the results page
                return render_template("results.html", dataframes=df_html, form=selected_model, file_url=file_path)
            
            if selected_model == "Form_C":
                df1, df2, df3, df4= process_image(selected_model, image, file_name)
                #print(df1, df2, df3, df4, df5, df6)

                # Convert DataFrames to HTML
                df_html = {
                    "OUTREACH CLINIC DATA": df1.to_html(classes="table table-striped", index=False),
                    "Other Services": df2.to_html(classes="table table-striped", index=False),
                    "__": df3.to_html(classes="table table-striped", index=False),
                    "___": df4.to_html(classes="table table-striped", index=False)}
                # print(df_html)

                # Render the results page
                return render_template("results.html", dataframes=df_html, form=selected_model, file_url=file_path)
            
            if selected_model == "Form_D":
                df1, df2, df3, df4 = process_image(selected_model, image, file_name)
                #print(df1, df2, df3, df4, df5, df6)

                # Convert DataFrames to HTML
                df_html = {
                    "OUTREACH CLINIC DATA": df1.to_html(classes="table table-striped", index=False),
                    "HIV COUNSELLING AND TESTING (HCT)": df2.to_html(classes="table table-striped", index=False),
                    "New Positives ": df3.to_html(classes="table table-striped", index=False),
                    "Patients Tested for First Time": df4.to_html(classes="table table-striped", index=False)
                }
                # print(df_html)

                # Render the results page
                return render_template("results.html", dataframes=df_html, form=selected_model, file_url=file_path)
            
            if selected_model == "Form_E":
                df1, df2, df3, df4 = process_image(selected_model, image, file_name)
                #print(df1, df2, df3, df4, df5, df6)

                # Convert DataFrames to HTML
                df_html = {
                    "OUTREACH CLINIC DATA": df1.to_html(classes="table table-striped", index=False),
                    "Family Planning, Maternal & Child Health": df2.to_html(classes="table table-striped", index=False),
                    "Perinatal Health": df3.to_html(classes="table table-striped", index=False),
                    "COVID-19 Vaccinations": df4.to_html(classes="table table-striped", index=False)
                }
                # print(df_html)

                # Render the results page
                return render_template("results.html", dataframes=df_html, form=selected_model, file_url=file_path)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)