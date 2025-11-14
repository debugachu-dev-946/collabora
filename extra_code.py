def enhance_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply adaptive thresholding to get a binary image
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 11, 3)

    # Morphological operations to enhance the text
    kernel = np.ones((1, 1), np.uint8)
    morph = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)

    # Optionally, you can apply dilation to make the text thicker
    dilated = cv2.dilate(morph, kernel, iterations=1)

    return dilated

def process_image_gemini_2_E(image,file_name):
    
    # Load an image from a local directory
    image_path = file_name



    enhanced_image = enhance_image(image_path)
    image_path=f"{file_name}_enhanced_image.jpg"
    cv2.imwrite(image_path, enhanced_image)
        # Read the image file
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        
    
    print(type(enhanced_image),'----------',type(image_data))
    model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")
    prompt = """
    You are a highly intelligent and smart assistant.
    Your task is extract the the following json values from the given image.
    Extract the correct values for:
    {
        "Outreach Clinical Data Form E":{
        "Person Filling Form": "Value",
        "Health Centre": "Value",
        "Position": "Value",
        "Village visited": "Value",
        "Date": "Value"
        }
        "Family Planning, Maternal & Child Health": {
            "Total Served": Value,
            "Male Condom (# NO. of patients)": "Value",
            "Male Condom (# NO. of dispensed)": "Value",
            "Oral Contrac"(# NO. of patients): "Value",
            "Oral Contrac"(# NO. of cycles): "Value",
            "Depo-Provera": "Value",
            "Sayana Press": "Value",
            "Other Injectable": "Value",
            "Implanon": "Value",
            "Jadelle": "Value",
            "Other Implant": "Value",
            "IUD": "Value",
            "Emergency Contraception": "Value",
            "Female Condom": "Value",
            "FP referrals": "Value"
            "FP counselling only": "Value",
        },
        "Perinatal Health": {
            "Total Srved": "Value"
            "Antenatal Care (ANC)": "Value",
            "Postnatal care (PNC)": "Value",
            "Immunisation referrals": "Value"
        },
        "COVID-19 Vaccinations": {
            "Male": "Value",
            "Female": "Value"
        }
    }
    Only return the given json which is valid to dump in json.loads
    
    """
    response = model.generate_content(
        [
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(image_data).decode("utf-8"),
            },
            prompt,
        ]
    )
    
    # Display the response
    # Markdown(">" + response.text)
    a=response.text
    a=a.replace("`","").replace("json","")
    print(a)
    json_data = json.loads(a)
    # json_data = replace_values(json_data)

    # Extract DataFrames
    df1 = pd.DataFrame(json_data["Outreach Clinical Data Form E"], index=[0])
    df2 = pd.DataFrame(json_data["Family Planning, Maternal & Child Health"], index=[0])
    df3 = pd.DataFrame(json_data["Perinatal Health"], index=[0])
    df4 = pd.DataFrame(json_data["COVID-19 Vaccinations"], index=[0])
    
    df1.replace(to_replace=[None, "Null","null"], value=0, inplace=True)
    df1.fillna(0, inplace=True)
    df2.replace(to_replace=[None, "Null","null"], value=0, inplace=True)
    df2.fillna(0, inplace=True)
    df3.replace(to_replace=[None, "Null","null"], value=0, inplace=True)
    df3.fillna(0, inplace=True)
    df4.replace(to_replace=[None, "Null","null"], value=0, inplace=True)
    df4.fillna(0, inplace=True)
    df2.head()

    return df1, df2, df3, df4


def process_image_gemini_2_D(image,file_name):
    image_path = file_name
    enhanced_image = enhance_image(image_path)
    image_path=f"{file_name}_enhanced_image.jpg"
    cv2.imwrite(image_path, enhanced_image)
        # Read the image file
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        
    print(type(enhanced_image),'----------',type(image_data))
    model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")
    prompt = """
    You are a highly intelligent and smart assistant.
    Your task is extract the the following json values from the given image.
    Extract the correct values for:
    {
        "OUTREACH CLINIC DATA FROM D": {
            "Person filling form": "value",
            "Health Centre": "value",
            "Position": "value",
            "Village visited": "value",
            "Date": "value"
        },
        "HIV Counselling and Testing (HCT)": {
            "Total No. of Clients served": {
                "Male": "value",
                "Female": "value"
            },
            "Counselled": {
                "Pre-test": "value",
                "Post-test": "value"
            },
            "HIV Test Results": {
                "Total Positive": "value",
                "Total Negative": "value"
            },
            "Conselled & Tested as couple": "value",
            "Received results as couple": "value",
            "Discordant Results": "value",
            "Returned for viral load results": "value",
            "TB Suspect": "value"
        },
        "New Positives": {
            "Age" : [
            "18 mos-4 years",
            "5-9 years",
            "10 - 14 years",
            "15 - 18 years",
            "19 - 49 years",
            "50 and up",
            "Total"
            ],
            "Male": {
                "18 mos-4 years": "value",
                "5-9 years": "value",
                "10 - 14 years": "value",
                "15 - 18 years": "value",
                "19 - 49 years": "value",
                "50 and up": "value",
                "Total": "value"
            },
            "Female": {
                "18 mos-4 years": "value",
                "5-9 years": "value",
                "10 - 14 years": "value",
                "15 - 18 years": "value",
                "19 - 49 years": "value",
                "50 and up": "value",
                "Total": "value"
            }
        },
        "Patients Tested for First Time": {
            "Age" : [
            "18 mos-4 years",
            "5-9 years",
            "10 - 14 years",
            "15 - 18 years",
            "19 - 49 years",
            "50 and up",
            "Total"
            ],
            "Male": {
                "18 mos-4 years": "value",
                "5-9 years": "value",
                "10 - 14 years": "value",
                "15 - 18 years": "value",
                "19 - 49 years": "value",
                "50 and up": "value",
                "Total": "value"
            },
            "Female": {
                "18 mos-4 years": "value",
                "5-9 years": "value",
                "10 - 14 years": "value",
                "15 - 18 years": "value",
                "19 - 49 years": "value",
                "50 and up": "value",
                "Total": "value"
            }
        }
    }
    
    
    Only return the given json which is valid and ready to dump in json.loads.
    """
    response = model.generate_content(
        [
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(image_data).decode("utf-8"),
            },
            prompt,
        ]
    )
    
    # Display the response
    # Markdown(">" + response.text)
    a=response.text
    a=a.replace("`","").replace("json","")
    data = json.loads(a)  # Replace 'your_json_string' with your actual JSON string
    
    # Extract OUTREACH CLINIC DATA FROM D
    df1 = pd.DataFrame(data["OUTREACH CLINIC DATA FROM D"].items(), columns=['Field', 'Value'])
    
    # Extract HIV Counselling and Testing (HCT)
    hct_data = data["HIV Counselling and Testing (HCT)"]
    df2 = pd.DataFrame({
        'Field': hct_data.keys(),
        'Value': [hct_data[key] for key in hct_data]
    })
    
    # Extract New Positives
    new_positives_data = data["New Positives"]
    df3 = pd.DataFrame({
        'Age Group': new_positives_data['Age'],
        'Male': [new_positives_data['Male'][age] for age in new_positives_data['Age']],
        'Female': [new_positives_data['Female'][age] for age in new_positives_data['Age']]
    })
    
    # Extract Patients Tested for First Time
    patients_tested_data = data["Patients Tested for First Time"]
    df4 = pd.DataFrame({
        'Age Group': patients_tested_data['Age'],
        'Male': [patients_tested_data['Male'][age] for age in patients_tested_data['Age']],
        'Female': [patients_tested_data['Female'][age] for age in patients_tested_data['Age']]
    })
    df1=df1.fillna(0)
    df2 = df2.fillna(0)
    df3 = df3.fillna(0)
    df4 = df4.fillna(0)

    df1 = df1.replace("null", 0)
    df2 = df2.replace("null", 0)
    df3 = df3.replace("null", 0)
    df4 = df4.replace("null", 0)
    return df1, df2, df3, df4


def process_image_gemini_2_C(image,file_name):
    image_path = file_name
    enhanced_image = enhance_image(image_path)
    image_path=f"{file_name}_enhanced_image.jpg"
    cv2.imwrite(image_path, enhanced_image)
        # Read the image file
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        
    print(type(enhanced_image),'----------',type(image_data))
    model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")
    prompt = """
    You are a highly intelligent and smart assistant.
    Your task is extract the the following json values from the given image.
    Extract the correct values for:
    {
        "OUTREACH CLINIC DATA FROM C": {
            "Person filling form": "value",
            "Health Centre": "value",
            "Position": "value",
            "Village visited": "value",
            "Date": "value",
            "Additional treatments or test given and number of patients for each": "value",
            "Additional comments/concerns (continue on back if necessary)": "value"
        },
        "Other Services": {
            "Total Pregnant Women Served at Outreach for any reason":{
                "Syphilis": {
                    "Total Tested": "value",
                    "Tested Positive": "value",
                    "Given Treatment": "value"
                },
                "Hypertension": {
                    "Total Screened": "value",
                    "Diagnosed": "value",
                    "Given Medication": "value"
                },
                "Malaria": {
                    "Suspected Fever": "value",
                    "Tested Positive": "value",
                    "Given Medication": "value"
                }
            },
            "Diabetes Referrals": "value",
            "Child Check-ups": "value",
            "Patients Given Pain Relievers": "value",
            "Immunizations Given": "value",
            "Vitamins Given": "value",
            "__" : {
                "Diagnosed": {
                    "Intestinal Worms": "value",
                    "Fungal (Non-Candidiasis)": "value",
                    "Candidiasis": "value",
                    "Ulcers": "value",
                    "Diarrhea": "value",
                    "Gastro-Intestinal Disorders (Non-Diarrhea)": "value",
                    "Allergy": "value",
                    "Chronic Respiratory Disease": "value"
                },
                "Given Treatment": {
                    "Intestinal Worms": "value",
                    "Fungal (Non-Candidiasis)": "value",
                    "Candidiasis": "value",
                    "Ulcers": "value",
                    "Diarrhea": "value",
                    "Gastro-Intestinal Disorders (Non-Diarrhea)": "value",
                    "Allergy": "value",
                    "Chronic Respiratory Disease": "value"
                }
            },
            "___" : {
                "Diagnosed": {
                    "Measles": "value",
                    "RTI": "value",
                    "Malnutrition": "value",
                    "Burns": "value",
                    "Injuries": "value",
                    "UTI": "value",
                    "Gonorrhoea": "value",
                    "Other STIs": "value",
                    "Eye Infection": "value"
                },
                "Given Treatment": {
                    "Measles": "value",
                    "RTI": "value",
                    "Malnutrition": "value",
                    "Burns": "value",
                    "Injuries": "value",
                    "UTI": "value",
                    "Gonorrhoea": "value",
                    "Other STIs": "value",
                    "Eye Infection": "value"
                }
            }
          
        }
    }
    Only return the given json which is valid and ready to dump in json.loads.
    """
    response = model.generate_content(
        [
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(image_data).decode("utf-8"),
            },
            prompt,
        ]
    )
    a=response.text
    a=a.replace("`","").replace("json","")
    data = json.loads(a)  # Replace 'your_json_string' with your actual JSON string

    # Extract OUTREACH CLINIC DATA FROM C
    df1 = pd.DataFrame(data["OUTREACH CLINIC DATA FROM C"].items(), columns=['Field', 'Value'])
    
    # Extract Other Services
    other_services_data = data["Other Services"]
    df2 = pd.DataFrame({
        'Service': other_services_data.keys(),
        'Value': [other_services_data[key] for key in other_services_data]
    })
    
    # Extract the first nested service (e.g., Syphilis)
    syphilis_data = other_services_data["Total Pregnant Women Served at Outreach for any reason"]["Syphilis"]
    df3 = pd.DataFrame(syphilis_data.items(), columns=['Field', 'Value'])
    
    # Extract the second nested service (e.g., Hypertension)
    hypertension_data = other_services_data["Total Pregnant Women Served at Outreach for any reason"]["Hypertension"]
    df4 = pd.DataFrame(hypertension_data.items(), columns=['Field', 'Value'])
    
    # Extract Given Treatment from the first nested service (e.g., __)
    given_treatment_data_1 = other_services_data["__"]["Given Treatment"]
    df5 = pd.DataFrame(given_treatment_data_1.items(), columns=['Condition', 'Given Treatment'])
    
    # Extract Given Treatment from the second nested service (e.g., ___)
    given_treatment_data_2 = other_services_data["___"]["Given Treatment"]
    df6 = pd.DataFrame(given_treatment_data_2.items(), columns=['Condition', 'Given Treatment'])
    df1=df1.fillna(0)
    df2 = df2.fillna(0)
    df3 = df3.fillna(0)
    df4 = df4.fillna(0)
    df5=df5.fillna(0)
    df6=df6.fillna(0)

    df1 = df1.replace("null", 0)
    df2 = df2.replace("null", 0)
    df3 = df3.replace("null", 0)
    df4 = df4.replace("null", 0)
    df5 = df5.replace("null", 0)
    df6 = df6.replace("null", 0)
    return df1, df2, df3, df4, df5,df6