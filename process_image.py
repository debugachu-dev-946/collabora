import google.generativeai as genai
import base64
import pandas as pd
import cv2
import json
import numpy as np
import time

# local function to build try and except for json 

genai.configure(api_key='YOUR_API_KEY')

# Function to replace null values
def replace_values(data):
    if isinstance(data, dict):
        return {key: replace_values(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [replace_values(item) for item in data]
    elif data in [None, "Null", "null", "", pd.NA]:
        return 0
    else:
        return data

# Function to enhance the image
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


# Placeholder functions for image processing
def process_image_gemini_2_A(image, file_name):
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
    "OUTREACH CLINIC DATA FROM A": {
        "Person filling form": "value",
        "Position": "value",
        "LCA name (If Applicable)": "value",
        "Village": "value",
        "District": "value"
    },
    "OUTREACH CLINIC INFORMATION": {
        "Date": "value",
        "Intended Start Time": "value",
        "Actual Start Time": "value",
        "End Time": "value",
        "Was an HAC worker present? (Circle)": "Yes/No",
        "Were any other organisations involved or contributing to this outreach?": "Yes/No",
        "If yes, name of organisation": "value",
        "List names of villages patients came from": ["value1", "value2"]
    },
    "Total Patients": {
        "Ages": [
            "0-28 days",
            "29 days-4 years",
            "5 years-9 years",
            "10 years-18 years",
            "19 years-24 years",
            "25 years-59 years",
            "60 years and up"
        ],
        "Male": {
            "0-28 days": "Nill",
            "29 days-4 years": "value",
            "5 years-9 years": "value",
            "10 years-18 years": "value",
            "19 years-24 years": "value",
            "25 years-59 years": "value",
            "60 years and up": "value"
        },
        "Female": {
            "0-28 days": "Nill",
            "29 days-4 years": "value",
            "5 years-9 years": "value",
            "10 years-18 years": "value",
            "19 years-24 years": "value",
            "25 years-59 years": "value",
            "60 years and up": "value"
        },
        "Total": {
            "0-28 days": "Nill",
            "29 days-4 years": "value",
            "5 years-9 years": "value",
            "10 years-18 years": "value",
            "19 years-24 years": "value",
            "25 years-59 years": "value",
            "60 years and up": "value"
        }
    },
    "Money Collected": {
        " ": [
            "Adults",
            "Children",
            "Exceptions",
            "Reduced Payments",
            "Fees waived",
            "Total"
        ],
        "Number of People": {
            "Adults": "value",
            "Children": "value",
            "Exceptions": "value",
            "Reduced Payments": "value",
            "Fees waived": "value",
            "Total": "value"
        },
        "UGX per person": {
            "Adults": "value",
            "Children": "value",
            "Exceptions": "value",
            "Reduced Payments": "value",
            "Fees waived": "value",
            "Total": "value"
        },
        "Total Collected": {
            "Adults": "value",
            "Children": "value",
            "Exceptions": "value",
            "Reduced Payments": "value",
            "Fees waived": "value",
            "Total": "value"
        }
    },
    "Money Distributed": {
        " ": [
            "Total given to HCWs",
            "Total given to VHTs/LCA Member",
            "Transporters",
            "Other Costs",
            "Total"
        ],
        "Amount Distributed": {
            "Total given to HCWs": "value",
            "Total given to VHTs/LCA Member": "value",
            "Transporters": "value",
            "Other Costs": "value",
            "Total": "value"
        }
    }
    }
    Only return the given json which is valid and ready to dump in json.loads.
    """
    SUCCESS = False
    error = None  # Initialize error variable

    while not SUCCESS:
        try:
            # Send request to the model
            request_payload = [
                {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(image_data).decode("utf-8"),
                },
                prompt,
            ]

            if error:  # Include error in the request if it exists
                request_payload.append(error)

            response = model.generate_content(request_payload)

            # Process the response
            a = response.text
            a = a.replace("`", "").replace("json", "")

            if a:  # Ensure 'a' is not empty or None
                data = json.loads(a)
                SUCCESS = True  # Mark success only after valid JSON is parsed
            else:
                raise ValueError("Response is empty or not in JSON format.")

        except Exception as e:
            error = str(e)  # Capture the exception as the error
            print(f"Error occurred: {error}")  # Optional: Log the error



    # Create DataFrames for each section
    df1 = pd.DataFrame([data["OUTREACH CLINIC DATA FROM A"]])
    df2 = pd.DataFrame([data["OUTREACH CLINIC INFORMATION"]])

    # Convert columns into rows using melt
    df1 = df1.melt(var_name="Attribute", value_name="Value")
    df2 = df2.melt(var_name="Atribute", value_name="Value")

    # Handle "Total Patients"
    total_patients_data = data["Total Patients"]
    df3 = pd.DataFrame({
        "Age Group": total_patients_data["Ages"],
        "Male": list(total_patients_data["Male"].values()),
        "Female": list(total_patients_data["Female"].values()),
        "Total": list(total_patients_data["Total"].values())
    })

    # Handle "Money Collected"
    money_collected_data = data["Money Collected"]
    df4 = pd.DataFrame({
        "Category": money_collected_data[" "],
        "Number of People": list(money_collected_data["Number of People"].values()),
        "UGX per person": list(money_collected_data["UGX per person"].values()),
        "Total Collected": list(money_collected_data["Total Collected"].values())
    })

    # Handle "Money Distributed"
    money_distributed_data = data["Money Distributed"]
    df5 = pd.DataFrame({
        "Category": money_distributed_data[" "],
        "Amount Distributed": list(money_distributed_data["Amount Distributed"].values())
    })

    df1=df1.fillna(0)
    df2 = df2.fillna(0)
    df3 = df3.fillna(0)
    df4 = df4.fillna(0)
    df5=df5.fillna(0)
    # df6=df6.fillna(0)

    df1 = df1.replace("null", 0)
    df2 = df2.replace("null", 0)
    df3 = df3.replace("null", 0)
    df4 = df4.replace("null", 0)
    df5 = df5.replace("null", 0)
    # df6 = df6.replace("null", 0)
    return df1, df2, df3, df4, df5

def process_image_gemini_2_B(image, file_name):
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
    "OUTREACH CLINIC DATA FROM B": {
        "Person filling form": "value",
        "Position": "value",
        "Health Centre": "value",
        "Village": "value",
        "Date": "value"
    },
    "OUTREACH CLINIC INFORMATION": {
        "Names of health workers who attended": "value",
        "Number of Health Workers": {
            "Total Health Workers": "value",
            "Enrolled Midwives": "value",
            "Registered Midwives": "value",
            "Case Officer": "value",
            "Counsellors": "value",
            "Lab Technician": "value",
            "Nursing Assistant": "value",
            "Enrolled Nurse": "value",
            "Registered Nurse": "value",
            "Other (Specify)": "value"
        },
        "Services offered": [
            "General Treatment",
            "ART Refills",
            "Family Planning",
            "ANC/PNC",
            "HIV Counselling/Testing",
            "Immunization"
        ],
        "Were there stockouts? (circle one)": "Y/N",
        "What services were IMPACTED or UNAVAILABLE due to stockouts?": {
            "General Treatment": [
                "Malaria Treatment",
                "Amoxicillin",
                "Other"
            ],
            "HIV/ART": [
                "ART",
                "Testing",
                "Other"
            ],
            "Family Planning": [
                "Pregnancy Tests",
                "Other"
            ],
            "ANC/PNC": [
                "value"
            ]
        }
    },
    "ANTI-Retroviral Treatment (ART)": {
        "1. TB": {
            "Diagnosed": "value",
            "Treated": "value"
        },
        "2. Number of Patients given each ART duration": {
            "1 Month": "value",
            "2 Month": "value",
            "3 Month": "value",
            "___months": "value",
            "Total number of people": "value",
            "Total no. months given": "value"
        },
        "3. Patients Receiving ART": {
            "Age": [
                "0-1 year",
                "2-4 years",
                "5-14 years",
                "15 and up",
                "Total"
            ],
            "Male": {
                "0-1 year": "value",
                "2-4 years": "value",
                "5-14 years": "value",
                "15 and up": "value",
                "Total": "value"
            },
            "Female": {
                "0-1 year": "value",
                "2-4 years": "value",
                "5-14 years": "value",
                "15 and up": "value",
                "Total": "value"
            }
        },
        "3. Patients Newly Linked to Core": {
            "Age": [
                "0-1 year",
                "2-4 years",
                "5-14 years",
                "15 and up",
                "Total"
            ],
            "Male": {
                "0-1 year": "value",
                "2-4 years": "value",
                "5-14 years": "value",
                "15 and up": "value",
                "Total": "value"
            },
            "Female": {
                "0-1 year": "value",
                "2-4 years": "value",
                "5-14 years": "value",
                "15 and up": "value",
                "Total": "value"
            }
        },
        "5. Viral Load Blood Samples Token": {
            " ": [
                "Total"
            ],
            "Male": {
                "Total": "value"
            },
            "Female": {
                "Total": "value"
            }
        }
    }
    }
    Only return the given json which is valid and ready to dump in json.loads.
    """
    SUCCESS = False
    error = None  # Initialize error variable

    while not SUCCESS:
        try:
            # Send request to the model
            request_payload = [
                {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(image_data).decode("utf-8"),
                },
                prompt,
            ]

            if error:  # Include error in the request if it exists
                request_payload.append(error)

            response = model.generate_content(request_payload)

            # Process the response
            a = response.text
            a = a.replace("`", "").replace("json", "")

            if a:  # Ensure 'a' is not empty or None
                json_data = json.loads(a)
                SUCCESS = True  # Mark success only after valid JSON is parsed
            else:
                raise ValueError("Response is empty or not in JSON format.")

        except Exception as e:
            error = str(e)  # Capture the exception as the error
            print(f"Error occurred: {error}")  # Optional: Log the error

    # Flatten the 'OUTREACH CLINIC DATA FROM B' section
    df1 = pd.DataFrame([json_data["OUTREACH CLINIC DATA FROM B"]])

    # Flatten the 'Number of Health Workers' section
    df2 = pd.DataFrame([json_data["OUTREACH CLINIC INFORMATION"]["Number of Health Workers"]])

    # Convert columns into rows using melt
    df1 = df1.melt(var_name="Attribute", value_name="Value")
    df2 = df2.melt(var_name="Health Worker Role", value_name="Count")

    # Extract 'Services offered'
    df3 = pd.DataFrame({"Services Offered": json_data["OUTREACH CLINIC INFORMATION"]["Services offered"]})

    # Flatten the 'What services were IMPACTED or UNAVAILABLE due to stockouts?' section
    stockouts_data = json_data["OUTREACH CLINIC INFORMATION"]["What services were IMPACTED or UNAVAILABLE due to stockouts?"]
    stockouts_list = []
    for service, impacts in stockouts_data.items():
        for impact in impacts:
            stockouts_list.append({"Service": service, "Impact": impact})
    df4 = pd.DataFrame(stockouts_list)

    art_data = json_data["ANTI-Retroviral Treatment (ART)"]
    # Create separate DataFrames for each subsection
    tb_df = pd.DataFrame([art_data["1. TB"]])
    patients_duration_df = pd.DataFrame([art_data["2. Number of Patients given each ART duration"]])

    # Convert columns into rows using melt
    df5 = tb_df.melt(var_name="Attribute", value_name="Value")
    df6 = patients_duration_df.melt(var_name="Atribute", value_name="Value")

    # Process "3. Patients Receiving ART"
    receiving_art_data = art_data["3. Patients Receiving ART"]
    df7 = pd.DataFrame({
        "Age Group": receiving_art_data["Age"],
        "Male": list(receiving_art_data["Male"].values()),
        "Female": list(receiving_art_data["Female"].values())
    })

    # Process "3. Patients Newly Linked to Core"
    newly_linked_data = art_data["3. Patients Newly Linked to Core"]
    df8 = pd.DataFrame({
        "Age Group": newly_linked_data["Age"],
        "Male": list(newly_linked_data["Male"].values()),
        "Female": list(newly_linked_data["Female"].values())
    })

    # Process "5. Viral Load Blood Samples Token"
    viral_load_data = art_data["5. Viral Load Blood Samples Token"]
    df9 = pd.DataFrame({
        "Gender": ["Male", "Female"],
        "Total": [viral_load_data["Male"]["Total"], viral_load_data["Female"]["Total"]]
    })


    df1=df1.fillna(0)
    df2 = df2.fillna(0)
    df3 = df3.fillna(0)
    df4 = df4.fillna(0)
    df5=df5.fillna(0)
    df6=df6.fillna(0)
    df7=df7.fillna(0)
    df8=df8.fillna(0)
    df9=df9.fillna(0)
    

    df1 = df1.replace("null", 0)
    df2 = df2.replace("null", 0)
    df3 = df3.replace("null", 0)
    df4 = df4.replace("null", 0)
    df5 = df5.replace("null", 0)
    df6 = df6.replace("null", 0)
    df7 = df7.replace("null", 0)
    df8 = df8.replace("null", 0)
    df9 = df9.replace("null", 0)
    return df1, df2, df3, df4, df5, df6, df7, df8, df9

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
        "Total Pregnant Women Served at Outreach for any reason": "value",
        "Syphilis": {
                "Total Tested": "value",
                "Tested Positive": "value",
                "Given Treatment": "value"},
        "Hypertension": {
                "Total Screened": "value",
                "Diagnosed": "value",
                "Given Medication": "value"},
        "Malaria": {
                "Suspected Fever": "value",
                "Tested Positive": "value",
                "Given Medication": "value"},
        "Diabetes Referrals": "value",
        "Child Check-ups": "value",
        "Patients Given Pain Relievers": "value",
        "Immunizations Given": "value",
        "Vitamins Given": "value"
        },
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
    SUCCESS = False
    error = None  # Initialize error variable

    while not SUCCESS:
        try:
            # Send request to the model
            request_payload = [
                {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(image_data).decode("utf-8"),
                },
                prompt,
            ]

            if error:  # Include error in the request if it exists
                request_payload.append(error)

            response = model.generate_content(request_payload)

            # Process the response
            a = response.text
            a = a.replace("`", "").replace("json", "")

            if a:  # Ensure 'a' is not empty or None
                data = json.loads(a)
                SUCCESS = True  # Mark success only after valid JSON is parsed
            else:
                raise ValueError("Response is empty or not in JSON format.")

        except Exception as e:
            error = str(e)  # Capture the exception as the error
            print(f"Error occurred: {error}")  # Optional: Log the error

    # Extract OUTREACH CLINIC DATA FROM C
    df1 = pd.DataFrame(data["OUTREACH CLINIC DATA FROM C"].items(), columns=['Field', 'Value'])
    
    # Extract Other Services
    other_services_data = data["Other Services"]
    df2 = pd.DataFrame({
        'Service': other_services_data.keys(),
        'Value': [other_services_data[key] for key in other_services_data]
    })
    
    # # Extract the second nested service (e.g., Hypertension)
    # hypertension_data = other_services_data["Total Pregnant Women Served at Outreach for any reason"]["Hypertension"]
    # df4 = pd.DataFrame(hypertension_data.items(), columns=['Field', 'Value'])
    
    # Extract Given Treatment from the first nested service (e.g., __)
    given_treatment_data_1 = data["__"]
    df3 = pd.DataFrame(given_treatment_data_1.items(), columns=['Condition', 'Given Treatment'])
    
    # Extract Given Treatment from the second nested service (e.g., ___)
    given_treatment_data_2 = data["___"]
    df4 = pd.DataFrame(given_treatment_data_2.items(), columns=['Condition', 'Given Treatment'])
    
    df1=df1.fillna(0)
    df2 = df2.fillna(0)
    df3 = df3.fillna(0)
    df4 = df4.fillna(0)
    # df6=df6.fillna(0)

    df1 = df1.replace("null", 0)
    df2 = df2.replace("null", 0)
    df3 = df3.replace("null", 0)
    df4 = df4.replace("null", 0)
    # df6 = df6.replace("null", 0)
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
    SUCCESS = False
    error = None  # Initialize error variable

    while not SUCCESS:
        try:
            # Send request to the model
            request_payload = [
                {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(image_data).decode("utf-8"),
                },
                prompt,
            ]

            if error:  # Include error in the request if it exists
                request_payload.append(error)

            response = model.generate_content(request_payload)

            # Process the response
            a = response.text
            a = a.replace("`", "").replace("json", "")

            if a:  # Ensure 'a' is not empty or None
                data = json.loads(a)
                SUCCESS = True  # Mark success only after valid JSON is parsed
            else:
                raise ValueError("Response is empty or not in JSON format.")

        except Exception as e:
            error = str(e)  # Capture the exception as the error
            print(f"Error occurred: {error}")  # Optional: Log the error
    
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
    SUCCESS = False
    error = None  # Initialize error variable

    while not SUCCESS:
        try:
            # Send request to the model
            request_payload = [
                {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(image_data).decode("utf-8"),
                },
                prompt,
            ]

            if error:  # Include error in the request if it exists
                request_payload.append(error)

            response = model.generate_content(request_payload)

            # Process the response
            a = response.text
            a = a.replace("`", "").replace("json", "")

            if a:  # Ensure 'a' is not empty or None
                json_data = json.loads(a)
                SUCCESS = True  # Mark success only after valid JSON is parsed
            else:
                raise ValueError("Response is empty or not in JSON format.")

        except Exception as e:
            error = str(e)  # Capture the exception as the error
            print(f"Error occurred: {error}")  # Optional: Log the error

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
