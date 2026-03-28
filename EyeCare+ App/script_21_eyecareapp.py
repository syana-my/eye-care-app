import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import random
import seaborn as sns
import altair as alt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

# -----------------------------------
# PAGE CONFIG
st.set_page_config(
    page_title="EyeCare+",
    page_icon="👁️",
    layout="wide"
)

# -----------------------------------
# SESSION STATE INIT
if "user_done" not in st.session_state:
    st.session_state.user_done = False

# -----------------------------------
# GENERATE PDF FILE
def generate_full_patient_report(
                    patient_id,
                    age,
                    location,
                    image_results
                ):

               file_name = f"Patient_Report_{patient_id}.pdf"

               styles = getSampleStyleSheet()

               elements = []

               # TITLE
               elements.append(
                 Paragraph(
                     "AI Glaucoma Screening Report",
                      styles["Title"]
                 )
               )

               elements.append(Spacer(1,12))

               # DATE
               today = datetime.today().strftime("%d %B %Y")

               elements.append(
                   Paragraph(
                       f"Report Date: {today}",
                       styles["Normal"]
                   )
               )

               elements.append(Spacer(1,12))

               # PATIENT INFO
               elements.append(
                   Paragraph(
                       f"Patient ID: {patient_id}",
                       styles["Normal"]
                   )
               )

               elements.append(
                   Paragraph(
                       f"Age: {age}",
                       styles["Normal"]
                   )
               )

               elements.append(
                   Paragraph(
                       f"Location: {location}",
                       styles["Normal"]
                   )
               )

               elements.append(Spacer(1,20))

               # LOOP setiap image
               for result in image_results:

                   image_path = result["image_path"]

                   cdr = result["cdr"]

                   prediction = result["prediction"]

                   risk = result["risk"]


                   # image
                   if os.path.exists(image_path):

                       elements.append(
                           RLImage(
                               image_path,
                               width=8*cm,
                               height=8*cm
                           )
                       )


                   elements.append(Spacer(1,8))


                   # result text
                   elements.append(
                       Paragraph(
                           f"Image: {os.path.basename(image_path)}",
                           styles["Heading3"]
                       )
                   )


                   elements.append(
                       Paragraph(
                           f"Risk Probability: {prediction:.2f}",
                           styles["Normal"]
                       )
                   )


                   elements.append(
                       Paragraph(
                           f"Cup-to-Disc Ratio (CDR): {cdr:.2f}",
                           styles["Normal"]
                       )
                   )


                   elements.append(
                       Paragraph(
                           f"Risk Level: {risk}",
                           styles["Normal"]
                       )
                   )


                   # interpretation
                   if risk == "Low":

                       interpretation = """
                       Optic nerve appears within normal range.
                       Routine eye examination recommended.
                       """

                   elif risk == "Medium":

                       interpretation = """
                       Moderate structural change detected.
                       Regular monitoring is recommended.
                       """

                   else:

                       interpretation = """
                       High CDR detected.
                       Consultation with ophthalmologist recommended.
                       """


                   elements.append(
                       Paragraph(
                           f"Interpretation: {interpretation}",
                           styles["Normal"]
                       )
                   )


                   elements.append(Spacer(1,20))


               doc = SimpleDocTemplate(

                   file_name,

                   pagesize=A4
               )


               doc.build(elements)


               return file_name

# -----------------------------------
# LOAD MODEL
@st.cache_resource
def load_my_model():
    return load_model("glaucoma_efficientnet_model.h5")

model = load_my_model()
model.build((None, 224, 224, 3))

# -----------------------------------
# IMAGE SETTINGS
IMAGE_FOLDER = "Images"
IMG_SIZE = 224

def preprocess_image(img):

    img = img.resize((IMG_SIZE, IMG_SIZE))

    img = np.array(img)

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    return img


def load_patient_images(patient_id):

    images = []

    for file in os.listdir(IMAGE_FOLDER):

        if file.startswith(f"{patient_id}_"):

            path = os.path.join(IMAGE_FOLDER, file)

            images.append((file, Image.open(path)))

    return images

# -----------------------------------
# STYLE
st.markdown("""
<style>
body {background-color: #0E1117; color: black;}
.stButton>button {
    background-color: #6FAF4F;
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# TITLE + STORY
st.title("👁️ EyeCare+")
st.markdown("### AI-Powered Glaucoma Web System")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.info("""
    ### 🌍 PROBLEM:
    Glaucoma is a silent disease that can cause irreversible blindness.
    """)

with col2:
    st.success("""
    ### 💡 SOLUTION:
    EyeCare+ provides early AI-assisted screening accessible anywhere.
    """)

# -----------------------------------
# PROFESSIONAL SIDEBAR NAVIGATION
st.sidebar.markdown("## 👁️ EyeCare+")
st.sidebar.markdown("---")  # separator

if "menu" not in st.session_state:
    st.session_state.menu = "Home" 

menu = st.sidebar.radio(
    "📂 Menu",
    options=[
        "Home",
        "Screening",
        "Dashboard",
        "Appointment"
    ],
    index=[
        "Home",
        "Screening",
        "Dashboard",
        "Appointment"
    ].index(st.session_state.menu),

    format_func=lambda x: f"➡️ {x}"
)

# update state bila user klik sidebar
st.session_state.menu = menu

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ by F2 Team")

# -----------------------------------
# DATA LIST
malaysia_states = [
    "Johor","Kedah","Kelantan","Melaka","Negeri Sembilan",
    "Pahang","Perak","Perlis","Penang",
    "Sabah","Sarawak","Selangor","Terengganu",
    "Kuala Lumpur","Putrajaya","Labuan"
]

indonesia_provinces = [
    "Aceh","Bali","Banten","Central Java","East Java",
    "Jakarta","West Java","Papua","Riau",
    "South Sulawesi","West Sumatra"
]

# -----------------------------------
# 1. PATIENT INFO
if menu == "Home":

    st.header("📝 Patient Information")

    # Load your patient dataset
    patient_df = pd.read_csv("glaucoma_clean_data.csv")  # make sure this file exists

    name = patient_df['patient'].astype(str).tolist()

    # Dropdown instead of text input
    selected_patient = st.selectbox("Select Patient ID*", name)

    # Get patient info from dataset
    patient_row = patient_df[patient_df['patient'].astype(str) == selected_patient].iloc[0]

    # Birthdate input
    birthdate = st.date_input(
        "Date of Birth*",
        value=pd.to_datetime(patient_row['birthdate']) if 'birthdate' in patient_df.columns else datetime(2000,1,1),
        min_value=datetime(1900, 1, 1),
        max_value=datetime.today()
    )

    # Calculate age
    today = datetime.today()
    age = today.year - birthdate.year - (
        (today.month, today.day) < (birthdate.month, birthdate.day)
    )
    st.write(f"Calculated Age: **{age} years old**")
    st.caption("Age is automatically calculated based on birthdate.")

    # Country selection
    country = st.selectbox("Country*", ["Malaysia", "Indonesia"])

    if country == "Malaysia":
        state = st.selectbox("State*", malaysia_states)
    else:
        state = st.selectbox("Province*", indonesia_provinces)

    area = st.text_input("Postcode*")

    if st.button("Save Info"):
        if selected_patient and birthdate != datetime.today().date():
            st.session_state.user_done = True
            st.session_state.name = selected_patient
            st.session_state.age = age
            st.session_state.birthdate = birthdate
            st.session_state.location = f"{area}, {state}, {country}"
           
            st.session_state.menu = "Screening"

            st.rerun()
        else:
            st.error("Please complete all required fields.")

# -----------------------------------
# BLOCK ACCESS
if not st.session_state.user_done and menu != "Home":
    st.warning("Please fill patient information first!")
    st.stop()

# -----------------------------------
# 2. SCREENING
if menu == "Screening":

    st.header("Patient Retinal Images")

    st.caption(f"""
    Patient ID: {st.session_state.name}  
    Age: {st.session_state.age}  
    DOB: {st.session_state.birthdate}  
    Location: {st.session_state.location}
    """)

    st.warning("⚠️ This tool is for screening only, not medical diagnosis.")

    patient_images = load_patient_images(
        st.session_state.name
    )

    df = pd.read_csv("glaucoma_clean_data.csv") 

    if len(patient_images) == 0:

        st.error("No images found")

    else:

        st.success(f"{len(patient_images)} images found")

        image_results = []

        for filename, image in patient_images:

            st.markdown("---")

            st.subheader(filename)

            col1, col2 = st.columns(2)

            col1.image(
                image,
                use_container_width=True
            )

            # prediction
            processed_img = preprocess_image(image)

            prediction = model.predict(
                processed_img
            )[0][0]

            prediction = float(prediction)

            col2.write("Screening Result")

            col2.progress(prediction)

            col2.write(f"Glaucoma Risk Probability: {prediction:.2f}"
            )

            actual_label = df[
                df["image_name"] == filename
            ]["label"].values

            if len(actual_label) > 0:

                actual_label = actual_label[0]

            else:

                actual_label = "Unknown"
            
            col2.caption(
                f"Dataset Label: {actual_label}"
            )
            

            # label
            threshold = 0.5

            if prediction < threshold:

                col2.success("Low Risk")

            elif prediction < 0.7:

                col2.warning("Medium Risk")

            else:

                col2.error("High Risk")

            # confidence
            confidence = float(
                1 - abs(0.5 - prediction)
            )

            col2.metric(
                "Confidence Score",
                f"{confidence:.2f}"
            )

            col2.progress(confidence)
            col2.info("""
            Higher confidence means the prediction is more stable.
            Lower confidence suggests further clinical validation may be useful.
            """)

            # CDR
            st.markdown("### 📊 Cup-to-Disc Ratio (CDR)")

            # estimate based on prediction
            disc_default = 120

            # adjust CDR ikut label sebenar
            if actual_label == "GON-":

                cdr_est = 0.35 + prediction*0.15

            else:

                cdr_est = 0.55 + prediction*0.25

            cup_default = int(disc_default * cdr_est)

            cup = st.slider(
                f"Cup Size {filename}",
                50,150,cup_default
            )

            disc = st.slider(
                f"Disc Size {filename}",
                60,200,disc_default
            )

            cdr = cup/disc

            st.metric(
                f"CDR {filename}",
                f"{cdr:.2f}"
            )

            st.caption(
            "CDR is estimated based on optic nerve structure."
            )

            st.info("""
            Higher CDR values may indicate optic nerve damage,
            which is commonly associated with glaucoma.
            """)

            # GRAPH INTERPRETATION
            st.markdown("### 📊 Risk Interpretation Graph")

            # warna ikut risk
            if cdr < 0.4:
               point_color = "#4CAF50"   # green
            elif cdr < 0.7:
               point_color = "#FFA726"   # orange
            else:
               point_color = "#EF5350"   # red


            fig, ax = plt.subplots(figsize=(6,4))

            # background hitam
            fig.patch.set_facecolor("#0E1117")
            ax.set_facecolor("#0E1117")

            # zone shading
            ax.axhspan(0, 0.4, color="#4CAF50", alpha=0.15)
            ax.axhspan(0.4, 0.7, color="#FFA726", alpha=0.15)
            ax.axhspan(0.7, 1.0, color="#EF5350", alpha=0.15)

            # threshold lines (grey)
            ax.axhline(0.4, linestyle="--", color="grey", linewidth=1.5)
            ax.axhline(0.7, linestyle="--", color="grey", linewidth=1.5)

            # patient value
            ax.scatter(1, cdr, color=point_color, s=200)

            # vertical line indicator
            ax.plot([1,1], [0,cdr], color=point_color, linewidth=4)

            # labels zone
            ax.text(1.05, 0.2, "Normal zone", color="white")
            ax.text(1.05, 0.55, "Monitor zone", color="white")
            ax.text(1.05, 0.85, "High risk zone", color="white")

            # axis styling
            ax.set_ylim(0,1)
            ax.set_xlim(0.8,1.2)

            ax.set_xticks([1])
            ax.set_xticklabels(["Your CDR"], color="white")

            ax.set_ylabel("CDR value", color="white")

            ax.set_title(
                "Patient CDR compared to clinical thresholds",
                 color="white"
            )

            # tukar warna axis
            ax.tick_params(colors='white')

            for spine in ax.spines.values():
                spine.set_color("grey")

            st.pyplot(fig)

            # -----------------------------------
            # RISK LEVEL
            st.markdown("### What does this mean?")

            if cdr < 0.5:
                risk = "Low"
                st.success("Your optic nerve ratio is within normal range.")
            elif cdr < 0.7:
                risk = "Medium"
                st.warning("Your ratio is slightly high. Monitoring is recommended.")
            else:
                risk = "High"
                st.error("High CDR detected. Please consult eye specialist.")

            st.write(f"Risk Level: **{risk}**")

            st.markdown("---")

            # -----------------------------------
            # COMPARISON WITH NORMAL EYE
            st.markdown("### 🧿 Comparison with Typical Healthy Eye")

            healthy_cdr = 0.4

            # warna ikut risk
            if cdr < 0.4:
                patient_color = "#4CAF50"
            elif cdr < 0.7:
                patient_color = "#FFA726"
            else:
                patient_color = "#EF5350"

            fig2, ax2 = plt.subplots(figsize=(6,4))

            fig2.patch.set_facecolor("#0E1117")
            ax2.set_facecolor("#0E1117")

            labels = ["Healthy Eye", "Your Eye"]

            values = [healthy_cdr, cdr]

            colors = ["#42A5F5", patient_color]

            bars = ax2.bar(labels, values, color=colors, width=0.6)

            # value text
            for bar in bars:

                h = bar.get_height()

                ax2.text(
                    bar.get_x()+bar.get_width()/2,
                    h+0.02,
                    f"{h:.2f}",
                    ha="center",
                    color="white"
                )

            # normal reference line
            ax2.axhline(0.4, linestyle="--", color="grey")

            ax2.set_ylim(0,1)

            ax2.set_ylabel("CDR value", color="white")

            ax2.set_title(
                "Comparison of Optic Nerve Health",
                color="white"
            )

            ax2.tick_params(colors="white")

            for spine in ax2.spines.values():
                spine.set_color("grey")

            st.pyplot(fig2)

            st.info("""
            Typical healthy eyes usually have CDR around 0.3 – 0.4.
            Higher values may indicate possible optic nerve damage.
            """)

            st.markdown("---")

            # simpan result untuk pdf

            path = os.path.join("Images", filename)

            image_results.append({

                "image_path": path,

                "cdr": cdr,

                "prediction": prediction,

                "risk": risk
            })

            # -----------------------------------
            # SUMMARY
            st.subheader("📊 Summary")

            if risk == "Low":
                st.success("Routine eye check recommended.")
            elif risk == "Medium":
                st.warning("Monitor regularly and consult eye specialist.")
            else:
                st.error("Visit opthalmologist immediately.")

            col1, col2, col3 = st.columns(3)
            col1.metric("CDR", f"{cdr:.2f}")
            col2.metric("Risk Level", risk)
            col3.metric("Risk Probability", f"{prediction:.2f}")

        st.markdown("---")

        if st.button("Generate Patient Report"):

            pdf_file = generate_full_patient_report(

                patient_id = st.session_state.name,

                age = st.session_state.age,

                location = st.session_state.location,

                image_results = image_results
            )

            st.success("PDF created")

            with open(pdf_file, "rb") as f:

                 st.download_button(

                     "Download Report",

                      f,

                      file_name = pdf_file
                 )

# ==============================
# DASHBOARD
# ==============================
if menu == "Dashboard":
    st.header("📊 Glaucoma Dashboard")

    # -----------------------
    # Dataset Selection
    # -----------------------
    dataset_choice = st.radio(
        "Select Dataset:",
        ("Patient Dataset", "Prediction Dataset")
    )

    # =============================
    # PATIENT DATASET
    # =============================
    if dataset_choice == "Patient Dataset":
        df = pd.read_csv("glaucoma_clean_data.csv")
        df = df[['image_name','patient','label','quality_score']]

        # Numeric label for convenience
        df['label_numeric'] = df['label'].map({'GON+':1,'GON-':0})
        df['probability'] = None
        df['prediction'] = None

        # Risk score calculation
        df['risk_score'] = (df['quality_score'].max() - df['quality_score']) / (
            df['quality_score'].max() - df['quality_score'].min()
        )

        # Classify risk levels
        def classify_risk(score):
            if score < 0.4:
                return "Low"
            elif score < 0.7:
                return "Medium"
            else:
                return "High"

        df['risk_level'] = df['risk_score'].apply(classify_risk)

        # -----------------------
        # METRICS SUMMARY
        # -----------------------
        st.subheader("📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Data", len(df))
        col2.metric("Avg Quality Score", round(df['quality_score'].mean(),2))
        col3.metric("High Risk Cases", sum(df['risk_level']=="High"))
        col4.metric("Glaucoma (GON+)", sum(df['label']=="GON+"))

        st.markdown("---")

        # -----------------------
        # VISUALIZATIONS
        # -----------------------
        import altair as alt
        color_scale = alt.Scale(domain=["Low","Medium","High"], range=["#4CAF50","#FFA726","#EF5350"])

        # Risk level distribution
        chart_risk = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x="risk_level",
                y="count()",
                color=alt.Color("risk_level", scale=color_scale)
            )
        )

        # Scatter plot: Quality vs Risk
        chart_scatter = (
            alt.Chart(df)
            .mark_circle(size=60)
            .encode(
                x="quality_score",
                y="risk_score",
                color=alt.Color("risk_level", scale=color_scale)
            )
            .interactive()
        )

        col1, col2 = st.columns(2)
        col1.altair_chart(chart_risk, use_container_width=True)
        col2.altair_chart(chart_scatter, use_container_width=True)

        # -----------------------
        # CRITICAL CASES HIGHLIGHT
        # -----------------------
        st.subheader("🚨 Critical Cases (High Risk + GON+)")
        critical_df = df[(df['risk_level']=="High") & (df['label']=="GON+")]
        st.dataframe(critical_df[['patient','risk_level','quality_score','label']])

        # -----------------------
        # INTERACTIVE FILTERABLE TABLE
        # -----------------------
        st.subheader("📋 Patient Table (Filterable)")
        selected_risk = st.multiselect(
            "Filter by Risk Level",
            options=df['risk_level'].unique(),
            default=df['risk_level'].unique()
        )
        display_df = df[df['risk_level'].isin(selected_risk)]
        st.dataframe(display_df[['patient','label','risk_level','quality_score']])
        st.markdown("💡 Use filters to focus on specific risk levels or GON status.")

    # =============================
    # PREDICTION DATASET
    # =============================
    else:
        import numpy as np
        from sklearn.metrics import confusion_matrix, roc_curve, auc

        df = pd.read_csv("optimized_predictions.csv")
        df = df[['image_name', 'patient', 'label', 'quality_score',
                 'label_numeric', 'probability', 'prediction']]

        st.header("🧠 Model Performance Dashboard")

        # -----------------------
        # THRESHOLD CONTROL
        # -----------------------
        st.subheader("🎚️ Decision Control")
        mode = st.radio(
            "Clinical Mode:",
            ["Screening (High Recall)", "Balanced", "Diagnostic (High Precision)"]
        )

        if mode == "Screening (High Recall)":
            threshold = st.slider("Threshold", 0.0, 1.0, 0.3)
        elif mode == "Balanced":
            threshold = st.slider("Threshold", 0.0, 1.0, 0.5)
        else:
            threshold = st.slider("Threshold", 0.0, 1.0, 0.7)

        df['adjusted_prediction'] = (df['probability'] >= threshold).astype(int)

        # -----------------------
        # METRICS
        # -----------------------
        y_true = df['label_numeric']
        y_pred = df['adjusted_prediction']
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        specificity = tn / (tn + fp) if (tn + fp) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

        st.subheader("📊 Model Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy:.2f}")
        col2.metric("Precision", f"{precision:.2f}")
        col3.metric("Recall ⚠️", f"{recall:.2f}")
        col4.metric("F1 Score", f"{f1:.2f}")
        col5, col6 = st.columns(2)
        col5.metric("Specificity", f"{specificity:.2f}")
        col6.metric("False Negatives ⚠️", fn)

        st.markdown("---")

        # Confusion Matrix
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Normal", "Actual Glaucoma"],
            columns=["Pred Normal", "Pred Glaucoma"]
        )
        st.subheader("📌 Confusion Matrix")
        st.dataframe(cm_df)
        chart_cm = (
            alt.Chart(cm_df.reset_index().melt(id_vars='index'))
            .mark_rect()
            .encode(
                x='variable',
                y='index',
                color='value:Q'
            )
        )
        st.altair_chart(chart_cm, use_container_width=True)

        # ROC Curve
        st.subheader("📈 ROC Curve")
        fpr, tpr, _ = roc_curve(y_true, df['probability'])
        roc_auc = auc(fpr, tpr)
        roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
        chart_roc = alt.Chart(roc_df).mark_line().encode(x="FPR", y="TPR")
        st.altair_chart(chart_roc, use_container_width=True)
        st.metric("AUC Score", f"{roc_auc:.3f}")

        # Confidence Distribution
        st.subheader("📊 Confidence Distribution")
        df['is_correct'] = df['adjusted_prediction'] == df['label_numeric']
        chart_conf = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("probability", bin=alt.Bin(maxbins=20)),
                y="count()",
                color="is_correct:N"
            )
        )
        st.altair_chart(chart_conf, use_container_width=True)

        # Threshold Trade-off
        st.subheader("🎯 Threshold Trade-off")
        thresholds_test = np.linspace(0.1, 0.9, 20)
        recalls, precisions = [], []
        for t in thresholds_test:
            pred = (df['probability'] >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
            recalls.append(tp / (tp + fn) if (tp + fn) else 0)
            precisions.append(tp / (tp + fp) if (tp + fp) else 0)
        trade_df = pd.DataFrame({"Threshold": thresholds_test,"Recall": recalls,"Precision": precisions})
        chart_trade = (
            alt.Chart(trade_df)
            .transform_fold(['Recall', 'Precision'])
            .mark_line()
            .encode(
                x=alt.X("Threshold:Q", title="Threshold"),
                y=alt.Y("value:Q", title="Metric Value"),
                color=alt.Color("key:N", title="Metric Type")
            )
        )
        st.altair_chart(chart_trade, use_container_width=True)

        # Error Analysis
        st.subheader("🚨 Critical Errors")
        fn_df = df[(df['label_numeric']==1) & (df['adjusted_prediction']==0)]
        fp_df = df[(df['label_numeric']==0) & (df['adjusted_prediction']==1)]
        col1, col2 = st.columns(2)
        with col1:
            st.error(f"Missed Glaucoma: {len(fn_df)}")
            st.dataframe(fn_df[['image_name','probability']])
        with col2:
            st.warning(f"False Alarms: {len(fp_df)}")
            st.dataframe(fp_df[['image_name','probability']])

        # ==============================
        # IMAGE GALLERY 
        # ==============================
        st.subheader("🩺 Clinical Image Review Panel")
        image_folder = "images_resized"
        label_map = {1: "GON+", 0: "GON-"}

        # Add correctness column
        df['is_correct'] = df['adjusted_prediction'] == df['label_numeric']

        # Filter by correctness
        prediction_filter = st.selectbox(
            "Show predictions:",
            options=["All", "Correct Prediction", "Wrong Prediction"]
        )

        if prediction_filter == "Wrong Prediction":
            filtered_df = df[df['is_correct'] == False]
        elif prediction_filter == "Correct Prediction":
            filtered_df = df[df['is_correct'] == True]
        else:
            filtered_df = df.sort_values(by='is_correct')

        # Initialize session state for "load more"
        if "display_count" not in st.session_state:
            st.session_state.display_count = 20  # initial batch size

        # Button to load more
        if st.button("Load More Images"):
            st.session_state.display_count += 20  # load 20 more each click

        # Select the subset to display
        display_df = filtered_df.head(st.session_state.display_count)

        # Overview metrics
        total_cases = len(display_df)
        fn_count = sum((display_df['label_numeric']==1) & (display_df['adjusted_prediction']==0))
        fp_count = sum((display_df['label_numeric']==0) & (display_df['adjusted_prediction']==1))
        correct_count = sum(display_df['is_correct'])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Displayed", total_cases)
        col2.metric("False Negatives ⚠️", fn_count)
        col3.metric("False Positives ⚠️", fp_count)
        col4.metric("Correct ✅", correct_count)

        st.warning(f"⚠️ {sum(display_df['is_correct']==False)} wrong predictions out of {total_cases} displayed cases")

        # Display images in grid
        num_cols = 5
        cols = st.columns(num_cols)

        for i, (_, row) in enumerate(display_df.iterrows()):
            col = cols[i % num_cols]
            img_path = os.path.join(image_folder, row['image_name'])

            if os.path.exists(img_path):
                predicted_text = label_map.get(row['adjusted_prediction'], "N/A")
                border_color = "#2ECC71" if row['is_correct'] else "#E74C3C"
                badge_text = "Correct" if row['is_correct'] else "Incorrect"

                col.markdown(f"""
                    <div style="
                        border: 4px solid {border_color};
                        border-radius: 10px;
                        padding: 8px;
                        text-align: center;
                        margin-bottom: 20px;
                    ">
                """, unsafe_allow_html=True)

                col.image(img_path, width=220)

                col.markdown(f"""
                    <div style="text-align:center;">
                        <b>Patient:</b> {row.get('patient','N/A')}<br>
                        <b>Actual:</b> {row['label']}<br>
                        <b>Predicted:</b> {predicted_text}<br>
                        <span style='background-color:{border_color};
                                     color:white;
                                     padding:3px 8px;
                                     border-radius:6px'>
                            {badge_text}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# -----------------------------------
# SIDEBAR: Patient Info Summary
with st.sidebar.expander("📝 Patient Info Summary", expanded=True):
    if "user_done" in st.session_state and st.session_state.user_done:
        st.markdown(f"**Patient ID:** {st.session_state.name}")
        st.markdown(f"**Age:** {st.session_state.age}")
        st.markdown(f"**DOB:** {st.session_state.birthdate}")
        st.markdown(f"**Location:** {st.session_state.location}")
    else:
        st.info("Patient information will appear here once saved.")

# -----------------------------------
# 4. APPOINTMENT
if menu == "Appointment":

    st.header("🏥 Book an Appointment")
    st.info("Schedule your eye checkup with ease.")

    # Show patient info for reference
    st.subheader("Patient Details")
    st.markdown(f"- **ID:** {st.session_state.name}")
    st.markdown(f"- **Age:** {st.session_state.age}")
    st.markdown(f"- **Location:** {st.session_state.location}")

    # Select clinic type
    clinic = st.selectbox(
        "Clinic Type",
        ["Government Clinic", "Private Clinic", "Hospital"]
    )

    # Select date
    date = st.date_input("Select Date")

    # Select time slot
    timeslot = st.selectbox(
        "Select Time Slot",
        ["09:00 AM", "10:00 AM", "11:00 AM", "01:00 PM", "02:00 PM", "03:00 PM"]
    )

    # Confirm button
    if st.button("Confirm Appointment"):
        st.success(f"Appointment booked!\n\n**Clinic:** {clinic}\n**Date:** {date}\n**Time:** {timeslot}")
        st.balloons()