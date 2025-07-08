# 🐶 Cat vs Dog Classifier 🐱  
A deep learning web app that classifies whether an uploaded image is of a **Cat or a Dog** using **Transfer Learning** with **MobileNetV2** and a user-friendly **Streamlit interface**.


---

## 🔍 Project Overview

This project uses a pre-trained **MobileNetV2** model, fine-tuned to classify cats vs dogs, deployed via **Streamlit**.

### 🧠 Features:
- ✅ Real-time image upload
- ✅ Predicts if image is a Cat or Dog
- ✅ Shows prediction confidence
- ✅ Beautiful web interface with Streamlit
- ✅ Free deployment using Hugging Face Spaces

---

## 📂 Folder Structure

cat-vs-dog-classifier/
            
├── app.py # Streamlit app

├── cat_dog_model.h5 # Pre-trained Keras model

├── requirements.txt # Python dependencies

├── README.md # Project description

└── sample_images/ # Example test images


---

## 🚀 How to Run Locally

### ✅ 1. Clone the Repo

```bash
git clone https://github.com/hulkalan/CAT_VS_DOG.git

```

### ✅ 2. Create and Activate Virtual Environment
````bash
python -m venv .venv
.venv\Scripts\activate   # On Windows
# OR
source .venv/bin/activate  # On macOS/Linux
````
### ✅ 3. Install Requirements
````bash
pip install -r requirements.txt
````
### ✅ 4. Run the Streamlit App
````bash
streamlit run app.py

````

### 📈 Model Performance
   - 📊 Accuracy: ~93% on validation data

   - 📁 Dataset: Kaggle Dogs vs Cats Dataset

   - 🔁 Preprocessing: Resized to 150x150, normalized

### 🤝 Contributions
PRs and suggestions are welcome! Feel free to fork and improve this project.

📜 License
  MIT License.
  Free to use, modify, and distribute.

🙋‍♂️ Author
  
 **Alan Mathew**

💼 LinkedIn • 🐙 GitHub