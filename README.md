# 🥖 Food-Not-Food Text Classifier

A lightweight NLP project that fine-tunes **DistilBERT** to detect whether a sentence is about food or not.  
Built with **PyTorch** and **Hugging Face Transformers**, and deployed as an interactive **Gradio** demo.

---

## ✨ Project Highlights
- **Model**: DistilBERT (`distilbert-base-uncased`) fine-tuned on a **250-caption** synthetic dataset  
- **Accuracy**: ~**95 %** on held-out test set  
- **Fast Inference**: Optimized batching enables **10k+ predictions/minute** with GPU acceleration  
- **Deployment**: Live Gradio demo + model hosted on the **Hugging Face Hub**

---

## 📂 Repository Structure
food-not-food-classifier/
├─ foodnotfoodtextclassification.py # Training & evaluation script
├─ models/ # Saved model checkpoints
├─ demos/food_not_food_classifier/ # Gradio demo app
└─ requirements.txt # Python dependencies

---

## 🚀 Quick Start

### 1️⃣ Clone the repo
```bash
git clone https://github.com/<your-username>/food-not-food-classifier.git
cd food-not-food-classifier
2️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
3️⃣ Train (optional)
To retrain the model:

python foodnotfoodtextclassification.py

This will:

Download the dataset from Hugging Face Datasets

Tokenize text

Fine-tune DistilBERT for 10 epochs

Save the model to models/food_not_food_text_classifier_base_uncased/

4️⃣ Launch the demo
bash
Copy code
cd demos/food_not_food_classifier
python app.py
Open the local URL to interact with the Gradio interface.

🧩 Usage in Python
python
Copy code
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="AbdulmajeedKabala/food_not_food_text_classifier_base_uncased"
)

classifier("A bowl of strawberries sat on the counter, glistening with water.")
# [{'label': 'food', 'score': 0.98}]
📊 Results
Metric	Value
Test Accuracy	~95 %
Training Epochs	10
Batch Size	32
Inference Speed	~0.02 s per sentence (GPU)

🛠️ Tech Stack
Python 3.10

PyTorch

Hugging Face Transformers / Datasets / Evaluate

Gradio (web demo)

Hugging Face Hub (model hosting)

🤝 Contributing
Issues and pull requests are welcome!
For major changes, please open an issue first to discuss what you’d like to change.

📜 License
Apache-2.0

Live Demo: Hugging Face Space
