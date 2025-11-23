# Quick Start Guide - Disease Prediction System 2023

Get up and running in **5 minutes**!

## 🚀 Installation (2 minutes)

```bash
# 1. Clone repository
git clone <your-repo-url>
cd Disease-Prediction-Using-Machine-Learning-Algorithms

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## 🎯 Option 1: Web Application (Fastest)

If you just want to use the app without training:

```bash
# Download pre-trained models (if available)
# OR train models first (see Option 2)

# Launch web app
streamlit run src/app.py
```

Open browser to `http://localhost:8501` and start predicting!

## 🧠 Option 2: Train Models First (Recommended)

```bash
# Train all 5 models (takes 5-10 minutes)
python src/train_and_evaluate.py

# Then launch web app
streamlit run src/app.py
```

**Training Output:**
```
Loading data...
Training Decision Tree... ✓
Training Random Forest... ✓
Training Naive Bayes... ✓
Training Neural Network... ✓
Training Transformer... ✓

FINAL RESULTS:
Decision Tree       : 95.12%
Random Forest       : 96.34%
Naive Bayes         : 93.87%
Neural Network      : 97.21%
Transformer         : 98.15% ← Best!

Models saved to: saved_models/
```

## 💻 Option 3: Use Programmatically

```python
from src.preprocessing.data_loader import DiseaseDataLoader
from src.models.classical_models import ClassicalMLModels

# Load data
loader = DiseaseDataLoader()
train_df, test_df = loader.load_data()
X_train, X_test, y_train, y_test, features, diseases = loader.preprocess_data(
    train_df, test_df
)

# Train a model
ml_models = ClassicalMLModels()
ml_models.train_random_forest(X_train, y_train)

# Make prediction
symptoms = ['fever', 'cough', 'fatigue', 'headache', 'muscle_pain']
X = loader.preprocess_symptoms(symptoms)
prediction = ml_models.predict('random_forest', X)
disease = diseases[prediction[0]]

print(f"Predicted Disease: {disease}")
```

## 🎨 Using the Web Interface

1. **Select Model**: Choose from 5 models (Transformer recommended)
2. **Enter Patient Name**: Optional
3. **Select Symptoms**: Search and select up to 5 symptoms
4. **Predict**: Click "Predict Disease" button
5. **View Results**: See predicted disease with confidence score

## 📊 Quick Model Comparison

Want to see all models perform?

```python
# Run the complete evaluation
python src/train_and_evaluate.py
```

This will:
- Train all 5 models
- Evaluate on test set
- Show performance comparison
- Save all models
- Generate training curves

## 🔍 Model Explainability

Understand why a model made a prediction:

```python
from src.utils.explainability import ModelExplainer
import joblib

# Load a trained model
model = joblib.load('saved_models/random_forest_model.pkl')

# Create explainer
explainer = ModelExplainer(model, model_type='tree')
explainer.create_explainer(X_train)

# Explain prediction
shap_values = explainer.explain_instance(X_sample, feature_names)
top_features = explainer.get_top_features(shap_values, feature_names, top_k=10)

print("Top Contributing Symptoms:", top_features)
```

## 🐛 Troubleshooting

### Import Errors
```bash
# Make sure you're in the virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### CUDA/GPU Issues
```bash
# If you don't have GPU, models will automatically use CPU
# To force CPU:
export CUDA_VISIBLE_DEVICES=""
python src/train_and_evaluate.py
```

### Port Already in Use
```bash
# Change Streamlit port
streamlit run src/app.py --server.port 8502
```

### Models Not Found Error
```bash
# Train models first!
python src/train_and_evaluate.py
```

## 📚 Next Steps

1. **Read Full Documentation**: See `README_2023.md`
2. **Experiment**: Try different models and compare
3. **Customize**: Modify hyperparameters in `configs/config.py`
4. **Deploy**: Consider Docker, AWS, or cloud deployment

## 💡 Pro Tips

- **Use Transformer** for best accuracy (~98%)
- **Use Random Forest** for good balance of speed and accuracy
- **Enable GPU** for faster deep learning training
- **Tune Hyperparameters** for even better performance
- **Use SHAP** to understand and trust predictions

## 🎓 Learning Resources

- Original paper on disease prediction
- Transformer architecture: "Attention is All You Need"
- SHAP explainability: [SHAP documentation](https://shap.readthedocs.io/)
- Streamlit tutorials: [Streamlit docs](https://docs.streamlit.io/)

## ❓ Need Help?

- Check `README_2023.md` for detailed documentation
- Open an issue on GitHub
- Review the code comments

---

**Happy Predicting! 🏥**
