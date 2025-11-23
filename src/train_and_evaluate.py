"""
Main Training and Evaluation Script for Disease Prediction Models (2023)
Includes Classical ML, Deep Learning, and Transformer models
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add to path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.data_loader import DiseaseDataLoader
from src.models.classical_models import ClassicalMLModels
from src.models.deep_learning_models import DeepLearningTrainer
from configs.config import MODELS_DIR


def plot_confusion_matrix(cm, classes, title, save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()

    plt.close()


def train_classical_models(tune_hyperparams=False):
    """Train and evaluate classical ML models"""
    print("\n" + "="*80)
    print("CLASSICAL MACHINE LEARNING MODELS")
    print("="*80)

    # Load and preprocess data
    loader = DiseaseDataLoader()
    train_df, test_df = loader.load_data()
    X_train, X_test, y_train, y_test, features, diseases = loader.preprocess_data(
        train_df, test_df, scale_features=True
    )

    # Initialize models
    ml_models = ClassicalMLModels()

    # Train models
    print("\n1. Training Decision Tree...")
    ml_models.train_decision_tree(X_train, y_train, tune_hyperparams=tune_hyperparams)

    print("\n2. Training Random Forest...")
    ml_models.train_random_forest(X_train, y_train, tune_hyperparams=tune_hyperparams)

    print("\n3. Training Naive Bayes...")
    ml_models.train_naive_bayes(X_train, y_train)

    # Evaluate models
    results = {}

    for model_name in ['decision_tree', 'random_forest', 'naive_bayes']:
        metrics = ml_models.evaluate_model(model_name, X_test, y_test, diseases)
        results[model_name] = metrics

        # Save model
        model_path = MODELS_DIR / f"{model_name}_model.pkl"
        ml_models.save_model(model_name, model_path)

    # Compare models
    print("\n" + "="*80)
    print("MODEL COMPARISON - CLASSICAL ML")
    print("="*80)
    comparison_df = pd.DataFrame(results).T
    print(comparison_df.round(4))

    return ml_models, results, loader


def train_deep_learning_models():
    """Train and evaluate deep learning models"""
    print("\n" + "="*80)
    print("DEEP LEARNING MODELS")
    print("="*80)

    # Load and preprocess data
    loader = DiseaseDataLoader()
    train_df, test_df = loader.load_data()
    X_train, X_test, y_train, y_test, features, diseases = loader.preprocess_data(
        train_df, test_df, scale_features=True
    )

    # Create validation split
    X_tr, X_val, y_tr, y_val = loader.create_validation_split(X_train, y_train)

    input_dim = X_train.shape[1]
    output_dim = len(diseases)

    results = {}

    # Train Neural Network
    print("\n1. Training Deep Neural Network...")
    nn_trainer = DeepLearningTrainer(
        model_type='neural_network',
        input_dim=input_dim,
        output_dim=output_dim
    )

    nn_history = nn_trainer.fit(X_tr, y_tr, X_val, y_val, verbose=True)
    nn_results = nn_trainer.evaluate(X_test, y_test)
    results['neural_network'] = nn_results

    # Save model
    nn_path = MODELS_DIR / "neural_network_model.pth"
    nn_trainer.save_model(nn_path)

    # Train Transformer
    print("\n2. Training Transformer Model...")
    transformer_trainer = DeepLearningTrainer(
        model_type='transformer',
        input_dim=input_dim,
        output_dim=output_dim
    )

    transformer_history = transformer_trainer.fit(X_tr, y_tr, X_val, y_val, verbose=True)
    transformer_results = transformer_trainer.evaluate(X_test, y_test)
    results['transformer'] = transformer_results

    # Save model
    transformer_path = MODELS_DIR / "transformer_model.pth"
    transformer_trainer.save_model(transformer_path)

    # Plot training curves
    plot_training_curves(nn_history, transformer_history)

    return results, loader


def plot_training_curves(nn_history, transformer_history):
    """Plot training curves for deep learning models"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Neural Network - Loss
    axes[0, 0].plot(nn_history['train_losses'], label='Train Loss')
    axes[0, 0].plot(nn_history['val_losses'], label='Val Loss')
    axes[0, 0].set_title('Neural Network - Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Neural Network - Accuracy
    axes[0, 1].plot(nn_history['train_accs'], label='Train Acc')
    axes[0, 1].plot(nn_history['val_accs'], label='Val Acc')
    axes[0, 1].set_title('Neural Network - Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Transformer - Loss
    axes[1, 0].plot(transformer_history['train_losses'], label='Train Loss')
    axes[1, 0].plot(transformer_history['val_losses'], label='Val Loss')
    axes[1, 0].set_title('Transformer - Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Transformer - Accuracy
    axes[1, 1].plot(transformer_history['train_accs'], label='Train Acc')
    axes[1, 1].plot(transformer_history['val_accs'], label='Val Acc')
    axes[1, 1].set_title('Transformer - Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    save_path = MODELS_DIR / "training_curves.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining curves saved to {save_path}")
    plt.close()


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("DISEASE PREDICTION SYSTEM - 2023 VERSION")
    print("Modern ML, Deep Learning, and Transformer Models")
    print("="*80)

    # Create models directory if it doesn't exist
    MODELS_DIR.mkdir(exist_ok=True)

    # Train classical models
    print("\n[1/2] Training Classical ML Models...")
    ml_models, ml_results, loader = train_classical_models(tune_hyperparams=False)

    # Train deep learning models
    print("\n[2/2] Training Deep Learning Models...")
    dl_results, _ = train_deep_learning_models()

    # Final comparison
    print("\n" + "="*80)
    print("FINAL MODEL COMPARISON - ALL MODELS")
    print("="*80)

    all_results = {
        'Decision Tree': ml_results['decision_tree']['accuracy'],
        'Random Forest': ml_results['random_forest']['accuracy'],
        'Naive Bayes': ml_results['naive_bayes']['accuracy'],
        'Neural Network': dl_results['neural_network']['test_accuracy'] / 100,
        'Transformer': dl_results['transformer']['test_accuracy'] / 100
    }

    for model, accuracy in all_results.items():
        print(f"{model:20s}: {accuracy*100:.2f}%")

    best_model = max(all_results, key=all_results.get)
    print(f"\nBest Model: {best_model} ({all_results[best_model]*100:.2f}%)")

    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nModels saved in: {MODELS_DIR}")
    print("\nNext steps:")
    print("1. Run the Streamlit app: streamlit run src/app.py")
    print("2. Or use the models for predictions programmatically")


if __name__ == "__main__":
    main()
