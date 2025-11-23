"""
Model Explainability Module using SHAP (2025)
Provides interpretability for both classical and deep learning models
"""
import shap
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))


class ModelExplainer:
    """
    Provides model explanations using SHAP values
    """

    def __init__(self, model, model_type: str = 'tree'):
        """
        Initialize explainer

        Args:
            model: Trained model
            model_type: Type of model ('tree', 'linear', 'deep')
        """
        self.model = model
        self.model_type = model_type
        self.explainer = None

    def create_explainer(self, X_train: np.ndarray):
        """Create SHAP explainer based on model type"""
        print(f"Creating SHAP explainer for {self.model_type} model...")

        if self.model_type in ['tree', 'random_forest', 'decision_tree']:
            # Tree explainer for tree-based models
            self.explainer = shap.TreeExplainer(self.model)

        elif self.model_type == 'linear':
            # Linear explainer for linear models
            self.explainer = shap.LinearExplainer(self.model, X_train)

        elif self.model_type == 'deep':
            # Deep explainer for neural networks
            # Use a subset of training data as background
            background = shap.sample(X_train, 100)
            self.explainer = shap.DeepExplainer(self.model, background)

        else:
            # Kernel explainer as fallback (model-agnostic)
            background = shap.sample(X_train, 100)
            self.explainer = shap.KernelExplainer(self.model.predict_proba, background)

        print("Explainer created successfully!")

    def explain_instance(
        self,
        X: np.ndarray,
        feature_names: List[str] = None
    ) -> shap.Explanation:
        """
        Explain a single prediction

        Args:
            X: Input features (single instance)
            feature_names: List of feature names

        Returns:
            SHAP explanation object
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer first.")

        shap_values = self.explainer.shap_values(X)

        return shap_values

    def plot_waterfall(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_names: List[str] = None,
        max_display: int = 10,
        save_path: Path = None
    ):
        """
        Create waterfall plot showing feature contributions

        Args:
            shap_values: SHAP values for the instance
            X: Input features
            feature_names: List of feature names
            max_display: Maximum features to display
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 6))

        # For multi-class, use the predicted class
        if isinstance(shap_values, list):
            predicted_class = np.argmax([sv[0].sum() for sv in shap_values])
            shap_values_to_plot = shap_values[predicted_class][0]
        else:
            shap_values_to_plot = shap_values[0]

        # Create explanation object
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(X[0]))]

        explanation = shap.Explanation(
            values=shap_values_to_plot,
            base_values=self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
            data=X[0],
            feature_names=feature_names
        )

        shap.plots.waterfall(explanation, max_display=max_display, show=False)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Waterfall plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_force(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_names: List[str] = None,
        save_path: Path = None
    ):
        """
        Create force plot showing feature contributions
        """
        if isinstance(shap_values, list):
            predicted_class = np.argmax([sv[0].sum() for sv in shap_values])
            shap_values_to_plot = shap_values[predicted_class][0]
            expected_value = self.explainer.expected_value[predicted_class]
        else:
            shap_values_to_plot = shap_values[0]
            expected_value = self.explainer.expected_value

        force_plot = shap.force_plot(
            expected_value,
            shap_values_to_plot,
            X[0],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Force plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_summary(
        self,
        X: np.ndarray,
        feature_names: List[str] = None,
        max_display: int = 20,
        save_path: Path = None
    ):
        """
        Create summary plot showing overall feature importance
        """
        shap_values = self.explainer.shap_values(X)

        plt.figure(figsize=(12, 8))

        shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_names,
            max_display=max_display,
            show=False
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Summary plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def get_top_features(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        top_k: int = 10
    ) -> Dict[str, float]:
        """
        Get top K most important features for a prediction

        Returns:
            Dictionary of feature names and their SHAP values
        """
        if isinstance(shap_values, list):
            # Multi-class: use predicted class
            predicted_class = np.argmax([sv[0].sum() for sv in shap_values])
            shap_values_to_use = shap_values[predicted_class][0]
        else:
            shap_values_to_use = shap_values[0]

        # Get absolute values for ranking
        abs_values = np.abs(shap_values_to_use)

        # Get top K indices
        top_indices = np.argsort(abs_values)[-top_k:][::-1]

        # Create dictionary
        top_features = {
            feature_names[idx]: shap_values_to_use[idx]
            for idx in top_indices
        }

        return top_features


def main():
    """Test explainability module"""
    print("Model Explainability module loaded successfully!")
    print("SHAP-based explanations available for all model types")


if __name__ == "__main__":
    main()
