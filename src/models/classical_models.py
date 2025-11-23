"""
Classical Machine Learning Models with Modern Best Practices
"""
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
import joblib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import MODEL_CONFIG, CV_FOLDS, RANDOM_SEED


class ClassicalMLModels:
    """
    Classical ML models with hyperparameter tuning and evaluation
    """

    def __init__(self):
        self.models = {}
        self.best_params = {}
        self.cv_scores = {}

    def train_decision_tree(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        tune_hyperparams: bool = False
    ) -> DecisionTreeClassifier:
        """Train Decision Tree with optional hyperparameter tuning"""
        print("\n" + "="*50)
        print("Training Decision Tree")
        print("="*50)

        if tune_hyperparams:
            param_grid = {
                'max_depth': [10, 15, 20, 25, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            }

            dt = DecisionTreeClassifier(random_state=RANDOM_SEED)
            grid_search = GridSearchCV(
                dt, param_grid, cv=CV_FOLDS,
                scoring='accuracy', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)

            self.models['decision_tree'] = grid_search.best_estimator_
            self.best_params['decision_tree'] = grid_search.best_params_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            dt = DecisionTreeClassifier(**MODEL_CONFIG['decision_tree'])
            dt.fit(X_train, y_train)
            self.models['decision_tree'] = dt

        # Cross-validation
        cv_scores = cross_val_score(
            self.models['decision_tree'], X_train, y_train,
            cv=CV_FOLDS, scoring='accuracy'
        )
        self.cv_scores['decision_tree'] = cv_scores
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        return self.models['decision_tree']

    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        tune_hyperparams: bool = False
    ) -> RandomForestClassifier:
        """Train Random Forest with optional hyperparameter tuning"""
        print("\n" + "="*50)
        print("Training Random Forest")
        print("="*50)

        if tune_hyperparams:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [15, 20, 25, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }

            rf = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)
            grid_search = GridSearchCV(
                rf, param_grid, cv=CV_FOLDS,
                scoring='accuracy', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)

            self.models['random_forest'] = grid_search.best_estimator_
            self.best_params['random_forest'] = grid_search.best_params_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            rf = RandomForestClassifier(**MODEL_CONFIG['random_forest'])
            rf.fit(X_train, y_train)
            self.models['random_forest'] = rf

        # Cross-validation
        cv_scores = cross_val_score(
            self.models['random_forest'], X_train, y_train,
            cv=CV_FOLDS, scoring='accuracy'
        )
        self.cv_scores['random_forest'] = cv_scores
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        return self.models['random_forest']

    def train_naive_bayes(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> GaussianNB:
        """Train Naive Bayes classifier"""
        print("\n" + "="*50)
        print("Training Naive Bayes")
        print("="*50)

        nb = GaussianNB(**MODEL_CONFIG['naive_bayes'])
        nb.fit(X_train, y_train)
        self.models['naive_bayes'] = nb

        # Cross-validation
        cv_scores = cross_val_score(
            nb, X_train, y_train,
            cv=CV_FOLDS, scoring='accuracy'
        )
        self.cv_scores['naive_bayes'] = cv_scores
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        return nb

    def evaluate_model(
        self,
        model_name: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        disease_names: Dict[int, str] = None
    ) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        model = self.models[model_name]
        y_pred = model.predict(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }

        print(f"\n{'='*50}")
        print(f"Evaluation Results - {model_name.upper()}")
        print(f"{'='*50}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"Recall (macro): {metrics['recall_macro']:.4f}")
        print(f"F1-Score (macro): {metrics['f1_macro']:.4f}")
        print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
        print(f"Recall (weighted): {metrics['recall_weighted']:.4f}")
        print(f"F1-Score (weighted): {metrics['f1_weighted']:.4f}")

        if model_name in self.cv_scores:
            cv_mean = self.cv_scores[model_name].mean()
            cv_std = self.cv_scores[model_name].std()
            print(f"CV Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")

        return metrics

    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.models[model_name].predict(X)

    def predict_proba(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if hasattr(self.models[model_name], 'predict_proba'):
            return self.models[model_name].predict_proba(X)
        return None

    def save_model(self, model_name: str, filepath: Path):
        """Save trained model"""
        joblib.dump(self.models[model_name], filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, model_name: str, filepath: Path):
        """Load trained model"""
        self.models[model_name] = joblib.load(filepath)
        print(f"Model loaded from {filepath}")

    def get_feature_importance(self, model_name: str) -> np.ndarray:
        """Get feature importances (for tree-based models)"""
        model = self.models[model_name]
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        return None


def main():
    """Test classical models"""
    print("Classical ML Models module loaded successfully!")


if __name__ == "__main__":
    main()
