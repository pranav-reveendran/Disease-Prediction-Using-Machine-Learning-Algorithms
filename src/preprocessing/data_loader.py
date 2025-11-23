"""
Data Loading and Preprocessing Module
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import TRAINING_DATA, TESTING_DATA, RANDOM_SEED


class DiseaseDataLoader:
    """
    Handles data loading, preprocessing, and feature engineering for disease prediction
    """

    def __init__(self, training_path: Path = TRAINING_DATA, testing_path: Path = TESTING_DATA):
        self.training_path = training_path
        self.testing_path = testing_path
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.disease_names = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and testing data"""
        print("Loading data...")
        train_df = pd.read_csv(self.training_path)
        test_df = pd.read_csv(self.testing_path)

        print(f"Training data shape: {train_df.shape}")
        print(f"Testing data shape: {test_df.shape}")

        return train_df, test_df

    def preprocess_data(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        scale_features: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Dict[int, str]]:
        """
        Preprocess data for model training

        Returns:
            X_train, X_test, y_train, y_test, feature_names, disease_mapping
        """
        # Separate features and target
        self.feature_names = [col for col in train_df.columns if col != 'prognosis']

        X_train = train_df[self.feature_names].values
        X_test = test_df[self.feature_names].values

        # Encode target labels
        y_train = self.label_encoder.fit_transform(train_df['prognosis'])
        y_test = self.label_encoder.transform(test_df['prognosis'])

        # Create disease mapping
        disease_mapping = {
            idx: disease for idx, disease in enumerate(self.label_encoder.classes_)
        }
        self.disease_names = disease_mapping

        # Scale features if requested
        if scale_features:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        print(f"Number of features: {len(self.feature_names)}")
        print(f"Number of diseases: {len(disease_mapping)}")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Testing samples: {X_test.shape[0]}")

        return X_train, X_test, y_train, y_test, self.feature_names, disease_mapping

    def create_validation_split(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        val_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create validation split from training data"""
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train,
            test_size=val_size,
            random_state=RANDOM_SEED,
            stratify=y_train
        )

        print(f"Training samples: {X_tr.shape[0]}")
        print(f"Validation samples: {X_val.shape[0]}")

        return X_tr, X_val, y_tr, y_val

    def get_feature_names(self) -> List[str]:
        """Return feature names"""
        return self.feature_names

    def get_disease_names(self) -> Dict[int, str]:
        """Return disease mapping"""
        return self.disease_names

    def preprocess_symptoms(self, symptoms: List[str]) -> np.ndarray:
        """
        Preprocess user input symptoms for prediction

        Args:
            symptoms: List of symptom names

        Returns:
            Preprocessed feature vector
        """
        if self.feature_names is None:
            raise ValueError("Model not fitted. Call preprocess_data first.")

        # Create feature vector
        feature_vector = np.zeros(len(self.feature_names))

        for symptom in symptoms:
            if symptom in self.feature_names:
                idx = self.feature_names.index(symptom)
                feature_vector[idx] = 1

        # Reshape and scale
        feature_vector = feature_vector.reshape(1, -1)
        feature_vector = self.scaler.transform(feature_vector)

        return feature_vector


def main():
    """Test data loading"""
    loader = DiseaseDataLoader()
    train_df, test_df = loader.load_data()
    X_train, X_test, y_train, y_test, features, diseases = loader.preprocess_data(
        train_df, test_df
    )

    print("\nData preprocessing completed successfully!")
    print(f"Unique diseases: {len(diseases)}")
    print(f"Sample diseases: {list(diseases.values())[:5]}")


if __name__ == "__main__":
    main()
