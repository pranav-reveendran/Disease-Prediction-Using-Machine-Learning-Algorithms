"""
Modern Deep Learning Models for Disease Prediction (2025)
Includes Neural Networks and Transformer-based architectures
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from typing import Dict, Any, Tuple, List
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import MODEL_CONFIG, RANDOM_SEED

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class DeepNeuralNetwork(nn.Module):
    """
    Deep Neural Network for disease prediction with modern architecture
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_layers: List[int] = None,
                 dropout: float = 0.3):
        super(DeepNeuralNetwork, self).__init__()

        if hidden_layers is None:
            hidden_layers = MODEL_CONFIG['neural_network']['hidden_layers']

        layers = []
        prev_dim = input_dim

        # Build hidden layers with BatchNorm and Dropout
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TransformerClassifier(nn.Module):
    """
    Transformer-based classifier for disease prediction (2025 approach)
    """

    def __init__(self, input_dim: int, output_dim: int,
                 d_model: int = 128, nhead: int = 8,
                 num_layers: int = 4, dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(TransformerClassifier, self).__init__()

        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding (for sequence modeling)
        self.pos_encoder = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

    def forward(self, x):
        # Project input to d_model dimensions
        x = self.input_projection(x)

        # Add positional encoding
        x = x.unsqueeze(1)  # Add sequence dimension
        x = x + self.pos_encoder

        # Pass through transformer
        x = self.transformer_encoder(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Classification
        x = self.classifier(x)

        return x


class DeepLearningTrainer:
    """
    Trainer class for deep learning models
    """

    def __init__(self, model_type: str = 'neural_network',
                 input_dim: int = 132, output_dim: int = 41):
        self.model_type = model_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize model
        if model_type == 'neural_network':
            config = MODEL_CONFIG['neural_network']
            self.model = DeepNeuralNetwork(
                input_dim, output_dim,
                hidden_layers=config['hidden_layers'],
                dropout=config['dropout']
            ).to(self.device)
            self.config = config
        elif model_type == 'transformer':
            config = MODEL_CONFIG['transformer']
            self.model = TransformerClassifier(
                input_dim, output_dim,
                d_model=config['d_model'],
                nhead=config['nhead'],
                num_layers=config['num_layers'],
                dim_feedforward=config['dim_feedforward'],
                dropout=config['dropout']
            ).to(self.device)
            self.config = config
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate']
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )

        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def prepare_data(self, X: np.ndarray, y: np.ndarray,
                     batch_size: int = None) -> DataLoader:
        """Prepare data loader"""
        if batch_size is None:
            batch_size = self.config['batch_size']

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset, batch_size=batch_size,
            shuffle=True
        )

        return dataloader

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            epochs: int = None, verbose: bool = True) -> Dict[str, List]:
        """Train the model"""
        if epochs is None:
            epochs = self.config['epochs']

        train_loader = self.prepare_data(X_train, y_train)

        if X_val is not None and y_val is not None:
            val_loader = self.prepare_data(X_val, y_val)
            use_validation = True
        else:
            use_validation = False

        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config.get('early_stopping_patience', 10)

        print(f"\n{'='*60}")
        print(f"Training {self.model_type.upper()}")
        print(f"{'='*60}")

        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # Validate
            if use_validation:
                val_loss, val_acc = self.validate(val_loader)
                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)

                # Learning rate scheduling
                self.scheduler.step(val_loss)

                if verbose and (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.best_model_state = self.model.state_dict()
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    # Restore best model
                    self.model.load_state_dict(self.best_model_state)
                    break
            else:
                if verbose and (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        print(f"Training completed!")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)

        return predicted.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        return probabilities.cpu().numpy()

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model"""
        test_loader = self.prepare_data(X_test, y_test)
        test_loss, test_acc = self.validate(test_loader)

        print(f"\n{'='*50}")
        print(f"Test Results - {self.model_type.upper()}")
        print(f"{'='*50}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")

        return {'test_loss': test_loss, 'test_accuracy': test_acc}

    def save_model(self, filepath: Path):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'model_type': self.model_type
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: Path):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {filepath}")


def main():
    """Test deep learning models"""
    print("Deep Learning Models module loaded successfully!")
    print(f"Available models: Neural Network, Transformer")


if __name__ == "__main__":
    main()
