"""
Classifiers Module for SEED EEG Emotion Recognition

This module implements various machine learning classifiers for EEG-based emotion recognition:
- Linear SVM
- Logistic Regression
- k-Nearest Neighbors
- Deep Belief Network
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class DeepBeliefNetwork(nn.Module):
    """
    Deep Belief Network implementation using PyTorch.
    
    A simple feedforward neural network that serves as a deep architecture
    for comparison with shallow methods.
    """
    
    def __init__(self, input_dim: int, hidden_layers: List[int], n_classes: int, dropout_rate: float = 0.2):
        """
        Initialize Deep Belief Network.
        
        Args:
            input_dim (int): Input feature dimension
            hidden_layers (List[int]): List of hidden layer sizes
            n_classes (int): Number of output classes
            dropout_rate (float): Dropout rate for regularization
        """
        super(DeepBeliefNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, n_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Output logits with shape (batch_size, n_classes)
        """
        return self.network(x)

class EEGClassifier:
    """
    Unified classifier interface for EEG emotion recognition.
    
    Supports multiple classifier types and provides consistent training/evaluation interface.
    """
    
    def __init__(self, classifier_type: str, config: Dict):
        """
        Initialize EEG classifier.
        
        Args:
            classifier_type (str): Type of classifier ('svm', 'logistic', 'knn', 'dbn')
            config (Dict): Configuration dictionary
        """
        self.classifier_type = classifier_type
        self.config = config
        self.scaler = StandardScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() and config['computation']['use_gpu'] else 'cpu')
        
        logger.info(f"Initialized {classifier_type} classifier on device: {self.device}")
    
    def _create_model(self, input_dim: int, n_classes: int = 3) -> Any:
        """
        Create the appropriate model based on classifier type.
        
        Args:
            input_dim (int): Input feature dimension
            n_classes (int): Number of classes
            
        Returns:
            Any: Initialized model
        """
        if self.classifier_type == 'svm':
            return SVC(
                kernel=self.config['models']['svm']['kernel'],
                C=self.config['models']['svm']['C'],
                random_state=self.config['models']['svm']['random_state']
            )
        
        elif self.classifier_type == 'logistic':
            return LogisticRegression(
                penalty=self.config['models']['logistic']['penalty'],
                C=self.config['models']['logistic']['C'],
                max_iter=self.config['models']['logistic']['max_iter'],
                random_state=self.config['models']['logistic']['random_state']
            )
        
        elif self.classifier_type == 'knn':
            return KNeighborsClassifier(
                n_neighbors=self.config['models']['knn']['n_neighbors'],
                weights=self.config['models']['knn']['weights']
            )
        
        elif self.classifier_type == 'dbn':
            return DeepBeliefNetwork(
                input_dim=input_dim,
                hidden_layers=self.config['models']['dbn']['hidden_layers'],
                n_classes=n_classes
            ).to(self.device)
        
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the classifier.
        
        Args:
            X_train (np.ndarray): Training features with shape (n_samples, n_features)
            y_train (np.ndarray): Training labels with shape (n_samples,)
        """
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Create and train model
        self.model = self._create_model(X_train_scaled.shape[1])
        
        if self.classifier_type == 'dbn':
            self._train_dbn(X_train_scaled, y_train)
        else:
            self.model.fit(X_train_scaled, y_train)
        
        logger.info(f"Trained {self.classifier_type} classifier on {len(X_train)} samples")
    
    def _train_dbn(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train Deep Belief Network using PyTorch.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
        """
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.LongTensor(y_train).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['models']['dbn']['batch_size'],
            shuffle=True
        )
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['models']['dbn']['learning_rate']
        )
        
        # Training loop
        self.model.train()
        for epoch in range(self.config['models']['dbn']['n_epochs']):
            total_loss = 0.0
            correct = 0
            total = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            if (epoch + 1) % 20 == 0:
                accuracy = 100 * correct / total
                logger.debug(f"Epoch [{epoch+1}/{self.config['models']['dbn']['n_epochs']}], "
                           f"Loss: {total_loss/len(dataloader):.4f}, Accuracy: {accuracy:.2f}%")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions on test data.
        
        Args:
            X_test (np.ndarray): Test features with shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Predicted labels with shape (n_samples,)
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Standardize features using fitted scaler
        X_test_scaled = self.scaler.transform(X_test)
        
        if self.classifier_type == 'dbn':
            return self._predict_dbn(X_test_scaled)
        else:
            return self.model.predict(X_test_scaled)
    
    def _predict_dbn(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions using Deep Belief Network.
        
        Args:
            X_test (np.ndarray): Test features
            
        Returns:
            np.ndarray: Predicted labels
        """
        self.model.eval()
        X_tensor = torch.FloatTensor(X_test).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        
        return predicted.cpu().numpy()
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X_test (np.ndarray): Test features
            
        Returns:
            np.ndarray: Class probabilities with shape (n_samples, n_classes)
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        X_test_scaled = self.scaler.transform(X_test)
        
        if self.classifier_type == 'dbn':
            return self._predict_proba_dbn(X_test_scaled)
        elif hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_test_scaled)
        else:
            # For models without predict_proba, return one-hot encoded predictions
            predictions = self.model.predict(X_test_scaled)
            n_classes = len(np.unique(predictions))
            probas = np.zeros((len(predictions), n_classes))
            probas[np.arange(len(predictions)), predictions] = 1.0
            return probas
    
    def _predict_proba_dbn(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using Deep Belief Network.
        
        Args:
            X_test (np.ndarray): Test features
            
        Returns:
            np.ndarray: Class probabilities
        """
        self.model.eval()
        X_tensor = torch.FloatTensor(X_test).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probas = torch.softmax(outputs, dim=1)
        
        return probas.cpu().numpy()
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate classifier performance.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)
        class_report = classification_report(y_test, predictions, output_dict=True)
        
        results = {
            'accuracy': accuracy,
            'predictions': predictions,
            'probabilities': probabilities,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report
        }
        
        logger.info(f"{self.classifier_type} accuracy: {accuracy:.4f}")
        
        return results

class ClassifierComparison:
    """
    Utility class for comparing multiple classifiers.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize classifier comparison.
        
        Args:
            config (Dict): Configuration dictionary
        """
        self.config = config
        self.classifiers = {}
        self.results = {}
        
    def add_classifier(self, name: str, classifier_type: str) -> None:
        """
        Add a classifier to the comparison.
        
        Args:
            name (str): Name for the classifier
            classifier_type (str): Type of classifier
        """
        self.classifiers[name] = EEGClassifier(classifier_type, self.config)
        logger.info(f"Added {classifier_type} classifier as '{name}'")
    
    def train_all(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train all classifiers.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
        """
        for name, classifier in self.classifiers.items():
            logger.info(f"Training {name}...")
            classifier.train(X_train, y_train)
    
    def evaluate_all(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict]:
        """
        Evaluate all classifiers.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            Dict[str, Dict]: Results for each classifier
        """
        self.results = {}
        
        for name, classifier in self.classifiers.items():
            logger.info(f"Evaluating {name}...")
            self.results[name] = classifier.evaluate(X_test, y_test)
        
        return self.results
    
    def get_accuracy_summary(self) -> Dict[str, float]:
        """
        Get accuracy summary for all classifiers.
        
        Returns:
            Dict[str, float]: Accuracy for each classifier
        """
        return {name: results['accuracy'] for name, results in self.results.items()}