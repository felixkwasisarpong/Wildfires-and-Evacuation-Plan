import sys
import xgboost as xgb
from sklearn.metrics import accuracy_score

class WildfirePredictor:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = None
    
    def train_model(self):
        """
        Trains an XGBoost model for wildfire prediction.
        """
        self.model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5)
        self.model.fit(self.X_train, self.y_train)
        print("✅ Model trained successfully!")
    
    def evaluate_model(self):
        """
        Evaluates the trained model.
        """
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"🔥 Model Accuracy: {accuracy:.2f}")
