import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


class ModelEvaluation:
    """Helper functions for model evaluation tasks."""
    
    @staticmethod
    def evaluate_classification_model(model, X_test, y_test, class_names=None):
        """
        Evaluate a classification model and print metrics.
        
        Parameters:
        -----------
        model : estimator
            Trained classification model
        X_test : array-like
            Test features
        y_test : array-like
            True labels
        class_names : list, default=None
            List of class names
            
        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        # Print results
        print("Confusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        # Return metrics
        return {
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'accuracy': class_report['accuracy']
        }
    
    @staticmethod
    def evaluate_regression_model(model, X_test, y_test):
        """
        Evaluate a regression model and print metrics.
        
        Parameters:
        -----------
        model : estimator
            Trained regression model
        X_test : array-like
            Test features
        y_test : array-like
            True values
            
        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Print results
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        # Return metrics
        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
    
    @staticmethod
    def cross_validate_model(model, X, y, cv=5, scoring=None):
        """
        Perform cross-validation on a model.
        
        Parameters:
        -----------
        model : estimator
            Model to cross-validate
        X : array-like
            Features
        y : array-like
            Target variable
        cv : int, default=5
            Number of cross-validation folds
        scoring : str or list, default=None
            Scoring metric(s) to evaluate
            
        Returns:
        --------
        dict
            Dictionary of cross-validation scores
        """
        if isinstance(scoring, list):
            cv_results = {}
            for score in scoring:
                cv_scores = cross_val_score(model, X, y, cv=cv, scoring=score)
                cv_results[score] = {
                    'scores': cv_scores,
                    'mean': cv_scores.mean(),
                    'std': cv_scores.std()
                }
                print(f"{score}: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            return cv_results
        else:
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            print(f"Cross-validation score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            return {
                'scores': cv_scores,
                'mean': cv_scores.mean(),
                'std': cv_scores.std()
            }
    
    @staticmethod
    def plot_learning_curve(model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5)):
        """
        Plot the learning curve of a model.
        
        Parameters:
        -----------
        model : estimator
            Model to evaluate
        X : array-like
            Features
        y : array-like
            Target variable
        cv : int, default=5
            Number of cross-validation folds
        train_sizes : array-like, default=np.linspace(0.1, 1.0, 5)
            Points on the training size
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        from sklearn.model_selection import learning_curve
        
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=cv, train_sizes=train_sizes, scoring='accuracy'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.title('Learning Curve')
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.grid()
        
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='orange')
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        plt.plot(train_sizes, test_mean, 'o-', color='orange', label='Cross-validation Score')
        plt.legend(loc='best')
        
        plt.tight_layout()
        plt.show()
        
        return plt.gcf()
