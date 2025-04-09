from sklearn.model_selection import train_test_split, GridSearchCV


class ModelHelper:
    """Helper functions for model training and tuning."""
    
    @staticmethod
    def train_test_val_split(X, y, test_size=0.2, val_size=0.25, random_state=None):
        """
        Split data into training, validation, and test sets.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target variable
        test_size : float, default=0.2
            Proportion of data for testing
        val_size : float, default=0.25
            Proportion of training data for validation
        random_state : int, default=None
            Random seed for reproducibility
            
        Returns:
        --------
        tuple
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: training + validation and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Second split: training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, random_state=random_state
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    @staticmethod
    def grid_search(model, param_grid, X, y, cv=5, scoring=None, n_jobs=-1):
        """
        Perform grid search for hyperparameter tuning.
        
        Parameters:
        -----------
        model : estimator
            Model to tune
        param_grid : dict
            Dictionary of parameter grids
        X : array-like
            Features
        y : array-like
            Target variable
        cv : int, default=5
            Number of cross-validation folds
        scoring : str, default=None
            Scoring metric
        n_jobs : int, default=-1
            Number of parallel jobs
            
        Returns:
        --------
        GridSearchCV
            Fitted grid search object
        """
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs
        )
        grid_search.fit(X, y)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_:.4f}")
        
        return grid_search
    
    @staticmethod
    def save_model(model, filename):
        """
        Save a trained model to a file.
        
        Parameters:
        -----------
        model : estimator
            Trained model to save
        filename : str
            Filename to save the model
        """
        import joblib
        joblib.dump(model, filename)
        print(f"Model saved to {filename}")
    
    @staticmethod
    def load_model(filename):
        """
        Load a trained model from a file.
        
        Parameters:
        -----------
        filename : str
            Filename to load the model from
            
        Returns:
        --------
        estimator
            Loaded model
        """
        import joblib
        model = joblib.load(filename)
        print(f"Model loaded from {filename}")
        return model
