import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    """Helper functions for data preprocessing tasks."""
    
    @staticmethod
    def handle_missing_values(df, strategy='mean', categorical_cols=None, numerical_cols=None):
        """
        Handle missing values in a DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame with missing values
        strategy : str, default='mean'
            Strategy for numerical columns: 'mean', 'median', 'most_frequent', 'constant'
        categorical_cols : list, default=None
            List of categorical column names
        numerical_cols : list, default=None
            List of numerical column names
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with imputed values
        """
        df_copy = df.copy()
        
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
        if numerical_cols is None:
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Handle numerical columns
        if numerical_cols:
            num_imputer = SimpleImputer(strategy=strategy)
            df_copy[numerical_cols] = num_imputer.fit_transform(df_copy[numerical_cols])
        
        # Handle categorical columns
        if categorical_cols:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df_copy[categorical_cols] = cat_imputer.fit_transform(df_copy[categorical_cols])
            
        return df_copy
    
    @staticmethod
    def encode_categorical_features(df, columns=None, method='onehot', drop_first=True):
        """
        Encode categorical features in a DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame
        columns : list, default=None
            List of categorical column names to encode
        method : str, default='onehot'
            Encoding method: 'onehot' or 'label'
        drop_first : bool, default=True
            Whether to drop the first category in onehot encoding
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with encoded categorical features
        """
        df_copy = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if method == 'onehot':
            df_encoded = pd.get_dummies(df_copy, columns=columns, drop_first=drop_first)
            return df_encoded
        
        elif method == 'label':
            for col in columns:
                df_copy[col] = df_copy[col].astype('category').cat.codes
            return df_copy
        
        else:
            raise ValueError("Method must be either 'onehot' or 'label'")
    
    @staticmethod
    def scale_features(df, columns=None, method='standard'):
        """
        Scale numerical features in a DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame
        columns : list, default=None
            List of numerical column names to scale
        method : str, default='standard'
            Scaling method: 'standard' or 'minmax'
            
        Returns:
        --------
        pandas.DataFrame, scaler
            DataFrame with scaled features and the fitted scaler
        """
        df_copy = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Method must be either 'standard' or 'minmax'")
        
        df_copy[columns] = scaler.fit_transform(df_copy[columns])
        
        return df_copy, scaler
    
    @staticmethod
    def remove_outliers(df, columns=None, method='iqr', threshold=1.5):
        """
        Remove outliers from a DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame
        columns : list, default=None
            List of numerical column names to check for outliers
        method : str, default='iqr'
            Method to detect outliers: 'iqr' or 'zscore'
        threshold : float, default=1.5
            Threshold for outlier detection
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with outliers removed
        """
        df_copy = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if method == 'iqr':
            for col in columns:
                Q1 = df_copy[col].quantile(0.25)
                Q3 = df_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_copy = df_copy[(df_copy[col] >= lower_bound) & (df_copy[col] <= upper_bound)]
                
        elif method == 'zscore':
            for col in columns:
                z_scores = np.abs((df_copy[col] - df_copy[col].mean()) / df_copy[col].std())
                df_copy = df_copy[z_scores < threshold]
                
        else:
            raise ValueError("Method must be either 'iqr' or 'zscore'")
            
        return df_copy
