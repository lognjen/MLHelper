import pandas as pd


class FeatureEngineering:
    """Helper functions for feature engineering tasks."""
    
    @staticmethod
    def create_polynomial_features(df, columns, degree=2):
        """
        Create polynomial features from selected columns.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame
        columns : list
            List of column names to create polynomial features
        degree : int, default=2
            Degree of polynomial features
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added polynomial features
        """
        df_copy = df.copy()
        
        for col in columns:
            for d in range(2, degree + 1):
                df_copy[f'{col}^{d}'] = df_copy[col] ** d
                
        return df_copy
    
    @staticmethod
    def create_interaction_terms(df, columns):
        """
        Create interaction terms between pairs of columns.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame
        columns : list
            List of column names to create interaction terms
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added interaction terms
        """
        df_copy = df.copy()
        
        for i in range(len(columns)):
            for j in range(i+1, len(columns)):
                col1 = columns[i]
                col2 = columns[j]
                df_copy[f'{col1}_{col2}_interaction'] = df_copy[col1] * df_copy[col2]
                
        return df_copy
    
    @staticmethod
    def bin_numerical_feature(df, column, bins=5, labels=None, strategy='quantile'):
        """
        Bin a numerical feature into categories.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame
        column : str
            Column name to bin
        bins : int or list, default=5
            Number of bins or list of bin edges
        labels : list, default=None
            Labels for the bins
        strategy : str, default='quantile'
            Binning strategy: 'quantile', 'uniform', or 'custom'
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added binned column
        """
        df_copy = df.copy()
        
        if strategy == 'quantile':
            df_copy[f'{column}_binned'] = pd.qcut(df_copy[column], q=bins, labels=labels)
        elif strategy == 'uniform':
            df_copy[f'{column}_binned'] = pd.cut(df_copy[column], bins=bins, labels=labels)
        elif strategy == 'custom' and isinstance(bins, list):
            df_copy[f'{column}_binned'] = pd.cut(df_copy[column], bins=bins, labels=labels)
        else:
            raise ValueError("Strategy must be 'quantile', 'uniform', or 'custom'")
            
        return df_copy
    
    @staticmethod
    def extract_datetime_features(df, column):
        """
        Extract features from a datetime column.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame
        column : str
            Datetime column name
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added datetime features
        """
        df_copy = df.copy()
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df_copy[column]):
            df_copy[column] = pd.to_datetime(df_copy[column], errors='coerce')
        
        # Extract datetime components
        df_copy[f'{column}_year'] = df_copy[column].dt.year
        df_copy[f'{column}_month'] = df_copy[column].dt.month
        df_copy[f'{column}_day'] = df_copy[column].dt.day
        df_copy[f'{column}_hour'] = df_copy[column].dt.hour
        df_copy[f'{column}_dayofweek'] = df_copy[column].dt.dayofweek
        df_copy[f'{column}_quarter'] = df_copy[column].dt.quarter
        df_copy[f'{column}_is_weekend'] = df_copy[column].dt.dayofweek >= 5
        
        return df_copy
