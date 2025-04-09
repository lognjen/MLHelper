import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class Visualization:
    """Helper functions for data visualization tasks."""
    
    @staticmethod
    def plot_feature_importance(model, feature_names, top_n=20):
        """
        Plot feature importance for a model.
        
        Parameters:
        -----------
        model : estimator
            Trained model with feature_importances_ attribute
        feature_names : list
            List of feature names
        top_n : int, default=20
            Number of top features to display
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        # Check if model has feature_importances_ attribute
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_ attribute")
        
        # Get feature importance
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Limit to top_n features
        n_features = min(top_n, len(feature_names))
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance')
        plt.bar(range(n_features), importances[indices[:n_features]], align='center')
        plt.xticks(range(n_features), [feature_names[i] for i in indices[:n_features]], rotation=90)
        plt.xlim([-1, n_features])
        plt.tight_layout()
        plt.show()
        
        return plt.gcf()
    
    @staticmethod
    def plot_correlation_matrix(df, method='pearson', annot=True, cmap='coolwarm'):
        """
        Plot correlation matrix for DataFrame columns.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame
        method : str, default='pearson'
            Correlation method: 'pearson', 'kendall', 'spearman'
        annot : bool, default=True
            Whether to annotate the heatmap
        cmap : str, default='coolwarm'
            Colormap for the heatmap
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        # Calculate correlation matrix
        corr = df.corr(method=method)
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=annot, cmap=cmap, vmin=-1, vmax=1, linewidths=0.5)
        plt.title(f'Correlation Matrix ({method.capitalize()})')
        plt.tight_layout()
        plt.show()
        
        return plt.gcf()
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=False, cmap='Blues'):
        """
        Plot confusion matrix.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        class_names : list, default=None
            List of class names
        normalize : bool, default=False
            Whether to normalize the confusion matrix
        cmap : str, default='Blues'
            Colormap for the heatmap
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, cbar=False,
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
        return plt.gcf()
    
    @staticmethod
    def plot_distribution(df, columns=None, figsize=(15, 10), bins=30):
        """
        Plot distributions of DataFrame columns.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame
        columns : list, default=None
            List of columns to plot
        figsize : tuple, default=(15, 10)
            Figure size
        bins : int, default=30
            Number of bins for histograms
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        if columns is None:
            columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        n_cols = len(columns)
        n_rows = (n_cols + 2) // 3  # Ceiling division
        
        fig, axes = plt.subplots(n_rows, min(n_cols, 3), figsize=figsize)
        axes = axes.flatten() if n_cols > 1 else [axes]
        
        for i, col in enumerate(columns):
            sns.histplot(df[col], kde=True, bins=bins, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return fig
