import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

def perform_regression(aggregated_genre_df, umbrella_genre, dep_var):
    """
    Performs a regression of the desired columns of the DataFrame for a given umbrella genre on average inflation adjusted revenue
    
    Input:
        aggregated_genre_df: pd.DataFrame: DataFRame with the desired date
        umbrella_genre: str: desired genre
        
    Output:
        (model, genre_data): tuple, with the fitted model and the subset of our DataFrame with scaled features
    """
    # Extract the specific genre
    genre_data = aggregated_genre_df.loc[umbrella_genre]

    # Selecting columns for regression
    selected_columns = [
        'Movie runtime', 'Month', 'Female Percentage',
        'Number of ethnicities', 'Number of languages', 'Unemployment',
        dep_var, 'Typecasting', 'Actor Popularity'
    ]
    
    genre_data = genre_data[selected_columns]

    # Drop rows with missing values
    genre_data.dropna(subset=selected_columns, inplace=True)

    # One-hot encode the 'Month' column, removing January (our base column) to avoid multicollinearity in the regression
    genre_data = pd.get_dummies(genre_data, columns=['Month',], prefix='Month',dtype=int)
    genre_data = genre_data.drop('Month_1.0', axis=1)

    # minmax normalization for features and target
    scaler = MinMaxScaler()
    features_to_scale = [
        'Movie runtime', 'Female Percentage',
        'Number of ethnicities', 'Number of languages', 'Unemployment',
        dep_var, 'Typecasting', 'Actor Popularity'
    ]
    genre_data[features_to_scale] = scaler.fit_transform(genre_data[features_to_scale])

    formula = f"Q('{dep_var}') ~ " + \
              " + ".join([f"Q('{col}')" for col in genre_data.columns if col != dep_var])


    model = smf.ols(formula=formula, data=genre_data).fit()

    return model,genre_data
#usage:
# ols_model,genre_data = perform_regression_revenue(aggregated_genre_df, 'Action', dep_var)

def visualize_individual_effects(genre_data, dep_var, line=False):
    """
    Helper function to visualize individual effects of each column on the dependent variable, using residuals.

    Input:
        genre_data: pd.DataFrame: Input DataFrame containing both independent and dependent variables.
        dep_var: str: The name of the column representing the dependent variable.

    Output:
        None
    """
    X_columns = genre_data.drop(columns=[dep_var]).columns
    X = genre_data.drop(columns=[dep_var]).values
    y = genre_data[dep_var].values

    for i in range(0, X.shape[1]):
        partial_model = sm.OLS(X[:, i], np.delete(X, i, axis=1)).fit()
        residual_X = X[:, i] - partial_model.predict(np.delete(X, i, axis=1))

        partial_model_y = sm.OLS(y, np.delete(X, i, axis=1)).fit()
        residual_y = y - partial_model_y.predict(np.delete(X, i, axis=1))

        # Scatter plot of the residuals
        plt.figure(figsize=(6, 4))
        plt.scatter(residual_X, residual_y)
        plt.title(f'Effect of Feature {X_columns[i]}')
        plt.xlabel(f'Feature {X_columns[i]} (residualized)')
        plt.ylabel(f'Target {dep_var} (residualized)')
        if line:
            # Fit a linear regression line through the scatter plot
            line_params = np.polyfit(residual_X, residual_y, 1)
            line_x = np.linspace(min(residual_X), max(residual_X), 100)
            line_y = np.polyval(line_params, line_x)
            plt.plot(line_x, line_y, color='red', label='Regression Line')
            plt.legend()
        plt.show()

def correlation_matrix(aggregated_genre_df, desired_dep_variables, indep_var, umbrella_genre):
    genre_data = aggregated_genre_df.loc[umbrella_genre]
    # include dependent variable as first one in the matrix
    desired_columns = [indep_var] + desired_dep_variables
    genre_data = genre_data[desired_columns]

    # Drop rows with missing values
    genre_data.dropna(subset=desired_columns, inplace=True)
    
    correlation_matrix = genre_data.corr()
    
    sns.heatmap(correlation_matrix, annot=True)
    plt.title(f"Correlation matrix for {indep_var} and dependent variables for genre {umbrella_genre}")
    plt.show()
