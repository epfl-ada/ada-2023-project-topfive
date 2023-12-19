import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go

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

def interactive_correlation_matrix(aggregated_genre_df, desired_dep_variables, indep_var, categories):

    """
    Creates an interactive Plotly heatmap displaying the correlation matrix for different categories.
    
    Parameters:
    aggregated_genre_df (DataFrame): A DataFrame indexed by category (genre) with columns for the variables of interest.
    desired_dep_variables (list of str): A list of strings representing the dependent variables to be included in the correlation matrix.
    indep_var (str): The independent variable whose correlation with the dependent variables is to be analyzed.
    categories (list of str): A list of categories (genres) for which the correlation matrix is to be created.

    Returns:
    plotly.graph_objs._figure.Figure: A Plotly Figure object containing the interactive heatmaps.

    """

    fig = go.Figure()

    for umbrella_genre in categories:
        genre_data = aggregated_genre_df.loc[umbrella_genre]
        desired_columns = [indep_var] + desired_dep_variables
        genre_data = genre_data[desired_columns]

        genre_data.dropna(subset=desired_columns, inplace=True)
        correlation_matrix = genre_data.corr()

        heatmap = go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='magma',
                visible=False,
                name=umbrella_genre,
                text=correlation_matrix.values.round(2),  # Display rounded correlation values
                texttemplate="%{text}",  # Use texttemplate for displaying text values
            )

        fig.add_trace(heatmap)

    # Update layout to add and move dropdown
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    dict(label=genre,
                         method='update',
                         args=[{'visible': [genre == g for g in categories]},
                               {'title': f"Correlation Matrix for {genre}"}])
                    for genre in categories
                ],
                direction="down",
                showactive=True,
                x=0.8,  # Adjust x position
                y=1.25,  # Adjust y position
                xanchor='center',
                yanchor='top'
            )
        ],
        title="Select a Genre to View its Correlation Matrix",
    )

    fig.data[0].visible = True

    return fig

def interactive_residuals_scatterplot(genre_data, dep_var, line=False):
    """
    Generates an interactive Plotly scatter plot of residuals for each feature in the dataset.

    Parameters:
    genre_data (pd.DataFrame): [Description]
    dep_var (str): [Description]
    line (bool): [Description, defaults to False]

    Returns:
    plotly.graph_objs._figure.Figure: [Description]
    """
    X_columns = genre_data.drop(columns=[dep_var]).columns
    X = genre_data.drop(columns=[dep_var]).values
    y = genre_data[dep_var].values

    fig = go.Figure()

    for i, feature in enumerate(X_columns):
        partial_model = sm.OLS(X[:, i], np.delete(X, i, axis=1)).fit()
        residual_X = X[:, i] - partial_model.predict(np.delete(X, i, axis=1))

        partial_model_y = sm.OLS(y, np.delete(X, i, axis=1)).fit()
        residual_y = y - partial_model_y.predict(np.delete(X, i, axis=1))

        # Add the scatter plot for residuals
        fig.add_trace(go.Scatter(
            x=residual_X, 
            y=residual_y, 
            mode='markers', 
            name=f'Feature {feature}',
            visible=(i == 0)  # Only the first scatter plot is visible
        ))

        if line:
            # Fit a regression line through the scatter plot
            line_params = np.polyfit(residual_X, residual_y, 1)
            line_x = np.linspace(min(residual_X), max(residual_X), 100)
            line_y = np.polyval(line_params, line_x)

            # Add the regression line
            fig.add_trace(go.Scatter(
                x=line_x, 
                y=line_y, 
                mode='lines', 
                name='Regression Line',
                visible=(i == 0),  # Only the first regression line is visible
                line=dict(color='red')
            ))

    # Create dropdown buttons for feature selection
    buttons = []
    for i, feature in enumerate(X_columns):
        visibility = [False] * (len(X_columns) * (2 if line else 1))  # Initialize all to false
        visibility[i * (2 if line else 1)] = True  # Toggle scatter plot
        if line:
            visibility[i * 2 + 1] = True  # Toggle regression line

        button = dict(
            label=f'Feature {feature}',
            method='update',
            args=[{'visible': visibility},
                  {'title': f'Effect of Feature {feature}'}]
        )
        buttons.append(button)

    # Update layout to add dropdown
    fig.update_layout(
        updatemenus=[{
            'buttons': buttons,
            'direction': 'down',
            'showactive': True,
            'x': 1,  # x = 0.5 positions the button in the center of the graph horizontally
            'y': 1.2,  # y > 1 positions the button above the top of the graph
            'xanchor': 'center',  # 'center' ensures that the middle of the button aligns with x position
            'yanchor': 'top'  # 'top' ensures the button aligns above the graph based on the y position

        }],
        title=f"Effect of Feature on {dep_var}",
        xaxis_title="Feature (residualized)",
        yaxis_title=f"{dep_var} (residualized)"
    )

    return fig