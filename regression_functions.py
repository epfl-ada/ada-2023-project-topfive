import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
import scipy.stats as stats

def perform_regression(aggregated_genre_df, umbrella_genre, dep_var, selected_features):
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
    
    selected_columns = selected_features + [dep_var]
    # Selecting columns for regression
    """selected_columns = [
        'Movie runtime', 'Month', 'Female Percentage',
        'Number of ethnicities', 'Number of languages', 'Unemployment',
        dep_var, 'Typecasting', 'Actor Popularity'
    ]"""
    """selected_columns = [
        'Movie runtime', 'Month', 'Female Percentage',
        'Number of ethnicities', 'Number of languages', 'Unemployment',
        dep_var, 'Actor Popularity'
    ]"""
    
    genre_data = genre_data[selected_columns]

    # Drop rows with missing values
    genre_data.dropna(subset=selected_columns, inplace=True)

    # One-hot encode the 'Month' column, removing January (our base column) to avoid multicollinearity in the regression
    genre_data = pd.get_dummies(genre_data, columns=['Month',], prefix='Month',dtype=int)
    genre_data = genre_data.drop('Month_1.0', axis=1)

    formula = f"Q('{dep_var}') ~ " + \
              " + ".join([f"Q('{col}')" for col in genre_data.columns if col != dep_var])
                                                                    
    model = smf.ols(formula=formula, data=genre_data).fit()
    
    dfbetas = model.get_influence().dfbetas
    outliers = np.abs(dfbetas).max(axis=1) > 1
    
    data_without_outliers = genre_data[~outliers]
    
    clean_model = smf.ols(formula=formula, data=data_without_outliers).fit()

    return clean_model,data_without_outliers
    #return model, genre_data
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

def interactive_scatterplot(genre_data, var_x, var_y, desired_genres):
    """
    Generates an interactive Plotly scatter plot of one feature against another in the dataset.

    Parameters:
    genre_data (pd.DataFrame): data used for scatterplots (aggregated_genre_df), needs to have genres within
    var_x (str): variable on x-axis
    var_y (str): variable on y-axis
    desired_genres (list(str)): list of desired genres we want to see the effect on

    Returns:
    plotly.graph_objs._figure.Figure: figure we want to plot
    """
    fig = go.Figure()

    for genre in desired_genres:
        df_genre = genre_data.loc[genre]
        X = df_genre[var_x]
        Y = df_genre[var_y]
        # Add the scatter plot
        fig.add_trace(go.Scatter(
            x=X, 
            y=Y, 
            mode='markers', 
            name=f'Genre {genre}',
            visible=True  # Only the first scatter plot is visible
        ))

    # Update layout to add dropdown
    fig.update_layout(
        title=f"Scatter plot of {var_x} vs {var_y}",
        xaxis_title=f"{var_x}",
        yaxis_title=f"{var_y}"
    )

    return fig

def interactive_barplot(genre_data, var_x, var_y, desired_genres):
    """
    Generates an interactive Plotly bar plot of one feature against another in the dataset.

    Parameters:
    genre_data (pd.DataFrame): data used for bar plots (aggregated_genre_df), needs to have genres within
    var_x (str): variable on x-axis
    var_y (str): variable on y-axis
    desired_genres (list(str)): list of desired genres we want to see the effect on

    Returns:
    plotly.graph_objs._figure.Figure: figure we want to plot
    """
    fig = go.Figure()

    for genre in desired_genres:
        df_genre = genre_data.loc[genre_data['Umbrella Genre'] == genre]
        X = df_genre[var_x]
        Y = df_genre[var_y]
        # Add the scatter plot
        fig.add_trace(go.Bar(
            x=X, 
            y=Y, 
            name=f'Genre {genre}',
            visible=False  # Only the first scatter plot is visible
        ))
    
    df_genre = genre_data.loc[genre_data['Umbrella Genre'].isin(desired_genres)]
    X = df_genre[var_x]
    Y = df_genre[var_y]
    # Add the scatter plot
    fig.add_trace(go.Bar(
        x=X, 
        y=Y,
        name=f'Aggregate of desired genres',
        visible=True  # Only the aggregate scatter plot is visible
    ))
    
    buttons = []
    for i, genre in enumerate(desired_genres):
        visibility = [False] * (len(desired_genres) + 1) # Initialize all to false
        visibility[i] = True  # Toggle scatter plot

        button = dict(
            label=f'Genre {genre}',
            method='update',
            args=[{'visible': visibility},
                  {'title': f'test'}]
        )
        buttons.append(button)

    button = dict(
        label=f'Aggregate of genres',
        method='update',
        args=[{'visible': [False] * len(desired_genres) + [True]},
              {'title': f'test'}]
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
        title=f"Bar plot of {var_x} vs {var_y}",
        xaxis_title=f"{var_x}",
        yaxis_title=f"{var_y}"
    )

    return fig

def interactive_residuals_scatterplot(regression_data, dep_var, indep_var, desired_genres, line=False):
    """
    Generates an interactive Plotly scatter plot of residuals for each feature in the dataset.

    Parameters:
    regression_data (pd.DataFrame): data used for the actual regressions
    dep_var (str): dependent variable on which we want to study the effect
    indep_var (str): independent variable which effect we want to study
    desired_genres (list(str)): list of desired genres we want to see the effect on
    line (bool): toggle the line going through the mean of the residuals

    Returns:
    plotly.graph_objs._figure.Figure: figure of the effects we want to plot
    """
    fig = go.Figure()

    for genre in desired_genres:
        df_regr = regression_data.loc[regression_data['Umbrella Genre'] == genre]
        
        formula_residuals_x = f"Q('{indep_var}') ~ " + \
              " + ".join([f"Q('{col}')" for col in df_regr.columns if col not in [indep_var, dep_var, 'Umbrella Genre']])
    
        formula_residuals_y = f"Q('{dep_var}') ~ " + \
                  " + ".join([f"Q('{col}')" for col in df_regr.columns if col not in [indep_var, dep_var, 'Umbrella Genre']])

        partial_model = smf.ols(formula=formula_residuals_x, data=df_regr).fit()
        df_regr['resid_x'] = df_regr[indep_var] - partial_model.predict(df_regr.drop(columns=[indep_var, dep_var, 'Umbrella Genre']))

        partial_model_y = smf.ols(formula=formula_residuals_y, data=df_regr).fit()
        df_regr['resid_y'] = df_regr[dep_var] - partial_model_y.predict(df_regr.drop(columns=[indep_var, dep_var, 'Umbrella Genre']))

        # Add the scatter plot for residuals
        fig.add_trace(go.Scatter(
            x=df_regr['resid_x'], 
            y=df_regr['resid_y'], 
            mode='markers', 
            name=f'Feature {indep_var}',
            visible=(genre == 'Drama')  # Only the first scatter plot is visible
        ))

        if line:
            # Fit a regression line through the scatter plot
            line_params = np.polyfit(df_regr['resid_x'], df_regr['resid_y'], 1)
            line_x = np.linspace(min(df_regr['resid_x']), max(df_regr['resid_x']), 100)
            line_y = np.polyval(line_params, line_x)

            # Add the regression line
            fig.add_trace(go.Scatter(
                x=line_x, 
                y=line_y, 
                mode='lines', 
                name='Regression Line',
                visible=(genre == 'Drama'),  # Only the first regression line is visible
                line=dict(color='red')
            ))

    # Create dropdown buttons for feature selection
    buttons = []
    for i, genre in enumerate(desired_genres):
        visibility = [False] * (len(desired_genres) * (2 if line else 1))  # Initialize all to false
        visibility[i * (2 if line else 1)] = True  # Toggle scatter plot
        if line:
            visibility[i * 2 + 1] = True  # Toggle regression line

        button = dict(
            label=f'Genre {genre}',
            method='update',
            args=[{'visible': visibility},
                  {'title': f'Effect of Feature {indep_var} on {dep_var} in the genre {genre}'}]
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
        title=f"Effect of Feature {indep_var} on {dep_var}",
        xaxis_title=f"Feature {indep_var} (residualized)",
        yaxis_title=f"{dep_var} (residualized)"
    )
    
    return fig


def interactive_average_vs_years(aggregated_genre_df, var_to_plot, year_begin, year_end, categories):
    """
    Creates an interactive Plotly graph displaying the average of a variable against years for different categories.

    Parameters:
    aggregated_genre_df (DataFrame): A DataFrame indexed by category (genre) with columns for the variables of interest.
    var_to_plot (str): The variable to be plotted against years.
    year_begin (int): The starting year for the analysis.
    year_end (int): The ending year for the analysis.
    categories (list of str): A list of categories (genres) for which the plot is to be created.

    Returns:
    plotly.graph_objs._figure.Figure: A Plotly Figure object containing the interactive graph.
    """

    fig = go.Figure()

    for umbrella_genre in categories:
        genre_data = aggregated_genre_df.loc[umbrella_genre]

        # Filter data based on the specified year range using the 'Year' column
        filtered_data = genre_data.loc[(genre_data['Year'] >= year_begin) & (genre_data['Year'] <= year_end)]

        # Calculate the average of the variable to plot for each year
        average_values = filtered_data.groupby('Year')[var_to_plot].mean()

        scatter = go.Scatter(
            x=average_values.index,
            y=average_values.values,
            mode='lines+markers',
            name=umbrella_genre,
            visible=False,
        )

        fig.add_trace(scatter)

    # Update layout to add and move dropdown
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    dict(label=genre,
                         method='update',
                         args=[{'visible': [genre == g for g in categories]},
                               {'title': f"Average {var_to_plot} vs Years for {genre}"}])
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
        title="Select a Genre to View its Average vs Years",
        xaxis_title="Years",
        yaxis_title=f"Average {var_to_plot}",
    )

    fig.data[0].visible = True

    return fig
# Example usage: rf.interactive_average_vs_years(aggregated_genre_df, 'Female Percentage', 1940, 2010, desired_categories)

def interactive_barplot_average_revenue(regression_data, var_to_plot, intervals, categories):
    """
    Creates an interactive Plotly bar plot displaying the average box office revenue for different intervals of a variable.

    Parameters:
    regression_df_revenue (DataFrame): DataFrame containing the data for analysis.
    var_to_plot (str): The variable for which intervals are considered.
    intervals (list): List of intervals to categorize the variable.
    categories (list of str): A list of categories (genres) for which the plot is to be created.

    Returns:
    plotly.graph_objs._figure.Figure: A Plotly Figure object containing the interactive bar plot.
    """

    fig = go.Figure()

    for category in categories:
        # Filter data for the specified category
        category_data = regression_data[regression_data['Umbrella Genre'] == category]

        # Bin the variable into intervals
        category_data['Intervals'] = pd.cut(category_data[var_to_plot], bins=intervals, labels=[f'{intervals[i]}-{intervals[i+1]}' for i in range(len(intervals)-1)])

        # Calculate mean and 95% CI for each interval
        result = category_data.groupby('Intervals')['Inf adj movie box office revenue'].agg(['mean', 'sem'])
        ci = result['sem'] * stats.t.ppf((1 + 0.95) / 2, category_data.groupby('Intervals').size())

        # Create error bar plot
        bar = go.Bar(
            x=result.index,
            y=result['mean'],
            error_y=dict(type='data', array=ci),
            name=category,
            visible=False,
        )

        fig.add_trace(bar)

    # Update layout to add and move dropdown
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    dict(label=genre,
                         method='update',
                         args=[{'visible': [genre == g for g in categories]},
                               {'title': f'Average Box Office Revenue for Different {var_to_plot} Intervals in {genre} Genre'}])
                    for genre in categories
                ],
                direction="down",
                showactive=True,
                x=0.8,
                y=1.25,
                xanchor='center',
                yanchor='top'
            )
        ],
        title=f'Select a Genre to View its Average Box Office Revenue vs {var_to_plot} Intervals',
        xaxis_title=f'{var_to_plot} Intervals',
        yaxis_title='Average Box Office Revenue'
    )

    fig.data[0].visible = True

    return fig
# Example usage: rf.interactive_barplot_average_revenue(regression_df_revenue, 'Female Percentage', [0, 25, 50, 75, 100], desired_categories)

def interactive_Histogram2dContour(regression_data, dep_var, indep_var, desired_genres):
    """
    Creates an interactive graph with dropdown buttons to switch between genres.

    Parameters:
    regression_data (DataFrame): The data containing the variables and genres.
    dep_var (str): The name of the dependent variable column.
    indep_var (str): The name of the independent variable column.
    desired_genres (list of str): A list of genres to filter the data and create plots for.
    """

    # Initialize figure with subplots
    fig = go.Figure()

    # Create a subplot for each genre and its marginal histograms
    for genre in desired_genres:
        genre_data = regression_data[regression_data['Umbrella Genre'] == genre]

        # Contour plot for the 2D density
        fig.add_trace(go.Histogram2dContour(
            x=genre_data[indep_var],
            y=genre_data[dep_var],
            colorscale='Reds',
            name=genre,
            showscale=False,
            xaxis='x',
            yaxis='y'
        ))

        # Scatter plot for the actual data points
        fig.add_trace(go.Scatter(
            x=genre_data[indep_var],
            y=genre_data[dep_var],
            mode='markers',
            marker=dict(color='rgba(220, 0, 0, 0.8)', size=3),
            name=genre,
            xaxis='x',
            yaxis='y'
        ))

        # Marginal histogram for the x-axis variable
        fig.add_trace(go.Histogram(
            x=genre_data[indep_var],
            name='X marginal',
            marker=dict(color='rgba(220, 0, 0, 0.8)'),
            yaxis='y2'
        ))

        # Marginal histogram for the y-axis variable
        fig.add_trace(go.Histogram(
            y=genre_data[dep_var],
            name='Y marginal',
            marker=dict(color='rgba(220, 0, 0, 0.8)'),
            xaxis='x2'
        ))

    # Update layout for initial view
    fig.update_layout(
        xaxis=dict(domain=[0, 0.85], title=indep_var),
        yaxis=dict(domain=[0, 0.85], title=dep_var),
        xaxis2=dict(domain=[0.85, 1], showticklabels=False),
        yaxis2=dict(domain=[0.85, 1], showticklabels=False),
        height=600,
        width=800,
        hovermode='closest',
        showlegend=False,
        bargap=0.05
    )

    # Add dropdown menu buttons
    buttons = []
    for i, genre in enumerate(desired_genres):
        button = dict(
            args=[{'visible': [False] * len(fig.data)},
                  {'title': f'{genre} Genre: {indep_var} vs {dep_var}'}],
            label=genre,
            method='update'
        )

        # Set the visibility for the genre's traces
        for j in range(len(desired_genres)):
            if genre == desired_genres[j]:
                for k in range(4):
                    button['args'][0]['visible'][j * 4 + k] = True  # Toggle visibility

        buttons.append(button)

    # Set the first genre visible
    for i in range(4):  # Assuming 4 traces per genre (contour, scatter, x_hist, y_hist)
        fig.data[i].visible = True

    # Set initial titles and axis titles
    fig.update_layout(
        title=f"{desired_genres[0]} Genre: {indep_var} vs {dep_var}",
        updatemenus=[{
            'buttons': buttons,
            'direction': 'down',
            'x': 1.1,
            'xanchor': 'center',
            'y': 1.15,
            'yanchor': 'top'
        }]
    )

    # Remove duplicate axis titles
    fig['layout']['annotations'] = []

    # Show the figure
    fig.show()