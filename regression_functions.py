import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
import plotly.express as px
import scipy.stats as stats

def perform_regression(aggregated_genre_df, umbrella_genre, dep_var, selected_features):
    """
    Performs a regression of the desired columns of the DataFrame for a given umbrella genre on average inflation adjusted revenue
    
    Input:
        aggregated_genre_df (pd.DataFrame): DataFRame with the desired date
        umbrella_genre (str): desired genre
        
    Output:
        (model, genre_data): tuple, with the fitted model and the subset of our DataFrame with scaled features
    """
    # Extract the specific genre
    genre_data = aggregated_genre_df.loc[umbrella_genre]
    
    selected_columns = selected_features + [dep_var]
    
    genre_data = genre_data[selected_columns]

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

#usage:
# ols_model,genre_data = perform_regression_revenue(aggregated_genre_df, 'Action', dep_var)

def interactive_correlation_matrix(aggregated_genre_df, desired_dep_variables, indep_var, categories):

    """
    Creates an interactive Plotly heatmap displaying the correlation matrix for different categories.
    
    Parameters:
    aggregated_genre_df (DataFrame): A DataFrame indexed by genre with columns for the variables of interest.
    desired_dep_variables (list[str]): A list of strings representing the dependent variables to be included in the correlation matrix.
    indep_var (str): The independent variable whose correlation with the dependent variables is to be analyzed.
    categories (list[str]): A list of categories (genres) for which the correlation matrix is to be created.

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
                text=correlation_matrix.values.round(2),
                texttemplate="%{text}",
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
                x=1.1,
                y=1.25,
                xanchor='center',
                yanchor='top'
            )
        ],
        xaxis_title='',
        yaxis_title='',
        font_size=8,
        title="Select a Genre to View its Correlation Matrix",
    )

    fig.data[0].visible = True

    return fig

def interactive_scatterplot(genre_data, var_x, var_y, desired_genres):
    """
    Generates an interactive Plotly scatter plot of one feature against another in the dataset.

    Parameters:
    genre_data (pd.DataFrame): data used for scatter plots (aggregated_genre_df), needs to have genres as index
    var_x (str): variable on x-axis
    var_y (str): variable on y-axis
    desired_genres (list[str]): list of desired genres we want to see the effect on

    Returns:
    plotly.graph_objs._figure.Figure: A Plotly Figure object containing the scatter plot
    """
    fig = go.Figure()

    for genre in desired_genres:
        df_genre = genre_data.loc[genre]
        X = df_genre[var_x]
        Y = df_genre[var_y]
        
        fig.add_trace(go.Scatter(
            x=X, 
            y=Y, 
            mode='markers', 
            name=f'Genre {genre}',
            visible=True  # Only the first one is visible
        ))

    fig.update_layout(
        title=f"Scatter plot of {var_x} vs {var_y}",
        title_font=dict(size=18),
        xaxis_title=f"{var_x}",
        yaxis_title=f"{var_y}"
    )

    return fig

def interactive_barplot(genre_data, var_x, var_y, desired_genres):
    """
    Generates an interactive Plotly bar plot of one feature against another in the dataset.

    Parameters:
    genre_data (pd.DataFrame): data used for bar plots (aggregated_genre_df), needs to have genres as index
    var_x (str): variable on x-axis
    var_y (str): variable on y-axis
    desired_genres (list(str)): list of desired genres we want to see the effect on

    Returns:
    plotly.graph_objs._figure.Figure: A Plotly Figure object containing the bar plot
    """
    fig = go.Figure()

    for genre in desired_genres:
        df_genre = genre_data.loc[genre_data['Umbrella Genre'] == genre]
        X = df_genre[var_x]
        Y = df_genre[var_y]

        fig.add_trace(go.Bar(
            x=X, 
            y=Y, 
            name=f'Genre {genre}',
            visible=False  # Only the first one is visible
        ))
    
    df_genre = genre_data.loc[genre_data['Umbrella Genre'].isin(desired_genres)]
    X = df_genre[var_x]
    Y = df_genre[var_y]
    # Add the scatter plot
    fig.add_trace(go.Bar(
        x=X, 
        y=Y,
        name=f'Aggregate of desired genres',
        visible=True  # Only the aggregate bar plot is visible
    ))
    
    buttons = []
    for i, genre in enumerate(desired_genres):
        # Initialize all to false
        visibility = [False] * (len(desired_genres) + 1)
        # Toggle scatter plot for each i
        visibility[i] = True

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
            'x': 1,
            'y': 1.2,
            'xanchor': 'center',
            'yanchor': 'top'

        }],
        title=f"Bar plot of {var_x} vs {var_y}",
        xaxis_title=f"{var_x}",
        yaxis_title=f"{var_y}"
    )

    return fig

def interactive_residuals_scatterplot(regression_data, dep_var, indep_var, desired_genres, line=False):
    """
    Generates an interactive Plotly scatter plot of residuals for a chosen feature in the dataset.

    Parameters:
    regression_data (pd.DataFrame): data used for the residual regressions
    dep_var (str): dependent variable on which we want to study the effect
    indep_var (str): independent variable which effect we want to study
    desired_genres (list(str)): list of desired genres we want to see the effect on
    line (bool): toggle the line going through the mean of the residuals

    Returns:
    plotly.graph_objs._figure.Figure: A Plotly Figure containing the scatter plot of the residuals (and potentially the line)
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
            visible=(genre == 'Drama')  # Only the first one is visible
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
                visible=(genre == 'Drama'),
                line=dict(color='red')
            ))

    # Dropdown buttons for feature selection
    buttons = []
    for i, genre in enumerate(desired_genres):
        # Initialize all to false
        visibility = [False] * (len(desired_genres) * (2 if line else 1))
        # Toggle scatter plot
        visibility[i * (2 if line else 1)] = True  
        if line:
            # Toggle regression line
            visibility[i * 2 + 1] = True

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
            'x': 1.2,
            'y': 0.2,
            'xanchor': 'center',
            'yanchor': 'top'

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
                x=1,
                y=1.15,
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
    Creates an interactive Plotly bar plot displaying the average box office revenue for different intervals of a variable

    Parameters:
    regression_df_revenue (DataFrame): DataFrame containing the data for analysis
    var_to_plot (str): The variable for which intervals are considered
    intervals (list): List of intervals to categorize the variable
    categories (list of str): A list of categories (genres) for which the plot is to be created

    Returns:
    plotly.graph_objs._figure.Figure: A Plotly Figure object containing the interactive bar plot
    """

    fig = go.Figure()

    for category in categories:
        category_data = regression_data[regression_data['Umbrella Genre'] == category]

        # Bin variable into intervals
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
    Creates an interactive Histogram2dContour graph with dropdown buttons to switch between genres.

    Parameters:
    regression_data (DataFrame): The data containing the variables and genres.
    dep_var (str): The name of the dependent variable column.
    indep_var (str): The name of the independent variable column.
    desired_genres (list of str): A list of genres to filter the data and create plots for.
    
    Returns:
    plotly.graph_objs._figure.Figure: A Plotly Figure object containing the interactive Histogram2dContour.
    """

    fig = go.Figure()

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

        # Marginal histogram for the x-axis
        fig.add_trace(go.Histogram(
            x=genre_data[indep_var],
            name='X marginal',
            marker=dict(color='rgba(220, 0, 0, 0.8)'),
            yaxis='y2'
        ))

        # Marginal histogram for the y-axis
        fig.add_trace(go.Histogram(
            y=genre_data[dep_var],
            name='Y marginal',
            marker=dict(color='rgba(220, 0, 0, 0.8)'),
            xaxis='x2'
        ))

    # Initial view
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

        # Visibility for the genre's traces
        for j in range(len(desired_genres)):
            if genre == desired_genres[j]:
                for k in range(4):
                    button['args'][0]['visible'][j * 4 + k] = True  # Toggle visibility

        buttons.append(button)

    # 4 traces per genre (contour, scatter, x_hist, y_hist)
    for i in range(4):  
        fig.data[i].visible = True

    fig.update_layout(
        title=f"{desired_genres[0]} Genre: {indep_var} vs {dep_var}",
        updatemenus=[{
            'buttons': buttons,
            'direction': 'down',
            'x': 1,
            'xanchor': 'center',
            'y': 1,
            'yanchor': 'top'
        }],
        xaxis2=dict(title=''),
        yaxis2=dict(title='')
    )

    fig['layout']['annotations'] = []

    return fig

def mean_confidence_interval(data, confidence=0.95):
    """
    Calculates the mean and confidence interval for an array of data.

    Parameters:
    data (array-like): The dataset to calculate the mean and CI for.
    confidence (float): The confidence level to compute the CI at.

    Returns:
    tuple: A tuple containing the mean, lower bound of CI, and upper bound of CI.
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def create_interactive_boxplot(data_frame, dep_var, indep_var, desired_genres):
    """
    Creates an interactive box plot with dropdown buttons to switch between genres

    Parameters:
    data_frame (DataFrame): The data containing the variables and genres
    dep_var (str): The name of the dependent variable column
    indep_var (str): The name of the independent variable column
    desired_genres (list of str): A list of genres to filter the data and create plots for
    
    Returns:
    plotly.graph_objs._figure.Figure: A Plotly Figure object containing the interactive box plot
    """

    fig = go.Figure()

    genre_traces = {}

    for genre in desired_genres:
        genre_data = data_frame[data_frame['Umbrella Genre'] == genre]
        
        # List comprehension to construct data for ci_df
        ci_data = [{
            indep_var: lang,
            'Mean': mean,
            'Low CI': low,
            'High CI': high
        } for lang, data in genre_data.groupby(indep_var)[dep_var] for mean, low, high in [mean_confidence_interval(data.values)]]
        
        ci_df = pd.DataFrame(ci_data)

        box_fig = px.box(
            genre_data, 
            x=indep_var, 
            y=dep_var,
            labels={dep_var: 'Log Adjusted Revenue'},
            title=f'Boxplot of {dep_var} by {indep_var}'
        )

        # Add boxplot traces to the figure
        for trace in box_fig.data:
            trace.visible = False
            fig.add_trace(trace)

        # Add CI scatter trace to the figure
        ci_trace = go.Scatter(
            x=ci_df[indep_var],
            y=ci_df['Mean'],
            mode='markers',
            error_y=dict(
                type='data',
                symmetric=False,
                array=ci_df['High CI'] - ci_df['Mean'],
                arrayminus=ci_df['Mean'] - ci_df['Low CI']
            ),
            marker=dict(color='black', size=10),
            name='Mean and 95% CI',
            visible=False
        )
        fig.add_trace(ci_trace)
        # Store the reference to the newly added traces
        genre_traces[genre] = fig.data[-(len(box_fig.data) + 1):]

    # Create dropdown buttons
    buttons = []
    for genre, traces in genre_traces.items():
        button = dict(
            label=genre,
            method="update",
            args=[{"visible": [t in traces for t in fig.data]},
                  {"title": f"{genre} Genre: {dep_var} vs {indep_var}"}]
        )
        buttons.append(button)

    fig.update_layout(
        updatemenus=[{
            'buttons': buttons,
            'direction': 'down',
            'active': 0,
            'x': 1.15,
            'xanchor': 'center',
            'y': 1.25,
            'yanchor': 'top'
        }],
        title=f'Boxplot of {dep_var} by {indep_var}',
        xaxis_title=indep_var,
        yaxis_title=dep_var,
        template='plotly_white'
    )

    initial_genre = desired_genres[0]
    for trace in genre_traces[initial_genre]:
        trace.visible = True

    return fig