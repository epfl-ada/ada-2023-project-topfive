import pandas as pd
import statsmodels.formula.api as smf
from sklearn.preprocessing import MinMaxScaler

def perform_regression_revenue(aggregated_genre_df, umbrella_genre):
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
        'Inf adj movie box office revenue', 'Typecasting', 'Actor Popularity'
    ]
    
    genre_data = genre_data[selected_columns]

    # Drop rows with missing values
    genre_data.dropna(subset=selected_columns, inplace=True)

    # One-hot encode the 'Month' column, removing January (our base column) to avoid multicollinearity in the regression
    genre_data = pd.get_dummies(genre_data, columns=['Month',], prefix='Month',dtype=int)
    genre_data = genre_data.drop('1', axis=columns)

    # minmax normalization for features and target
    scaler = MinMaxScaler()
    features_to_scale = [
        'Movie runtime', 'Female Percentage',
        'Number of ethnicities', 'Number of languages', 'Unemployment',
        'Inf adj movie box office revenue', 'Typecasting', 'Actor Popularity'
    ]
    genre_data[features_to_scale] = scaler.fit_transform(genre_data[features_to_scale])

    formula = "Q('Inf adj movie box office revenue') ~ " + \
              " + ".join([f"Q('{col}')" for col in genre_data.columns if col != 'Inf adj movie box office revenue'])


    model = smf.ols(formula=formula, data=genre_data).fit()

    return model,genre_data
#usage:
# ols_model,genre_data = perform_regression_revenue(aggregated_genre_df, 'Action')

def perform_regression_rating(aggregated_genre_df, umbrella_genre):
    """
    Performs a regression of the desired columns of the DataFrame for a given umbrella genre on average IMDB ratings
    
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
        'averageRating', 'Typecasting', 'Actor Popularity'
    ]
    genre_data = genre_data[selected_columns]

    # Drop rows with missing values
    genre_data.dropna(subset=selected_columns, inplace=True)

    # One-hot encode the 'Month' column, removing January (our base column) to avoid multicollinearity in the regression
    genre_data = pd.get_dummies(genre_data, columns=['Month',], prefix='Month',dtype=int)
    genre_data = genre_data.drop('1', axis=columns)

    # minmax normalization for features and target
    scaler = MinMaxScaler()
    features_to_scale = [
        'Movie runtime', 'Female Percentage',
        'Number of ethnicities', 'Number of languages', 'Unemployment',
        'averageRating', 'Typecasting', 'Actor Popularity'
    ]
    genre_data[features_to_scale] = scaler.fit_transform(genre_data[features_to_scale])

    formula = "Q('averageRating') ~ " + \
              " + ".join([f"Q('{col}')" for col in genre_data.columns if col != 'averageRating'])


    model = smf.ols(formula=formula, data=genre_data).fit()

    return model,genre_data

# ols_model,genre_data = perform_regression_rating(aggregated_genre_df, 'Drama')