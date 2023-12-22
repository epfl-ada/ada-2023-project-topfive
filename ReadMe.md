# Navigating Through Genres: Uncovering the Key Factors Behind Box Office Revenue throughout Movie Categories

The wonderful story, Navigating Through Genres: Uncovering the Key Factors Behind Box Office Revenue throughout Movie Categories, can be found [here](https://alexander-schaller.github.io/ada-2023-project-topfive/data-story/).

## Abstract  

This research endeavor is focused on exploring and identifying the primary determinants that influence a movie's financial success at the box office across different cinematic categories. Driven by the assumption that elements such as the timing of a movie's release, the popularity of its actors or the macro-economic climate (among others) have a unique impact on revenue generation within specific genres, our goal is to thoroughly analyze these aspects in a modular fashion. By doing so, we will provide a detailed assessment of the extent to which each factor plays a role in the commercial triumph or shortfall of films. Our methodical approach is designed to illuminate the most influential factors for each film genre and to determine whether these factors have a consistent effect across the board or if their influence is significantly different from one genre to another. The narrative we aspire to craft through this analysis aims to offer valuable insights that could potentially transform industry practices regarding the production and marketing strategies tailored to maximize box office revenue generation within targeted film categories.

## Research Questions
After discussing several approaches to tackle our data story, we identified the following research questions as most relevant.

- How does seasonality affect box office revenues across different movie genres? How could one optimize release date for a given type of movie?
- What is the optimal movie length per genre to maximize box office revenue?
- Does the number of languages a movie is translated into correlate with its box office success? Should one expand its audience reach?
- How does an actor's popularity and its typecasting influence revenues in their respective genres?
- Does a film's financial performance benefit from having a diverse cast in terms of ethnicity and gender? 
- Does the timing of a movie's release amid periods of economic downturn (high unemployment) affect its box office earnings?
- How do overlapping release dates within a single genre affect the box office returns of movies?

## Used Datasets
- [**IMDb Dataset**](https://datasets.imdbws.com): Data on movie ratings and the number of votes from IMDb. These movie ratings will serve as an indirect measure of film criticism and audience perception. We will use it as a feature in determining movie's box office profitability. 
- [**US BLS Data**](https://data.bls.gov/timeseries/LNS14000000): Data on unemployment rates from the US Bureau of Labor Statistics. This variable will be used as a proxy for the global economic environment, which could have a correlation with the success of movie box office returns. 
- [**US CPI Data**](https://www.usinflationcalculator.com/inflation/consumer-price-index-and-annual-percent-changes-from-1913-to-2008/): Inflation rate data from the US Inflation Calculator will be utilized to adjust box office revenues to real terms, ensuring that the dollar values are comparable over time and reflect the actual purchasing power of the period. 


## Methods

### Regression Analysis: 

Our analysis employs Ordinary Least Squares Regression (OLS) to examine the linear dependencies between a mix of numerical and categorical predictors and our main dependent variable of interest, the Consumer Price Index (CPI) Adjusted Gross Movie Box Office Revenue.

### Time-Series Analysis: 

This method is utilized to capture trends over time, focusing particularly on the evolution of actors' careers, fluctuations in economic stability, and the timing of movie release dates. We aim to statistically discern both seasonal patterns and overarching trends to pinpoint the ideal moment for a genre-specific release schedule.

### Clustering:

This technique will be applied to discern and illustrate the disparities among categories throughout our analyses. Opting for clustering is advantageous in our context because it facilitates the visual representation of these differences, which is integral to the narrative of our data story.

### Hypothesis Testing:

Statistical hypothesis testing will serve as the foundational framework for formally addressing research questions that are amenable to such methodology. We will use a statistical significance level of 95%.


## Proposed Timeline and Internal Milestones

Given the current progress we have achieved 

### Week 1: 

Given that our data preprocessing and feature engineering are complete, we organize our data story execution into modules, dividing it among the various research questions and their corresponding models.

### Week 2-3: 

Execute all models and conduct the necessary statistical analyses to compile an initial set of results. At this stage the work will still be split in subproblems. Emphasis will be put in analytical correctness and code quality.  

### Week 4:

Assemble the results and aggregate the individual analyses to form an initial version of the data narrative. During this phase, we will also prioritize code refinement and enforce standard formatting.

### Week 5:

Concentrate on enhancing the textual content and honing the visual elements. Prepare to publish our completed data story on GitHub, poised for the final submission.


## Team Organization
- **Alexander**: Responsible for analyzing optimal movie length and strategizing release dates across genres.
- **Alp**: Focused on assessing the influence of gender/ethnic diversity and the effects of concurrent release dates.
- **Lucie**: Handling the investigation into actor typecasting/popularity.
- **Maxim**: Examining the effects of a film's release timing during economic downturns and the impact of movie language distribution.
- **Tuomas**: Dedicated to evaluating the the implications of title length and sentiment.

## Final Work Distribution

- **Alexander**: Worked on getting the data story website working, fixed template to match desired theming
- **Alp**: Analyzed the effects of diversity on movie success, produced regressions
- **Lucie**: Worked on the final analysis of genres, wrote texts for data story
- **Maxim**: Examined the effects of a film's release timing during economic downturns, produced interactive graphs for the website, wrote texts for data story
- **Tuomas**: Refactored code, wrote texts for data story, produced regressions

