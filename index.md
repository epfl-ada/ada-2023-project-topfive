---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: page
title: "Home"
---

# Abstract
From fast-paced action to heart-touching romances, each genre has its own recipe for success. With over 2,000 movies produced annually worldwide, what sets each genre apart in the race for box office glory? Let's explore the unique secrets behind the success of different movie genres! Dive into our data journey where numbers meet narratives, uncovering trends and patterns that define cinematic triumphs. Join us to uncover what really makes audiences flock to theaters and cash out!


# Introduction
We will be taking a dive in the backstage of the film industry's greatest hits through the lenses of their ratings and revenue! We will kick off with the spotlight on Release Strategies and Timing where we'll break down the effects of premiere timing, seasonality and overlapping release dates. Next on the reel, we'll be diving into Economic Factors and Box Office Resilience, where we'll explore how movies navigate the economic rollercoaster. Then, it'll be a tour into Casting and Star Power, where we’ll spill the beans on how actor momentum and diversity in the cast might make or break a movie. Finally we’ll be shifting gears to Creative Elements and Film Characteristics, where we’ll focus on the ideal movie length, and how do languages play into the box office spell.

And now...the plot thickens, and so does revenue ? – Let the data tell the tales!

## Explain choice of Umbrella Genres, choice of 7 first ones etc (feature engineering etc)

### Pie Chart with percentages

# Box Office Alchemy: Decoding Causality Across Revenue Drivers
Movies make money for various reasons. In this story, we'll show you how several drivers affect a movie's box office revenues. Think of it like solving a puzzle. In our pursuit of understanding the intricate factors that contribute to a movie's box office success, we primarily employ multiple regression. This choice is not arbitrary but a strategic necessity, rooted in the principles of sound statistical analysis.

Using individual regressions for each of the factors that influence movie revenue may seem like a straightforward approach. However, it's important to highlight a critical pitfall associated with this method: the risk of 'omitted variable bias'. This bias occurs when we neglect to account for relevant variables that may impact the outcome. In simpler terms, it's like trying to solve a puzzle with missing pieces.

To ensure the integrity of our causal analysis, we cannot afford to overlook potential variables. If we were to analyze each factor independently, it could lead to distorted and inaccurate results. More importantly, it would render our causal claims and inferences unreliable.

Instead, our approach involves capturing all 'pure' marginal effects separately within the same comprehensive linear model. This means that we account for all relevant factors simultaneously, allowing us to discern the individual impact of each variable while considering the interplay between them. It's akin to examining each puzzle piece while still seeing how they fit together to form the bigger picture.

But our exploration doesn't end there. To delve deeper into our analysis and gain a better understanding of how these factors interact, we employ a potent technique known as 'partialling out.' This process is made possible through the application of the Frisch-Waugh-Lovell (FWL) Theorem. Think of it as using a magnifying glass to precisely observe how a single puzzle piece influences box office revenues.

Here's how it works: Instead of dealing with a high-dimensional model, we can obtain the same coefficient estimate by conducting two auxiliary regressions. First, we regress the variable of interest on all the other variables, and then we do the same with the independent variable. Afterward, we calculate the residuals for both of these auxiliary regressions. Finally, we regress these two sets of residuals against each other. Remarkably, the coefficient of this last regression mirrors the one in the original multivariable regression, as dictated by the regression anatomy theorem.

The advantage of this approach is that it transforms a complex high-dimensional model into a visually friendly and interpretable 2-dimensional representation. Instead of dealing solely with numerical values, we now have a more intuitive way to grasp and explain the same effect. This enhances the reliability of our causal claims, ensuring a deeper understanding of the factors that underlie box office success.

To improve our regression analysis, we will also transform box office revenues using a logarithmic scale. This adjustment normalizes revenue magnitude, bringing it in line with our predictors for better model accuracy. It also helps in reducing the impact of outliers, ensuring they don't skew our results and accounts for high leverage observations. Despite these changes, the variable remains interpretable, allowing us to understand changes as relative percentages rather than absolute figures.

Last but certainly not least, we must also consider the impact of influential observations in our regression analysis. In this context, one of the most potent statistical tools at our disposal is the 'dfbeta.' To put it simply, the dfbeta provides us with a standardized numerical measure of how much a regression coefficient shifts in terms of standard deviations when a particular observation is excluded from the sample.

If, for a specific observation, the regression coefficient exhibits a movement exceeding 1 standard deviation (in either direction, hence considering the absolute value), that observation is deemed highly influential. In such cases, we make the decision to treat it as an outlier and remove it from the training dataset. This proactive step helps ensure that potential sources of noise in our estimation are minimized.

## Comments about correlation matrices

<iframe src="plots/correlation_matrix_revenue.html" width="750" height="550" frameborder="0">Correlation Matrix for Adj Revenue by Umbrella Genre</iframe>

<iframe src="plots/correlation_matrix_rating.html" width="750" height="550" frameborder="0">Correlation Matrix for Average Review by Umbrella Genre</iframe>

Looking at the correlation matrices for different relationships and different genres, we can see that in a large majority of cases, the correlations between dependent variables and our two independent variable candidates are not that different. For this reason, we decided to proceed with our original analysis and keep the logarithmic inflation adjusted box office as our main success factor.


# Release Strategies and Timing

## Premiere Party: Timing is Everything, Darling!
Choosing the right premiere date is like matchmaking for movies. We're exploring the release dates, their link to the success metrics namely movie ratings and box office revenue.

<iframe src="plots/aggregate_avg_revenue_movie.png" width="750" height="550" frameborder="0">Aggregate Average Revenue of Movies</iframe>

As we inspect the data spotlight  it revealed key insights. July, the summer blockbuster season, and November, anticipating the festive cheer, emerge as the peak months for box office success. On the flip side, January marks a post-holiday slump. The revenue rollercoaster showcases a substantial 3.7 times difference between peak and trough months, underscoring the pivotal role of premiere timing in determining box office triumph. 

The release calendar, shows September and December as the prime blockbuster battlegrounds, while February and July take a nap with the fewest releases. The variation in quantity is significant a 1.5-fold contrast in the number of movies released during high compared to low-release periods.

## Release Rumble: When Movies Go on Awkward First Dates
Ever seen two movies awkwardly run into each other at the release calendar? We're investigating the impact of  colliding release dates. 

<iframe src="plots/aggregate_distribution_movies.png" width="750" height="550" frameborder="0">Aggregate Distribution of Movies</iframe>

Putting on our audience engagement lens (number of votes), June and May steal the limelight, attracting the highest number of votes, while January and October fade into the background. This engagement of spectator translates into around double the interaction during peak months. In this release rumble, where movies collide for attention, the stakes are high, and the audience's verdict resonates. 

After taking a closer look into the average ratings, we notice homogeneity in average rating across all the months. With “top month” December's average rating being 6.66, while July, the “worst month” clocks in at 6.26. This observation begs reflection on the correlation between revenue, ratings, and the number of movies released. The plot thickens: revenue and ratings share a positively consistent bond, but the same cannot be said for the number of movies. The question of co-releasing movies unravels, revealing an insignificant and inconsistent correlation between the number of movies and both revenue and ratings. It seems the co-releasing synergy effect or cannibalization remains an enigma, leaving us to ponder the about the recipe for success. 


# Economic Factors and Box Office Resilience

## Movie Economics: Investigating the Cyclical Nature of Box Office Revenue
US Unemployment rates have long been regarded as a relevant proxy for assessing the overall health of the world's economy. This metric, a reflection of US job market conditions, can be a powerful indicator of global economic well-being. When unemployment rates are low, it often signals a robust job market, increased consumer confidence, and a greater willingness to spend on various goods and services. Conversely, rising unemployment rates may indicate economic challenges, such as reduced consumer spending and business uncertainty.

In the context of the movie industry, our research aims to utilize unemployment rates as a proxy for the macroeconomic environment. We intend to investigate whether there exists a discernible pattern across different movie genres that could potentially shed light on whether the movie industry operates as a cyclical business, or not. Specifically, we want to analyze whether fluctuations in unemployment rates coincide with shifts in box office performance for various movie genres.

To explore our initial question, we delve into the nuanced impact of unemployment rates on box office revenues by utilizing regression analysis. The visual depiction provided by the accompanying two-dimensional causality graph elucidates no discernable relationship across all genres.

<iframe src="plots/unemployment_log_revenue_residual.html" width="750" height="550" frameborder="0">"Partialed out" regression of log(Inflation adjusted revenue) on Unemployment by Umbrella Genre</iframe>

These uniformly non-significant results across all genres except for drama, which shows the only statistically significant marginally negative coefficient, suggest that the movie industry generally operates as a non-cyclical business. This implies that it consistently yields steady returns regardless of the prevailing economic conditions. Notably, the industry appears to be unaffected by variations in disposable income or leisure time, positioning it as a good that is rather inelastic to income shocks. 

<iframe src="plots/h2c_unemployment_log_revenue.html" width="750" height="550" frameborder="0">Joint plot with histograms, Unemployment vs logarithmic Revenue</iframe>

Now, we observe that revenue optimization typically occurs during periods of economic stability. This suggests that the relationship is not linear but rather intricate and multifaceted. By using a bivariate graph and modal analysis, we can illustrate various cross-genre trends. Specifically, we find that box office revenues peak during periods of full employment, typically when the US unemployment rate falls within the range of 4-6\%. As a result, the most profitable economic environment tends to coincide with a full-employment cycle. Additionally, there seems to be a second mode across the revenue variable as it is visible on the histogram. 


# Casting and Star Power

## Are we on a roll? The impact of Actor Momentum (and Experience)
In our cinematic odyssey, we are not merely deciphering the impact of colossal names gracing the marquee but also diving into the heartbeat of every ensemble — the supporting cast. Picture this: a few iconic actors, a sprinkle of emerging talent, some Hollywood veterans that hand out precious advice and perhaps an unsung hero or two. Together, they form the team of characters that breathe life into the narrative, but how to discover the optimal combination?

To unravel the mysteries surrounding the impact of actors, both main and supporting, on a movie's prosperity, we introduce a novel metric — the Actor Momentum. This metric, a source of insight, gauges the pulse of an actor's career over the last 5 years, current year included. Why, you ask? For we seek not only the emerging talent, but also the nuanced intersection of experience and relevance in Hollywood's ever-evolving landscape.

The Actor Momentum whispers tales of seasoned actors, adorned with the glow of experience, discerning them from the rising stars, just starting their celestial course. Yet, it also raises a question — can the relentless pursuit of roles at all costs impact an actor's performance and, by extension, a movie's triumph?

We start by exploring historical graphs describing tendencies in actor momentum across all categories.

<iframe src="plots/avg_movies_per_year.png" width="750" height="550" frameborder="0">Average movies per actor per year over all genres</iframe>

<iframe src="plots/avg_momentum_per_year.png" width="750" height="550" frameborder="0">Average momentum per actor per year over all genres</iframe>

In our inaugural expedition, where data and cinema converge, we cast our gaze upon the intersection of Actor Momentum and Box Office revenue. A visual symphony unfolds, revealing insights into the heartbeat of Hollywood's financial intricacies. Behold, the scatterplot — a canvas where Actor Momentum meets the towering (inflation-adjusted) revenues of our beloved films. The largest revenues seem to trace a skewed bell curve, gracefully peaking around 50 movies on average in the last 5 years. 

<iframe src="plots/scatter_popularity_revenue.html" width="750" height="550" frameborder="0"> Movie Revenue Scatterplot of average Actor Momentum vs Inflation adjusted revenue by Umbrella Genre</iframe>

However, we can clearly notice that this scatterplot goes against Gauss-Markov assumptions for all our movie genres and thus, Actor Momentum should not be used as a regressor. However, we can still interpret the results for each genre visually.

When taking a glimpse towards the correlation matrix above, we can notice that Actor Momentum and Inflation Adjusted Revenue have mostly very light correlation, culminating at 0.12, for Comedy. This seems to be in line with the scatterplots above, as Comedy, SF and Fantasy as well as Thriller genres have slightly more linear-looking relationships (or at least less flat ones) with the movie revenue.

<iframe src="plots/scatter_popularity_log_revenue.html" width="750" height="550" frameborder="0"> Movie Revenue Scatterplot of average Actor Momentum vs logarithmic Inflation adjusted revenue by Umbrella Genre</iframe>

As mentioned in the regression part, we then decided to choose the logarithm of Inflation Adjusted Revenue as our movie success metric in order to account for the scale differences and the outliers, while keeping an interpretable variable. When examining the scatterplot of the logarithmic revenue, it looks already better (w.r.t. linear regression), compared to the regular plot.

When plotting the residual regression to visualize the tendency within the residuals, we do not learn anything new: the mean of the residuals remains constantly at 0.

<iframe src="plots/popularity_log_revenue_residual.html" width="750" height="550" frameborder="0">"Partialed out" regression of log(Inflation adjusted revenue) on Actor Momentum by Umbrella Genre</iframe>

However, no matter whether we work in the linear or the logarithmic space, the regression results remain quite similar. The coefficient of Actor Momentum always has p-values above 5%, our significance level, no matter the genre. We can thus state that at that level, Actor Momentum is not a significant factor when predicting movie revenue, at least not in the context of a linear regression.

Even when plotting the residual regression between Actor Momentum and logarithmic revenue, we do not find linear-looking data, but rather something heteroskedastic, confirming our initial interpretation of regressions. Thus, the only analysis we will conduct for this feature in our framework is graphical analysis, as said above.

The solution we committed to was grouping the movies by Actor Momentum for each genre, taking the mean for each group, then choosing the group with the maximal mean, in line with the tendency of the graphs of peaking at one point. In order to have a relevant grouping, we decided to round our momentums to the nearest multiple of 5, as this seemed graphically in line with the frequencies of observations.

<iframe src="plots/interactive_scatter_popularity_revenue.html" width="750" height="550" frameborder="0">Scatter plot of Actor Momentum vs Adjusted Revenue</iframe>
<iframe src="plots/interactive_scatter_popularity_count.html" width="750" height="550" frameborder="0">Scatter plot of Actor Momentum vs Movie Count</iframe>

However, one should be careful, as momentum ranges with fewer movies are more affected by outliers. For this reason, we decided to multiply each average box office value by the amount of movies in each group, to have representative data with more robustness against outliers. The results can be seen below.

<iframe src="plots/interactive_scatter_popularity_BOxMC.html" width="750" height="550" frameborder="0">Scatter plot of Actor Momentum vs (Adjusted Revenue * Movie Count)</iframe>

For the majority of genres, this golden equilibrium point materializes at the threshold of 55, where the seasoned veterans and the rising stars converge in a harmonious dance. Yet, the realms of SF and Fantasy stand out at 40 and the realm of Thriller resonates at 50, as the two only different categories.

Now, filmmakers can use this information to determine how to hire their precious cast, mixing up the talent of tomorrow with the experienced professionals, in order to reach that optimal average momentum.

## Cinematic Ladies: Unveiling Women's Impact Over the Box Office
Let's take a joyride through the world of movies to uncover how ladies shine on screen and affect movie money! We will be diving into the history of films to see how women are portrayed and how that impacts how much cash movies make.
We start our analysis by peering through the lens of time, examining the trajectory of female representation across movie genres over the years. The graphs show a gradual rise in female representation, notably gaining momentum post-1980s. However, a zigzag pattern before 1980 raises eyebrows—an anomaly we can attribute to limited movie data for those years. 
What's striking in these pictures is how women are consistently fewer, usually staying below 40 percent, no matter the type of movie. Action films, especially, have barely 30 percent women. On the flip side, romantic stories often have more than 40 percent women, fitting well with what we expect from this kind of movie. These visual signs tell us that some types of movies share roles more evenly, while others lean heavily toward one gender.

<iframe src="plots/gender_over_years.html" width="750" height="550" frameborder="0">Average percentage of women in movies each year</iframe>

In our next exploration, we delved into understanding how the female percentage in movies influences the box office revenue across different genres. Employing linear regression and leveraging the FWL theorem, we meticulously assessed the impact of the female percentage feature on revenue within each genre.
Our analysis unfolded through scatter plots showing the relationship between residualized revenue (adjusted for other factors) and residualized female percentage for each specific genre.
In all genres, a consistent pattern emerged - a faint yet discernible negative trend line. This subtle downward trend shows a weak inverse relationship between the presence of women on screen and the corresponding box office revenue for each genre.

<iframe src="plots/gender_log_revenue_residual.html" width="750" height="550" frameborder="0">"Partialed out" regression of log(Inflation adjusted revenue) on Actor Momentum by Umbrella Genre</iframe>

In light of the initially observed weak relationship, we conducted a second analysis. Specifically, we scrutinized the data by plotting a bar graph showcasing the average revenue across different female percentage intervals. To enhance the depiction of variability, we incorporated 95% confidence intervals (CI) as error bars within the graph.
Upon closer inspection, this refined analysis showed us a more pronounced negative effect associated with varying female representation. Notably, the disparity in average revenue becomes strikingly evident across these intervals. For instance, in the 75-100 interval, the average revenue starkly contrasts, amounting to nearly a third of that in the 0-25 interval.

<iframe src="plots/fig_gender_barplot.html" width="750" height="550" frameborder="0">Female percentage in Movies vs Average Box Office Revenue (Adjusted)</iframe>

However, one should keep in mind that even though the partial effect of female percentage over the years is slightly negative (when not zero due to it being non-significant for some genres), the historical context needs to be taken into account. In an industry mostly dominated by men historically, the presence of women in large numbers is a quite new phenomenon and with the gender difference tightening over the years, one could expect to see a change in this coefficient in the future.

## Portraying Diversity: Ethnic Mosaic on the Silver Screen
Now, we are turning our focus to another crucial aspect: the diversity of ethnicities portrayed on screen. This exploration is pivotal as it sheds light on the changing landscape of representation in movies. We'll be closely examining the evolution of ethnic diversity in films over the years and its potential influence on the box office revenue.
Let's start our exploration by dissecting the evolution of ethnic diversity over the years. As we delved into the historical portrayal of ethnicities, an intriguing pattern emerged. Until the 1960s, the average number of ethnicities fluctuated around two, showcasing a relatively consistent portrayal. However, post this era, a noticeable upward trend in ethnic representation became evident, culminating in an average of around six ethnicities per movie by 2001.
In 2002, there was a sudden dip in the representation of ethnicities across all genres. This downturn might signify a momentary shift in storytelling patterns, potentially influenced by socio-political changes after 9/11 attacks. Post-2002, the trend in ethnic representation exhibited fluctuations without a distinct upward trajectory. It's essential to note that our dataset concludes in 2012, limiting our ability to analyze more recent developments in ethnic portrayal, where the situation could have taken a whole new turn.

<iframe src="plots/ethnicity_over_years.html" width="750" height="550" frameborder="0">Average number of Ethnicities over the years</iframe>

In our quest to explore the impact of the number of ethnicities on box office revenue, we employed the same linear regression method as we used in gender analysis. Visualizing the results through scatter plots of residualized revenue against residualized average number of ethnicities revealed a modest positive trend line. Notably, the trend exhibited its strongest correlation in action films, while appearing weaker in romantic, comedy, and drama genres.

<iframe src="plots/ethnicity_log_revenue_residual.html" width="750" height="550" frameborder="0">"Partialed out" regression of log(Inflation adjusted revenue) on Actor Momentum by Umbrella Genre</iframe>

To conclude, we can see that over all genres, the impact of having more ethnicities is positive and always significant. The largest impact is made on Romance and Thriller genres, while Drama movies seem to get the least benefits from a diverse cast.

Filmmakers should thus keep in mind that diversity is the key to (at least a part of) success!


# Creative Elements and Film Characteristics

## Runtime Rodeo: The Great Movie Length Roundup
Are you into short and sweet or epic odysseys? In this section we’ll take a look at the structural evolution of movie runtimes and  the effect of runtime of box office revenue and vice versa. 

Now, let's break it down. Movies these days are probably a bit longer because of the advancement in movie production technology. As we travel through the decades, the runtime has been going up by about 4.33% every ten years.  


<iframe src="plots/blockbusters_year.png" width="750" height="550" frameborder="0">Blockbusters and median revenue per year</iframe>

<iframe src="plots/short_movies_revenue.png" width="750" height="550" frameborder="0">Blockbusters and median revenue per year</iframe>

Turns out Blockbusters (2 standard deviations above median earnings), in particular, are on average, about 15.76% longer than your average flick (and statistically different at a level of 5%!). But here's the kicker: short movies aren't necessarily making less money. Nope, some of those shorties are hitting blockbuster status. So, in the vast cinema landscape, it turns out the length of the film doesn't always rope in success. 

Last but not least, let's have a look at the distribution between movie length density and logarithmic revenue.

<iframe src="plots/h2c_runtime_log_revenue.html" width="750" height="550" frameborder="0">Joint plot with histograms, Runtime vs logarithmic Revenue</iframe>


## Lost in Translation: Multilingual Movies and Theater Proceeds


In a globalized film industry, the impact of language diversity on box office revenues presents itself as a captivating enigma. Our investigation delves into the intriguing question of whether a film's multilingual approach influences its financial performance at the theater doors. Join us as we explore the dynamics of linguistic diversity in the world of cinema and its correlation with box office proceeds.

As is customary in our storytelling journey, we are deeply interested in understanding the behavior and making comparisons across different movie genres. In contrast to many other questions we aim to answer through this data exploration, this particular inquiry reveals notable variations among various movie categories.

To address this question, we employ a box plot analysis that showcases the means and their 95\% confidence intervals. This method offers a concise and visually intuitive framework for interpreting box office revenues concerning movies translated into a specific number of languages in addition to the original version (which is included in the count). A second, and more statistically rigorous measure is the 'partialled out' regression coefficient, which is equivalent to the corresponding coefficient of the multivariate regression of box office revenues on the languages variable across all genres. In all genres except 'romantic,' this coefficient is both statistically significant and positive. This finding provides a robust foundation for our upcoming, somewhat qualitative analysis.

<iframe src="plots/boxplot_language_log_revenue.html" width="750" height="550" frameborder="0"> Box plot of Number of languages vs log(Inflation adjusted revenue) by Umbrella Genre</iframe>

Upon examining the box plots, a similar trend emerges for the drama and niche genres. Both genres exhibit an increase in average revenue as the number of languages rises, particularly between the 1 and 4 language marks. Visually, we observe that the confidence intervals for, say, the case with only 1 language and the case with 4 languages do not overlap. While this is not a formal hypothesis testing procedure, it provides a visual representation of a pattern within these movie categories, a pattern that is less pronounced or even absent in other genres.

Of course, pinpointing the exact cause or causes behind this phenomenon is challenging, and statistical validation is even more complex. However, for interpretative purposes, we might intuitively consider that the drama genre places a higher demand on viewers in terms of plot comprehension and understanding of dialogue. In such cases, successful movies may choose to offer a broader range of languages to accommodate varying audience needs. This can be contrasted with genres where translation is not integral to the plot, as is often the case in action or sci-fi films, or genres where translation involves intricate and substantial changes to scenes and plot, such as comedy movies reliant on local idioms and language-specific expressions.


# Conclusion
