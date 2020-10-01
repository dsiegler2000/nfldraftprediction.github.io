# Project Proposal
### Introduction
The National Football League is a very popular league in America, and the draft is the primary way in which teams get players straight out of college. Thus, we are very interested in predicting what teams should draft players at which position and when. 

### Problem Definition
We are looking to find the ideal draft position for a team to move up or down to in order to get a player at the optimal position at the best value. Given football data and possible params, pick number, team, player rankings, positions of the draft-eligible players: output a position at which the team has the best value

### Methods
First, for the clustering problem, we plan to come up with some consistent metrics across all players/positions. This will enable us to run a clustering algorithm. We feel that a Gaussian Mixture Model would be most appropriate, as positions are fluid and thus we would like soft assignment, but we don’t expect there to be constant density in the data so we think DBSCAN would not do too well. We also plan to try hierarchical clustering; from this clustering, we hope to gain some insight into how to reduce the problem by combining some positions. 

We plan to simplify the problem by assuming that a team has a rank of positions that it wants (ex. QB, defensive lineman, etc), they will pick the top ranked player available for their top position, and if no players ranked in the top 10 of players still available are still available for the team’s top position, they will move on to their 2nd to top position, etc. Though this is not always true, this simplifying assumption allows us to reduce this into a classification problem. Because we are going to put a significant amount of work into feature engineering during our data analysis phase, we expect that a simple predictive model such as SVM will be sufficient, but we plan to explore an SVM, linear regression, and a simple deep neural network.  

### Potential Results
From our clustering, we want to primarily derive analysis; namely, we want to get more insight into what positions are similar and what positions we can combine to reduce the problem. Additionally, we hope to derive some “weightings” to help us to come up with unified statistics across most or all positions. Since each position has different statistics that are tracked, coming up with a few unified statistics about all players will greatly help us during the predictive modeling phase. 

From our predictive model, we hope to have a model that will predict what position a team wants to take for any given pick. We can then use this to do a full draft simulation, however, this will be highly stochastic as even just a small number of wrong predictions compared to reality will cause our simulation draft to greatly diverge from reality. Thus, a more practical use/result would be to simply predict the next pick for a team while the draft is happening. 

### Discussion
There has been significant research on this topic, for various purposes from entertainment to betting to academia. Some publicly available datasets of note are [1] and [2]. Additionally, [3] uses simple linear regression with feature engineering to achieve results in predicting quarterback performance. Finally, [4] applies similar methods to tight ends. 

### References
[1] Banta, K. (2018, March 17). _NFL Combine 2000-2017_. Kaggle. https://www.kaggle.com/kbanta11/nfl-combine. 

[2] Wexler, R. (2017, October 25). _EDA for NFL Draft Outcomes Data_. Kaggle. https://www.kaggle.com/rwexler/eda-for-nfl-draft-outcomes-data. 

[3] Wolfson, J., Addona, V., Schmicker, R. (2011, July 19). _The Quarterback Prediction Problem: Forecasting the Performance of College Quarterbacks Selected in the NFL Draft._ De Gruyter. https://www.degruyter.com/view/journals/jqas/7/3/article-jqas.2011.7.3.1302.xml.xml. 

[4] Mulholland, J., & Jensen, S. (2014, December 01). _Predicting the draft and career success of tight ends in the National Football League._ Retrieved October 01, 2020, from https://www.degruyter.com/view/journals/jqas/10/4/article-p381.xml
