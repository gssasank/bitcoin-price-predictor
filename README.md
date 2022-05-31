
## BITCOIN PRICE PREDICTOR

# Abstract

The disruptive potential of Decentralized Ledger Technologies and Digital Assets have gone mainstream since 2009. We have seen Bitcoin go all time high this past year, with a few major drops as well. Despite this dynamic rise and fall in the value of cryptocurrency, people’s interest in cryptocurrency has skyrocketed. How do we look for a solution to this dynamic state of cryptocurrency prices? A simple answer to this problem is to use Data Science on the available crypto data to predict the prices of the cryptocurrency. In this project, we use various models to try to predict Bitcoin (BTC) prices and compare them to the actual prices and find the best suitable model with minimum error. The models we have used include Linear Regression, Support Vector Machine (SVM), Extreme Gradient Boost (XGBoost), etc. Comparing the values of mean absolute error and root mean squared error, we can find the best-suited model for predicting cryptocurrency prices. 


# Introduction

Cryptocurrencies are a form of digital currency based on Blockchain Technology which is a type of cryptographic technique to secure digital payments. The Blockchain is a decentralized network of blocks distributed to all the nodes present in the network.  The Blockchain maintains a decentralized and secure record of all transactions that take place across the network. That’s why these transactions/currencies are nearly impossible to counterfeit or double-spend. The blockchain consists of a chain of blocks. In layman’s terms, these blocks each contain the transaction record, and a hash of the key to the previously filled block therefore, many such blocks are locked in with each other to form the Blockchain. Therefore, in order to access a transaction, the user needs to unlock all the blocks till they find the block with the hash of the block containing the transaction that they are looking for. Therefore, even if a single value is changed in this huge network of blocks, it changes all the hash values and this reflects in the whole decentralized network. That’s why blockchain is considered to be impossible to break into. 

Bitcoin is one such cryptocurrency, probably the most popular one which has gained popularity since 2009. Started by a mysterious person called Satoshi Nakamoto as an open-source technology, Bitcoin is owned and controlled by its users as already mentioned, it is a decentralized database of transactions. Bitcoin started from $0.8 per coin in 2010 and saw its first-ever growth of 3200% in June 2011. Thereafter, it has seen many rises and has seen an all-time high of 64,888 dollars per coin which is mind-blowing. So, it can be said it is the fastest-growing asset in the market right now and everybody is talking about it. We use time-series predictions to get a proper prediction of values. A time series is a set of data values with respect to successive moments in time.

**Motivations**

Cryptocurrency is a disruptive technology that has seen the interest of people from all around the world. However, the dynamic nature of the market and the value of the cryptocurrency needs to be studied to analyze and harness its full potential. We chose the price prediction of Bitcoin as it is one of the most popular cryptocurrencies with the largest market capitalization.  One of the main problems with decentralized cryptocurrencies is price volatility, which indicates the need for studying the underlying price model. Being able to predict the price of bitcoin will help users in a significant way as it is the cryptocurrency with the highest market capitalization and the one which has the maximum growth rate.


# Literature review

The name of main paper which was considered for the project is “Time-series forecasting of Bitcoin prices using high-dimensional features: a machine learning approach”. It was written by researchers from the University of Qatar, Qatar.  They claimed Bitcoin prices exhibit nonstationary behavior, where the statistical distribution of data changes over time. The paper demonstrated high-performance machine learning-based classification and regression models for predicting Bitcoin price movements and prices in short and medium terms. Prediction of Bitcoin price is usually done using only machine learning-based classification and has been studied for only a one-day time frame, while this work goes beyond that by using machine learning-based models for 1, 7, 30, and 90 days. Various different metrics were taken into consideration by the authors of this paper such as the number of Tweets that mentioned the word “bitcoin” like a spike in such Tweets directly correlated with a spike in the price of Bitcoin. There is a wide range of features in the data and our proposed ML model handles autocorrelation, seasonality, and trend effects. The training process of pure time-series models, however, needs to be manually tuned to address these effects. 

Another paper we have considered was “Bitcoin price prediction using machine learning: An approach to sample dimension engineering”. The statistical methods used in this paper were Logistic Regression and Discriminant Analysis. 

“Prediction of Bitcoin prices with machine learning methods using time-series data” was another paper we reviewed for our project. In this study, Bitcoin prediction is performed with Linear Regression (LR) and Support Vector Machine (SVM) from machine learning methods by using a time series consisting of daily Bitcoin closing prices between 2012-and 2018. The prediction model with the least error is obtained by testing with different parameter combinations such as SVM including linear and polynomial kernel functions. We measure the performance of our model by means of statistical errors namely Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). Another indicator used to measure performance was Pearson Correlation. It is seen that the price prediction performance of the proposed SVM model for the Bitcoin data set is better than that of the LR model as the error values come out to be lower in the case of SVM than in the LR model.

Instead of only considering the Twitter trends for Bitcoin, we have also decided to consider Google search trends as they might be an overall better indicator of the popularity of the asset. We have also considered various other metrics that were not mentioned in the paper to improve the analysis and prediction. 


# Proposed Model

<img width="589" alt="image" src="https://user-images.githubusercontent.com/62089940/171198001-3063dc51-da47-4777-a18a-fb3f5531a85a.png">

# Methodology


## Data Collection and Web Scraping

We used a website called bitinfocharts.com to obtain most of the data regarding our dataset and a few more features were obtained from Quandl. The features that were selected for analysis were as follows.

i) Number of transactions in blockchain per day

ii) Average block size (KB)

iii) Number of sent by addresses

iv) Number of active addresses

v) Average mining difficulty (Hash/day)

vi) Average hash rate (hash/s)

vii) Mining Profitability 

viii) Sent coins (USD)

ix) Average & Median transaction fee (USD)

x) Average block time (minutes)

xi) Average & Median Transaction Value (USD)

xii) Tweets & Google Trends to “Bitcoin” per day

xiii) Average Fee Percentage in Total Block Reward

xiv) Top 100 Richest Addresses to Total coins

xv) Miner Revenue (USD)

xvi) Number of coins in circulation


## Major Algorithms and Techniques used


### Random Forest

The Random Forest algorithm is an algorithm that makes the use of decision trees. Decision tree is a learning method to predict classification and regression. The Random Forest Algorithm is composed of different decision trees, to predict classification and regression. It merges the decisions of multiple decision trees in order to find an answer, which, in case of classification, is taking the majority of the votes and giving the classification and, in case of regression, is taking the average of the predicted values by the decision trees. The random forest algorithm is a supervised learning model; it uses labeled data to “learn” how to classify unlabeled data.


<img width="1199" alt="image" src="https://user-images.githubusercontent.com/62089940/171199573-1230320d-ff2c-46d3-9958-03239acadd71.png">


Either the Gini impurity or the Entropy is calculated of each branch on a node. This impurity or entropy can be used to determine which branch is most likely to occur. Here, pi represents the relative frequency of the class you are observing in the dataset. MSE stands for the Mean Squared Error which calculates the distance of each node from the predicted actual value (error calculation), helping to decide which branch is the better decision for your forest.


### Dummy Regression

The Dummy Regressor is a kind of Regressor that gives predictions based on simple strategies without paying any attention to the input data. This is a special kind of regressor that is usually only applied to time series data. This classifier is useful as a simple baseline to compare with other (real) classifiers. This is not meant to be a standalone classifier or regressor of any model.


### Linear Regression

Linear Regression is a supervised machine learning algorithm. It performs regression tasks. Regression models a target prediction value based on independent variables. The most common use of Linear Regression involves finding out the relationship between variables and forecasting. Different regression models differ based on – the kind of relationship between dependent and independent variables they are considering and the number of independent variables being used. 

<img width="760" alt="image" src="https://user-images.githubusercontent.com/62089940/171199997-c5bd11ca-3242-4314-9d3e-2550e9b07a0c.png">


### Support Vector Regression

Support Vector Regression is a supervised learning algorithm that is used to predict discrete values. Using the same principle as the SVMs, the basic idea behind SVR is to find the best fit line. In SVR, the best fit line is the hyperplane that has the maximum number of points. 

Equation of Hyperplane: Y = wx+b  

Decision Boundary Planes: wx+b= +a | wx+b= -a           

Any Hyperplane satisfying SVR: -a &lt; Y- wx+b &lt; +a            


### Adaboost 

AdaBoost also called Adaptive Boosting is a technique in Machine Learning used as an Ensemble Method where instead of many trees like in Random Forest, many stumps or weak classifiers are aggregated to give a specific outcome. Adaboost builds a model and gives equal weights to all the data points. It then assigns higher weights to points that are wrongly classified. Now all the points which have higher weights are given more importance in the next model. It will keep training models until and unless a lower error is received. 

Algorithm for Adaboost: 

<img width="958" alt="image" src="https://user-images.githubusercontent.com/62089940/171200422-27f664c5-41e2-4c5c-83b6-48002dd38164.png">

### XGBoost

Extreme Gradient Boost is a scalable and improved version of the gradient boosting algorithm designed for efficacy, computational speed, and model performance. It is the best Ensemble Technique as it incorporates the best of all the boosting algorithms. It especially improves on the Gradient Boost algorithm in two ways, the first being tree pruning and the second being sparsity aware split finding. These improvements increase the performance by a large margin over Gradient Boost. There are also system level improvements such as parallelization over multiple CPU cores and it also efficiently uses system cache to index and improve the performance of the algorithm.


# Experimentation and Results

Since there was no proper dataset for the experimentation we had to create our own dataset. After extensive research we found [https://bitinfocharts.com/](https://bitinfocharts.com/) to provide more reliable data with lots of features, so we scraped the day-wise bitcoin data ranging from 2013 to the present day ( 26-04-2022 ). After cleaning and formatting, the dataset contained 25 features and 3000 rows. After doing EDA we finalized the dataset by shortlisting the 10 most significant features using the feature importance from Random Forest Regressor.


### Final Dataset  
 
<img width="1240" alt="image" src="https://user-images.githubusercontent.com/62089940/171200598-bd92e013-4ec9-466f-b118-f21a9d45dacb.png">

### Features 

Note - We did feature smoothening using a simple moving average ( sma ). SMA calculates the average of a selected range of feature values for a number of periods in that range. It determines if a price will follow a bull or bear trend. Any feature with the prefix sma(x) means that we have taken a simple moving average over x days.

Sma90 avg_transaction_value :  Bitcoin transactions are messages digitally signed using cryptography and sent to the entire Bitcoin Network for verification. The number of daily transactions highlights the value of the Bitcoin network to securely transfer funds. Here the data is taken with sma90

Closing_price : Price of the bitcoin at the end of the business day 

sma7 highest_price : Highest price of the bitcoin recorded  at the end of the business day taken over sma7

opening_price	 : Price of the bitcoin at the start of the business day

sma90 closing_price : Price of the bitcoin at the end of the business day taken over sma90

sma7 opening_price	: Price of the bitcoin at the start of the business day taken over sma7

Highest_price : highest price of the bitcoin recorded  at the end of the business day

sma90 highest_price	: highest price of the bitcoin recorded  at the end of the business day taken over sma90

sma30 lowest_price : Lowest price of the bitcoin recorded  at the end of the business day

sma30 closing_price : Price of the bitcoin at the end of the business day taken over sma30

Next_day_closing_price : Price of the bitcoin at the end of the next business day. This is the field we have to predict 

### Training Method 

The target variable is highly volatile where prices are ranging from 100 USD to 63500 USD. Initially, we tried with time series split cross-validation, and random split cross-validation, but the results were mostly overfitting.

Thus, we decided to create multiple splits and train models for each split separately. Each train Split will consist of 500 data points & next 100 data points will be used for testing. E.g., If there are 10 splits, 10 models will be trained and each model will predict 100 data points. To summarize, the prediction of the next 100 days will be based on data considered from the last 500 days. The final metric will be the mean over metric reported by each split.


### Model Training and Results


#### Dummy Regressor

Results of Test Data

<img width="1176" alt="image" src="https://user-images.githubusercontent.com/62089940/171201002-de3aca2a-68af-4ac5-b305-ea9db49a31d2.png">

Model metrics 

<img width="418" alt="image" src="https://user-images.githubusercontent.com/62089940/171201058-17c5655d-ae9f-419a-895c-e3fb970830a1.png">

This is just the baseline model, so the results will not be as well as other models.


#### Support Vector Regressor

Results on Test Data

<img width="1151" alt="image" src="https://user-images.githubusercontent.com/62089940/171201151-b0f8bbc3-eeb6-4fff-8b84-876054785cc2.png">

Model metrics 

<img width="392" alt="image" src="https://user-images.githubusercontent.com/62089940/171201196-4d411e95-1588-480b-8730-90636a9e857c.png">

Similar to Linear Regression, SVR also is able to predict good results and capture the trend, even when there is a sudden spike in prices.


#### Adaboost Regressor

Results on Test Data

<img width="1156" alt="image" src="https://user-images.githubusercontent.com/62089940/171201481-b712b55c-1142-41ee-b447-3bda1925f6ec.png">

Model metrics 

<img width="414" alt="image" src="https://user-images.githubusercontent.com/62089940/171201530-228149dc-859a-4827-8f5a-a54068567636.png">

AdaBoost is highly overfitting. We can observe that when there is a sudden spike in prices, It predicts unsatisfactory results


#### XGBoost

Results on Test Data

<img width="1151" alt="image" src="https://user-images.githubusercontent.com/62089940/171201606-7a63daa6-065c-4cf3-b68c-a86223529c0d.png">

Model metrics 

<img width="412" alt="image" src="https://user-images.githubusercontent.com/62089940/171201678-adc6d787-4404-462f-a901-f5f4b9602d0b.png">

Similar to Adaboost, XGBoost is also highly overfitting. We can observe that when there is a sudden spike in prices, it predicts unsatisfactory results


#### Linear Regressor

Results on Test Data

<img width="1154" alt="image" src="https://user-images.githubusercontent.com/62089940/171201729-9359e107-9c65-4920-8add-d7bac8469588.png">

Model metrics 

<img width="392" alt="image" src="https://user-images.githubusercontent.com/62089940/171201768-f2424580-cda5-4bdc-8750-92ba0f970b33.png">

This model performs significantly better than SVR and XGBoost. As observed earlier in correlation, there was a very good linear relationship between some features and target variable. This ultimately has made linear models like SVR & Linear regression fit a good hyperplane on data.


# Conclusion and Limitations

From the above analysis, we conclude that Linear models like Linear Regression and SVR provide better results at predicting the value than any other model we have tested. The price of Bitcoin is very volatile and that makes it quite difficult to predict. The papers mentioned in the literature review also used deep learning models such as LSTM to predict the price and this could be done in a future implementation as an extension of the models used. Another challenge we faced was obtaining real-time, accurate data. While the dataset generated by scraping bitinfocharts is mostly clean and accurate, there are a lot of gaps in the dataset, especially in the fields of Twitter trends and Google Analytics. A paid dataset or an API that could provide filtered data would in theory perform slightly better overall. Extrapolating this model to other cryptocurrencies also may be a challenge as except for Bitcoin and maybe Ethereum, there exists no other cryptocurrency which has as many details publicly available freely. Bitcoin also has the edge of being the oldest cryptocurrency and as a result, has the most data available to train on. Newer cryptocurrencies do not have this advantage but this can be overcome to some extent by using more novel prediction techniques using Deep Learning.


# References



1. Mohammed Mudassir, Shada Bennbaia,  Devrim Unal and Mohammad Hammoudeh.( June 2020).”Time-series forecasting of Bitcoin prices using high-dimensional features: a machine learning approach”. _Springer_.
2. Muniye, Temesgen & Rout, Minakhi & Mohanty, Lipika & Satapathy, Suresh. (2020). “Bitcoin Price Prediction and Analysis Using Deep Learning Models”.
3. S. Karasu, A. Altan, Z. Saraç and R. Hacioğlu, "Prediction of Bitcoin prices with machine learning methods using time series data," _2018 26th Signal Processing and Communications Applications Conference (SIU), _2018, pp. 1-4
4. Zheshi Chen, Chunhong Li, Wenjun Sun. (2020). ” Bitcoin price prediction using machine learning: An approach to sample dimension engineering”, _Journal of Computational and Applied Mathematics._
