Intro to Data Science with R Programming
==================

Brought to you by [Lesley Cordero](http://www.columbia.edu/~lc2958) and [ADI](https://adicu.com)

## Table of Contents

- [0.0 Setup](#00-setup)
    + [0.1 R and R Studio](#01-r-and-r-studio)
    + [0.2 Packages](#02-packages)
- [1.0 Background](#10-background)
    + [1.1 Machine Learning](#11-Machine Learning)
    + [1.2 Data](#12-data)
    + [1.3 Overfitting vs Underfitting](#13-overfitting-vs-underfitting)
    + [1.4 Glossary](#14-glossary)
        * [1.4.1 Factors](#141-factors)
        * [1.4.2 Corpus](#142-corpus)
        * [1.4.3 Bias](#143-bias)
        * [1.4.4 Variance](#144-variance)
- [2.0 Data Preparation](#30-data-preparation)
    + [2.1 dplyr](#31-dplyr)
    + [2.2 Geopandas](#32-geopandas)
- [3.0 Exploratory Analysis](#30-exploratory-analysis)
- [4.0 Data Visualization](#50-data-visualization)
- [5.0 Machine Learning & Prediction](#50-machine-learning--prediction)
    + [5.1 Random Forests](#51-random-forests)
    + [5.2 Natural Language Processing](#52-natural-language-processing)
        * [5.2.1 ANLP](#521-anlp)
    + [5.3 K Means Clustering](#53-k-means-clustering)
- [6.0 Final Exercise]($60-final-exercise)
- [7.0 Final Words](#60-final-words)
    + [7.1 Resources](#61-resources)
    + [7.2 Mini Courses](#72-mini-courses)


## 0.0 Setup

This guide was written in R 3.2.3.


### 0.1 R and R Studio

Download [R](https://www.r-project.org/) and [R Studio](https://www.rstudio.com/products/rstudio/download/).


### 0.2 Packages

Next, to install the R packages, cd into your workspace, and enter the following, very simple, command into your bash: 

```
R
```

This will prompt a session in R! From here, you can install any needed packages. For the sake of this tutorial, enter the following into your terminal R session:

``` 
install.packages("ggvis”)
install.packages("gmodels")
install.packages("RCurl")
install.packages("tm")
install.packages("caTools")
install.packages("ggplot2")
install.packages("RFinfer")
install.packages("dplyr")
install.packages("lubridate")
install.packages("compare")
install.packages("downloader")
```

### 0.3 Virtual Environment

If you'd like to work in a virtual environment, you can set it up as follows: 
```
pip3 install virtualenv
virtualenv your_env
```
And then launch it with: 
```
source your_env/bin/activate
```

To execute the visualizations in matplotlib, do the following:

```
cd ~/.matplotlib
vim matplotlibrc
```
And then, write `backend: TkAgg` in the file. Now you should be set up with your virtual environment!

Cool, now we're ready to start!


## 1.0 Background

Before we head into an actual data science problem demo, let's go over some vital background information. 

### 1.1 What is Data Science?

Data Science is the application of statistical and mathematical methods to problems involving sets of data. In other words, it's taking techniques developed in the areas of statistics and math and using them to learn from some sort of data source. 

#### 1.1.1 What do you mean by data? 

Data is essentially anything that can be recorded or transcribed - numerical, text, images, sounds, anything!

#### 1.1.2 What background do you need to work on a data science problem?

It depends entirely on what you're working on, but generally speaking, you should be comfortable with probability, statistics, and some linear algebra.  

### 1.2 Is data science the same as machine learning?

Well, no. They do have overlap, but they are not the same! Whereas the topic of machine learning involves lots of theoretical components we won't worry about, data science takes these methods and applies them to the real world. It's important to note that studying these theoretical components can be very useful to your understanding of data science, however!

### 1.3 Why is Data Science important? 

Data Science has so much potential! By using data in creative and innovative ways, we can gain a lot of insight on the world, whether that be in economics, biology, sociology, math - any topic you can think of, data science has its role. 

### 1.4 Machine Learning

Generally speaking, Machine Learning can be split into three types of learning: supervised, unsupervised, and reinforcement learning. 

#### 1.4.1 Supervised Learning

This algorithm consist of a target / outcome variable (or dependent variable) which is to be predicted from a given set of predictors (independent variables). Using these set of variables, we generate a function that map inputs to desired outputs. The training process continues until the model achieves a desired level of accuracy on the training data. Examples of Supervised Learning: Regression, Decision Tree, Random Forest, KNN, Logistic Regression etc.


#### 1.4.2 Unsupervised Learning

In this algorithm, we do not have any target or outcome variable to predict / estimate.  It is used for clustering population in different groups, which is widely used for segmenting customers in different groups for specific intervention. Examples of Unsupervised Learning: Apriori algorithm, K-means.


#### 1.4.2 Reinforcement Learning

Using this algorithm, the machine is trained to make specific decisions. It works this way: the machine is exposed to an environment where it trains itself continually using trial and error. This machine learns from past experience and tries to capture the best possible knowledge to make accurate business decisions. Example of Reinforcement Learning: Markov Decision Process.


### 1.5 Data 

As a data scientist, knowing the different forms data takes is highly important. 

#### 1.5.1 Training vs Test Data

When it comes time to train your classifier or model, you're going to need to split your data into <b>testing</b> and <b>training</b> data. 

Typically, the majority of your data will go towards your training data, while only 10-25% of your data will go towards testing. It's important to note there is no overlap between the two. Should you have overlap or use all your training data for testing, your accuracy results will be wrong. Any classifier that's tested on the data it's training is obviously going to do very well since it will have observed those results before, so the accuracy will be high, but wrongly so. 


#### 1.5.2 Open Data 

What's open data, you ask? Simple, it's data that's freely  for anyone to use! Some examples include things you might have already heard of, like APIs, online zip files, or by scraping data!

You might be wondering where this data comes from - well, it can come from a variety of sources, but some common ones include large tech companies like Facebook, Google, Instagram. Others include large institutions, like the US government! Otherwise, you can find tons of data from all sorts of organizations and individuals. 

### 1.6 Overfitting vs Underfitting

In section 1.2.1, we mentioned the concept of overfitting your data. The concept of overfitting refers to creating a model that doesn't generalize to your model. In other words, if your model overfits your data, that means it's learned your data <i>too</i> much - it's essentially memorized it. This might not seem like it would be a problem at first, but a model that's just "memorized" your data is one that's going to perform poorly on new, unobserved data. 

Underfitting, on the other hand, is when your model is <i>too</i> generalized to your data. This model will also perform poorly on new unobserved data. This usually means we should increase the number of considered features, which will expand the hypothesis space. 


### 1.7 Glossary 

#### 1.7.1 Factors

Factors in R are stored as a vector of integer values with a corresponding set of character values to use when the factor is displayed. 

#### 1.7.2 Corpus

A Corpus (Plural: Corpora) is a collection of written texts that serve as our datasets.

#### 1.7.3 Bias

In machine learning, bias is the tendency for a learner to consistently learn the same wrong thing. 

#### 1.7.4 Variance 

Variance is the error from sensitivity to small fluctuations in the training set. High variance can cause overfitting since it causes a classifier to  model the random noise in the training data rather than the intended outputs.

## 2.0 Data Preparation

### 2.1 dplyr

dplyr allows us to transform and summarize tabular data with rows and columns. It contains a set of functions that perform common data manipulation operations like filtering rows, selecting specific columns, re-ordering rows, adding new columns, and summarizing data.

First we begin by loading in the needed packages:
``` R
library(dplyr)
library(downloader)
```

Using the data available in [this](https://github.com/lesley2958/data-science-r/blob/master/msleep_ggplot2.csv) repo, we''ll load the data into R:

``` R
url <- "https://raw.githubusercontent.com/genomicsclass/dagdata/master/inst/extdata/msleep_ggplot2.csv"
filename <- "msleep_ggplot2.csv"
if (!file.exists(filename)) download(url,filename)
msleep <- read.csv("msleep_ggplot2.csv")
head(msleep)
```

#### 2.1.1 select()

To demonstrate how the `select()` method works, we select the name and sleep_total columns.

``` R
sleepData <- select(msleep, name, sleep_total)
head(sleepData)
```

To select all the columns except a specific column, you can use the subtraction sign:

``` R
head(select(msleep, -name))
```

You can also select a range of columns with a colon:

``` R
head(select(msleep, name:order))
```

#### 2.1.2 filter()

Using the `filter()` method in dplyr we can select rows that meet a certain criterion, such as in the following:

``` R
filter(msleep, sleep_total >= 16)
```
There, we filter out the animals whose sleep total is less than 16 hours. If you want to expand the criteria, you can: 

```R
filter(msleep, sleep_total >= 16, bodywt >= 1)
```

#### 2.1.3 Functions

`arrange()`: re-order or arrange rows <br>
`filter()`: filter rows <br>
`group_by()`: allows for group operations in the “split-apply-combine” concept <br>
`mutate()`: create new columns <br>
`select()`: select columns <br>
`summarise()`: summarise values


## 3.0 Exploratory Analysis

### 3.1 summary()
R gives you the opportunity to go more in-depth with the summary() function. This will give you the minimum value, first quantile, median, mean, third quantile and maximum value of the data set Iris for numeric data types.

``` R
summary(iris) 
```

### 3.2 xda

xda contains tools to perform initial exploratory analysis on any input dataset. It includes custom functions for plotting the data as well as performing different kinds of analyses such as univariate, bivariate, and multivariate investigation - the typical first step of any predictive modeling pipeline. This is a great package to start off on any dataset because it gives you a good sense of the dataset before jumping on to building predictive models.


### 3.3 preprosim

[preprosim](https://mran.revolutionanalytics.com/web/packages/preprosim/vignettes/preprosim.html) helps to add contaminations (noise, missing values, outliers, low variance, irrelevant features, class swap (inconsistency), class imbalance and decrease in data volume) to data and then evaluate the simulated data sets for classification accuracy.


## 4.0 Data Visualization 


### 4.1 ggvis 

ggvis allows you to make scatterplots, as with the following: 


``` R
library(ggvis)
iris %>% ggvis(~Petal.Length, ~Petal.Width, fill = ~Species) %>% layer_points()

```

### 4.2 heatmaply

[heatmaply](https://mran.revolutionanalytics.com/package/heatmaply/) produces interactive heatmaps.

This code snippet shows the correlation structure of variables in the mtcars dataset:

``` R
library(heatmaply)
heatmaply(cor(mtcars), 
k_col = 2, k_row = 2,
limits = c(-1,1)) %>% 
layout(margin = list(l = 40, b = 40))
```

## 5.0 Machine Learning & Prediction


### 5.1 Random Forests

Random forest is a great choice for nearly any prediction problem, even non-linear ones, that belongs to a larger class of machine learning algorithms called ensemble methods.


#### 5.1.1 RFinfer

RFinfer provides functions that use the infinitesimal jackknife to generate predictions and prediction variances from random forest models.

Now we'll go through an exercise involving RFinfer. First, we'll load the needed package and example data included in R. (Specifically, the dataset we'll be using is the New York Air Quality Measurements). 

``` R
library(RFinfer)
library(ggplot2)
data('airquality')
```

Because calls to random forest do not allow missing data, we omit incomplete cases of the data and high outliers.

``` R
d.aq <- na.omit(airquality)
d.aq <- d.aq[d.aq$Ozone < 100, ]
```

Now we finally train the random forest model: 
``` R
rf <- randomForest(Ozone ~ .,data=d.aq,keep.inbag=T)
```

Here, we grab the prediction variances for the training data along with the 95% confidence intervals:

``` R 
rf.preds <- rfPredVar(rf,rf.data=d.aq,CI=TRUE)
str(rf.preds)
```

Then we get: 
```
## 'data.frame':    104 obs. of  4 variables:
##  $ pred       : num  37.2 29.5 19.8 21.9 23.9 ...
##  $ pred.ij.var: num  -1.29 15.7 20.31 4.94 3.78 ...
##  $ l.ci       : num  39.71 -1.25 -19.97 12.25 16.47 ...
##  $ u.ci       : num  34.7 60.3 59.6 31.6 31.3 ...
```

Next, we'll plot the predictions with their 95% confidence intervals in accordance to the actual values.

``` R
ggplot(rf.preds,aes(d.aq$Ozone,pred)) + 
    geom_abline(intercept=0,slope=1,lty=2, color='#999999')  +
   geom_point()  +
   geom_errorbar(aes(ymin=l.ci,ymax=u.ci,height=0.15)) + 
   xlab('Actual') + ylab('Predicted') +
   theme_bw()
```

Here, we can see that the random forest is generally less confident about its inaccurate predictions, which we visualize by plotting the prediction variance as a function of the prediction error.


``` R
qplot(d.aq$Ozone - rf.preds$pred,rf.preds$pred.ij.var, xlab='prediction error',ylab='prediction variance') + theme_bw()
```

### 5.2 Natural Language Processing

#### 5.2.1 ANLP

[ANLP](https://mran.revolutionanalytics.com/web/packages/ANLP/vignettes/ANLP_Documentation.html) provides functions for building text prediction models. It contains functions for cleaning text data, building N-grams and more. 


### 5.3 k-Means Clustering

K Means Clustering is an unsupervised learning algorithm that clusts data based on their similarity. In k means clustering, we have the specify the number of clusters we want the data to be grouped into. The algorithm randomly assigns each observation to a cluster, and finds the centroid of each cluster. Then, the algorithm iterates through two steps:

- Reassign data points to the cluster whose centroid is closest
- Calculate new centroid of each cluster

These two steps are repeated till the within cluster variation cannot be reduced any further. The within cluster variation is calculated as the sum of the euclidean distance between the data points and their respective cluster centroids.


## 6.0 Final Exercise 

For this final exercise, we'll be implementing a sentiment analysis classifier. Sentiment analysis involves building a system to collect and determine the emotional tone behind words. This is important because it allows you to gain an understanding of the attitudes, opinions and emotions of the people in your data. 

At a high level, sentiment analysis involves Natural language processing and artificial intelligence by taking the actual text element, transforming it into a format that a machine can read, and using statistics to determine the actual sentiment.

For the model portion of this exercise, we'll use linear models since they allow us to define our input variable as a linear combination of input variables. 

For this tutorial, we'll be using the following packages: 

``` R
library(RCurl)
library(tm)
library(caTools)
```

### 6.1 Data Preparation

Here, we're just loading the data from the URLs. Although the R function read.csv can work with URLs, it doesn't necessarily handle https, so we use the package RCurl to ensure our links are able to be downloaded.

``` R
test_data_url <- "https://dl.dropboxusercontent.com/u/8082731/datasets/UMICH-SI650/testdata.txt"
train_data_url <- "https://dl.dropboxusercontent.com/u/8082731/datasets/UMICH-SI650/training.txt"

test_data_file <- getURL(test_data_url)
train_data_file <- getURL(train_data_url)

train_data_df <- read.csv(
    text = train_data_file, 
    sep='\t', 
    header=FALSE, 
    quote = "",
    stringsAsFactor=F,
    col.names=c("Sentiment", "Text"))
test_data_df <- read.csv(
    text = test_data_file, 
    sep='\t', 
    header=FALSE, 
    quote = "",
    stringsAsFactor=F,
    col.names=c("Text"))
```

Here, we convert Sentiment to factor. 

``` R
train_data_df$Sentiment <- as.factor(train_data_df$Sentiment)
```

In R we will use the tm package for text mining, so we'll be using it to create a corpus. First we use all the data to get all possible words in our corpus. Then we create a VCorpus object that's essentiallya collection of content and metadata objects.

``` R
corpus <- Corpus(VectorSource(c(train_data_df$Text, test_data_df$Text)))
```

We want our data to be as clean as possible before sending it out for training - so we apply techniques like removing punctionation, stop words, and white space to make our data as consistent as possible. 

``` R 
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, stemDocument)
corpus <- tm_map(corpus, PlainTextDocument)
```

Next, we need a DocumentTermMatrix object for the corpus. A document matrix is a matrix containing a column for each different word in our whole corpus, and a row for each document. A given cell equals to the frequency in a document for a given term.

``` R
dtm <- DocumentTermMatrix(corpus)
```

We can take a glimpse of how it looks:
```
<<DocumentTermMatrix (documents: 40138, terms: 10)>>
Non-/sparse entries: 401380/0
Sparsity           : 0%
Maximal term length: 20
Weighting          : term frequency (tf)
```

Now we want to convert this matrix into a dataframe that we can use to train a classifier :

``` R
important_words_df <- as.data.frame(as.matrix(dtm))
colnames(important_words_df) <- make.names(colnames(important_words_df))
```

Here, we're just splitting our data into training and test, then adding them back to dataframes, and then getting rid of the original text field:
``` R
important_words_train_df <- head(important_words_df, nrow(train_data_df))
important_words_test_df <- tail(important_words_df, nrow(test_data_df))

train_data_words_df <- cbind(train_data_df, important_words_train_df)
test_data_words_df <- cbind(test_data_df, important_words_test_df)

train_data_words_df$Text <- NULL
test_data_words_df$Text <- NULL
```

In order to obtain our evaluation set, we split our dataset using sample.split from the caTools package:

``` R
set.seed(1234)
spl <- sample.split(train_data_words_df$Sentiment, .85)
```

Now we use `spl` to split our data into train and test
``` R
eval_train_data_df <- train_data_words_df[spl==T,]
eval_test_data_df <- train_data_words_df[spl==F,]
```


### 6.2 Data Analysis

Building a linear model in R requires only one function call, `glm`, so we use that to create our classifier. As a parameter, we set family to binomial to indicate that we want to use logistic regression:

``` R
log_model <- glm(Sentiment~., data=eval_train_data_df, family=binomial)
```

And as always, we now use our model on the test data:

``` R
log_pred <- predict(log_model, newdata=eval_test_data_df, type="response")
```

Using this table, we'll be able to calculate accuracy based on probability:

``` R
table(eval_test_data_df$Sentiment, log_pred>.5)
```

So then we get

```
(453 + 590) / nrow(eval_test_data_df)
```
```
0.9811853
```

This is a very good accuracy. It seems that our bag of words approach works nicely with this particular problem.


## 7.0 Final Words

This was a brief overview of Data Science and its different components. Obviously there is more to each component we went through, but this tutorial should have given you an idea of what a data problem should look like. 

### 7.1 Resources

[The Art of R Programming](https://www.dropbox.com/s/cr7mg2h20yzvbq3/The_Art_Of_R_Programming.pdf?dl=0)<br>
[R Bloggers](https://www.r-bloggers.com/) <br>
[kdnuggets](http://www.kdnuggets.com/)



### 7.2 Mini Courses

Learn about courses [here](www.byteacademy.co/all-courses/data-science-mini-courses/).

[Python 101: Data Science Prep](https://www.eventbrite.com/e/python-101-data-science-prep-tickets-30980459388) <br>
[Intro to Data Science & Stats with R](https://www.eventbrite.com/e/data-sci-109-intro-to-data-science-statistics-using-r-tickets-30908877284) <br>
[Data Acquisition Using Python & R](https://www.eventbrite.com/e/data-sci-203-data-acquisition-using-python-r-tickets-30980705123) <br>
[Data Visualization with Python](https://www.eventbrite.com/e/data-sci-201-data-visualization-with-python-tickets-30980827489) <br>
[Fundamentals of Machine Learning and Regression Analysis](https://www.eventbrite.com/e/data-sci-209-fundamentals-of-machine-learning-and-regression-analysis-tickets-30980917759) <br>
[Natural Language Processing with Data Science](https://www.eventbrite.com/e/data-sci-210-natural-language-processing-with-data-science-tickets-30981006023) <br>
[Machine Learning with Data Science](https://www.eventbrite.com/e/data-sci-309-machine-learning-with-data-science-tickets-30981154467) <br>
[Databases & Big Data](https://www.eventbrite.com/e/data-sci-303-databases-big-data-tickets-30981182551) <br>
[Deep Learning with Data Science](https://www.eventbrite.com/e/data-sci-403-deep-learning-with-data-science-tickets-30981221668) <br>
[Data Sci 500: Projects](https://www.eventbrite.com/e/data-sci-500-projects-tickets-30981330995)

