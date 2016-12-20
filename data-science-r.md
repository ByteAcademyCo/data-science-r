Intro to Data Science with R Programming
==================

Brought to you by [Lesley Cordero](http://www.columbia.edu/~lc2958) and [ADI](https://adicu.com)

## Table of Contents

- [0.0 Setup](#00-setup)
    + [0.1 R and R Studio](#01-r-and-r-studio)
- [1.0 Background](#10-background)
+ [1.1 What is Geospatial Data Analysis](#11-what-is-geospatial-data-analysis)
+ [1.2 Why is Geospatial Analysis Important?](#12-why-is-geospatial-analysis-important)
+ [1.3 Terminology](#13-terminology)
* [1.3.1 Interior Set](#131-interior-set)
* [1.3.2 Boundary Set](#132-boundary-set)
* [1.3.3 Exterior Set](#133-exterior-set)
+ [1.4 Data Types](#14-data-types)
* [1.4.1 Point](#141-point)
* [1.4.2 Polygon](#142-polygon)
* [1.4.3 Curve](#143-curve)
* [1.4.4 Surface](#144-surface)
- [2.0 Geojsonio & Geopandas](#30-geojsonio-geopandas)
+ [2.1 Geojsonio](#31-geojsonio)
+ [2.2 Geopandas](#32-geopandas)
- [3.0 Plotly](#30-plotly)
- [4.0 Shapely & Descartes](#50-Shapely-Descartes)
- [5.0 Final Words](#60-final-words)
    + [5.1 Resources](#61-resources)


## 0.0 Setup

This guide was written in R 3.2.3.

### 0.1 R and R Studio

Download [R](https://www.r-project.org/) and [R Studio](https://www.rstudio.com/products/rstudio/download/).


### 0.2 Packages


``` 
install.packages("ggvis”)
install.packages("gmodels")
```

## 1.0 Background

### 1.1  


## 2.0 Data Preparation

### 2.1 


``` R

```

### 2.2 


``` R
```
## 3.0 Exploratory Analysis

R gives you the opportunity to go more in-depth with the summary() function. This will give you the minimum value, first quantile, median, mean, third quantile and maximum value of the data set Iris for numeric data types.

``` R
summary(iris) 

```

## 4.0 Data Visualization 


ggvis allows you to make scatterplots, as with the following: 


``` R
library(ggvis)
iris %>% ggvis(~Petal.Length, ~Petal.Width, fill = ~Species) %>% layer_points()

```


## 5.0 Machine Learning & Prediction

### 5.1 k-Nearest Neighbors

The k-nearest neighbors algorithm is one of the simplest machine learning algorithms, where the distance between the stored data and a new instance of data is calculated through a similarity measure.


## 6.0 Final Words

### 6.1 Resources

[]() <br>
[]()
