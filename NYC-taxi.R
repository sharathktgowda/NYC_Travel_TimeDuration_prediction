#all the libraries required
library(tidyverse)
library(dplyr)
library(mlr)
library(xgboost)
library(knitr)
library(lubridate)
library(geosphere)
#libraries for Visualization
library(ggplot2)
library(ggthemes)

setwd("D:\\dataset\\DMML\\")
getwd()
trip<-read.csv("nyc_taxi_trip_duration.csv",header = T, stringsAsFactors = T)
head(trip)
summary(trip)

Strips<-trip[1:50000,]

kable(head(Strips,10))

ft<-Strips
str(ft)


#Preporcessing 
#checking is any NA values
colSums(is.na(Strips))

ft$pickup_datetime<-as.character(ft$pickup_datetime)
ft$dropoff_datetime<-as.character(ft$dropoff_datetime)

#feature engineering
ft$pickup_month <- month(ft$pickup_datetime)
ft$pickup_day <- day(ft$pickup_datetime)
ft$pickup_hour <- hour(ft$pickup_datetime)
ft$pickup_min <- minute(ft$pickup_datetime)

ft$dropoff_month <- month(ft$dropoff_datetime)
ft$dropoff_day <- day(ft$dropoff_datetime)
ft$dropoff_hour <- hour(ft$dropoff_datetime)
ft$dropoff_min <- minute(ft$dropoff_datetime)

ft$trip_duration_min <- ft$trip_duration/60

ft$weekday <- wday(ft$pickup_datetime)
ft$weekend <- cut(ft$weekday,breaks=c(0,5,7),labels= c("weekday","weekend"))

ft$rushhour <- cut(ft$pickup_hour, breaks=c(-1,6,9,15,17,24),
                     labels= c("late_night","morning_rush","day","evening_rush","night"))



#Visuals


ggplot(data=ft, aes(x=trip_duration)) + 
  geom_histogram(bins=50000, fill="green")+
  theme_bw()+
  theme(axis.title = element_text(size=10),axis.text = element_text(size=10))+
  ylab("Number of Trips")+coord_cartesian(x=c(0,9000))+
  ggtitle("Trip duration (Target Variable) histogram ")

#feature engineering
#We can get useful value from extracting the distance value from the pickup and dropoff coordinates. 
#We will use the distHaversine function from the geosphere package.

ft$distance <- (distHaversine(matrix(c(ft$pickup_longitude, ft$pickup_latitude), ncol = 2),
                              matrix(c(ft$dropoff_longitude,ft$dropoff_latitude), ncol = 2))/1000)

ggplot(ft, aes(distance, trip_duration)) +
  geom_point()+
  ylab("trip_duration in seconds")+
  xlab("distnace in km")+
  coord_cartesian(x=c(0,50))+
  coord_cartesian(y=c(0,8000))+
  geom_smooth(method = "loess") +
  ggtitle("Trip duration vs distance - Scatterplot")

# Split Train and Test
X<-sample(2,nrow(ft), replace = T, prob = c(0.70,0.30))#split into ratio of 70% and 30%
train<-Strips[X==1,]
test<-Strips[X==2,]
nrow(train)
nrow(test)


#Convert data.frame/tibble to DMatrix

traindata_x <- train %>% select(-id, -trip_duration)
traindata_y <- log(train$trip_duration)

testdata_x <- test %>% select(-id, -trip_duration)


dtrain <- xgb.DMatrix(data=data.matrix(traindata_x), label=traindata_y)
dtest <- xgb.DMatrix(data=data.matrix(testdata_x))
watchlist <- list(traindata=dtrain)


#Setting up Hyperparamters for XGBoost

# Set nfolds = 3, nrounds = 100
nfolds <- 3
nrounds <- 100



params <- list("eta"=0.3,
               "max_depth"=10,
               "booster" = "gbtree",
               "colsample_bytree"=0.3,
               "min_child_weight"=1,
               "subsample"=0.8,
               "eval_metric"= "rmse", 
               "objective"= "reg:squarederror")


model_xgb <- xgb.train(params=params,
                       data=dtrain,
                       nrounds=nrounds,
                       maximize=FALSE,
                       watchlist=watchlist,
                       print_every_n=3)


#predicting the model
predicted <- predict(model_xgb, dtest)
summary(predicted)
head(predicted)
predicted<-exp(predicted)

importance_matrix <- xgb.importance(colnames(traindata_x), model = model_xgb)
library(Ckmeans.1d.dp)
xgb.ggplot.importance(importance_matrix, top_n = 15, measure = "Gain")

library(pROC) 
str(test$trip_duration)
str(predicted)
test$trip_duration<-as.numeric(test$trip_duration)

roc_test <- roc(test$trip_duration, predicted, algorithm = 2)
plot(roc_test) 
auc(roc_test)



