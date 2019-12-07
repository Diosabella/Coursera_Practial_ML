###################
####PROJECT ML#####
###################

#Setting the directory and activating the packages
#getwd()
#setwd("C:/Users/diosa/Desktop/ML/Other")
library(knitr)
library(caret)
library(rpart)
library(rpart.plot)
library(rattle)
library(randomForest)
library(corrplot)
library(tidyverse)
library(corrgram)
library(Amelia)

##################
#Loading the data#
##################

Training <- read.csv(url("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"), header=TRUE)
Testing  <- read.csv(url("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"), header=TRUE)

####################
#Splitting the data#
####################

#I split the "Training" data into the training set (to fit the model) 
#and the validation set (to evaluate the model) leaving the "Testing" set
#(for final model evaluation) intact

set.seed(2019)
Index  <- createDataPartition(Training$classe, p=0.7, list=FALSE)
training <- Training[Index, ]
validation  <- Training[-Index, ]
dim(training)
dim(validation)

###################
#Cleaning the data#
###################

#First, I identify near zero variance variables in the dataset
#and remove them

nearZeroVar(training)
near_zero_variance <- nearZeroVar(training)
training <- training[, -near_zero_variance]
validation  <- validation[, -near_zero_variance]
dim(training)
dim(validation)

#Second, I remove all NA values and first 5 columns of the dataset
#representing ID values

remove  <- sapply(training, function(x) mean(is.na(x))) > 0.95
training <- training[, remove==FALSE]
validation  <- validation[, remove==FALSE]

training <- training[, -(1:5)]
validation  <- validation[, -(1:5)]
dim(training)
dim(validation)

#Thus I reduce the dataset for the analysis to the 54 variables only

###################
#Visualising data before starting modeling#
###################
# correlation among all of these variables will be visualized by
#corrplot and corrgram to get an idea about how variables are related
#before corrplot, non numeric varibles will be excluded
sum(sapply(training[-54], is.numeric) == FALSE)

#there is no non numeric column
corMatrix <- cor(training[, -54])
corrplot(corMatrix, order = "FPC", method = "color", type = "lower", 
         tl.cex = 0.8, tl.col = rgb(0, 0, 0), tl.srt = 1)
corrgram(training,order=TRUE, lower.panel=panel.shade,
         upper.panel=panel.pie, text.panel=panel.txt)

ggplot(training,aes(x=classe)) + geom_histogram(stat = "count",alpha=0.5,fill='blue') + theme_minimal()



###################
#Build Models#
###################

# three models will be used that are used for predicting categorical data
# these are
#1. Decision trees with CART (rpart)
#2. Stochastic gradient boosting trees (gbm)
#3. Random forest decision trees (rf)

#1. Decision trees with CART (rpart)
# This model has been choosen recursive partitioning architecture 
# performs better when response variable is categorical


#fitting model
set.seed(2019)
mod_DT <- rpart(classe ~ ., data=training, method="class")
fancyRpartPlot(mod_DT)

#predicting 
predict_DT <- predict(mod_DT, newdata=validation, type="class")
conf_Mat_DT <- confusionMatrix(predict_DT, validation$classe)
conf_Mat_DT

#ploting accuracy
plot(conf_Mat_DT$table, col = conf_Mat_DT$byClass, 
     main = paste("Decision Tree (rpart) - Accuracy =",
                  round(conf_Mat_DT$overall['Accuracy'], 3)))




#2. Stochastic gradient boosting trees (gbm)
# This model is a combination of decision tree and boosting model and frequently used for 
# prediction of categorical variable
set.seed(2019)
ctl_GBM <- trainControl(method = "cv", number = 5)
mod_GBM  <- train(classe ~ ., data=training, method = "gbm",
                    trControl = ctl_GBM, verbose = FALSE)
mod_GBM$finalModel

predict_GBM <- predict(mod_GBM, newdata=validation)
conf_Mat_GBM <- confusionMatrix(predict_GBM, validation$classe)
conf_Mat_GBM


plot(conf_Mat_GBM$table, col = conf_Mat_GBM$byClass, 
     main = paste("GBM - Accuracy =", round(conf_Mat_GBM$overall['Accuracy'], 3)))


#3. Random forest decision trees (rf)



set.seed(2019)
ctr_rf <- trainControl(method="cv", number=3, verboseIter=FALSE)
mod_RF <- train(classe ~ ., data=training, method="rf",
                          trControl=ctr_rf)
mod_RF$finalModel

predict_RF <- predict(mod_RF, newdata=validation)
conf_Mat_RF <- confusionMatrix(predict_RF, validation$classe)
conf_Mat_RF

plot(conf_Mat_RF$table, col = conf_Mat_RF$byClass, 
     main = paste("Random Forest - Accuracy =",
                  round(conf_Mat_RF$overall['Accuracy'], 4)))
# Random forest decision tree exhibits most accuracy so this model will be used on Testing data

predict_RF_Testing <- predict(mod_RF, newdata=Testing)
predict_RF_Testing



