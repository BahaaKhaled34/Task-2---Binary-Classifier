#Set the working directory to the folder that contains the dataset files.
#Loading necessary Libraries.
library(lubridate)
library(ROCR)
library(corrplot)
library(fastDummies)
library(MASS)
library(car)

################################################################################
#                               DATA CLEANING
################################################################################

#Cleaning the workspace.
rm(list=ls())

#Reading the data from the .csv files.
training_data <- read.table("training.csv", header=T, sep = ';', na.strings = c("", "NA"), fill=T)
test_data <- read.table("validation.csv", header=T, sep = ';', na.strings = c("", "NA"), fill=T)

#Removing rows with missing data.
training_data <- training_data[!is.na(training_data$removed2), ]
test_data <- test_data[!is.na(test_data$removed2), ]

#Removing column 20 which is full of NA's, and 22 which is redundant.
training_data <- subset(training_data, select = -c(removed1, removed2))
test_data <- subset(test_data, select = -c(removed1, removed2))

#Removing Rows with NA's.
training_data <- na.omit(training_data)
test_data <- na.omit(test_data)

#Merging the datasets to process the data all at once then split later.
data <- rbind(training_data, test_data)

# Categorical data to transform to n-1 variables
CatArray <- c("variable1", "variable6", "variable7", "variable8", "variable9", "variable16")

#Set t and f columns to 1 or 0
data$variable12 <- as.numeric(ifelse(data$variable12 == "t" , 1, 0))
data$variable13 <- as.numeric(ifelse(data$variable13 == "t" , 1, 0))
data$variable15 <- as.numeric(ifelse(data$variable15 == "t" , 1, 0))

# Transform categorical data to n-1 variables
data <- dummy_cols(data, select_columns = CatArray, remove_first_dummy = T, remove_selected_columns = T)
data <- data[,c(1:13, 15:39, 14)]

#converting all non-numeric data into numeric
data <- transform(data, variable4 = as.numeric(variable4),
                                 variable5 = as.numeric(variable5),
                                 variable10 = as.numeric(variable10),
                                 variable11 = as.numeric(variable11),
                                 variable14 = as.numeric(variable14),
                                 variable17 = as.numeric(variable17),
                                 variable18 = as.numeric(variable18),
                                 variable19 = as.numeric(variable19),
                                 classLabel = as.numeric(classLabel))

#Splitting the data into training and test sets after processing.
training_data <- data[1:1955,]
test_data <- data[1956:2077,]

#Calculating and plotting correlation matrix.
corMat <- cor(training_data[,1:38])
corrplot.mixed(corMat, upper = "ellipse", tl.cex = 0.40, tl.pos = 'd')

#Removing highly correlated variables.
training_data <- subset(training_data, select = -c(variable19, variable7_p, variable9_z, variable9_ff))
test_data <- subset(test_data, select = -c(variable19, variable7_p, variable9_z, variable9_ff))

#Calculating and plotting correlation matrix again after removing the correlated variables.
corMat <- cor(training_data[,1:34])
corrplot.mixed(corMat, upper = "ellipse", tl.cex = 0.40, tl.pos = 'd')

training_data <- as.data.frame(training_data)
test_data <- as.data.frame(test_data)

################################################################################
#                               CLASSIFICATION
################################################################################

#Logistic Regression.
basicModel <- glm(classLabel ~.,
                  data=training_data, binomial(link = "logit"))
summary(basicModel) 

model1 <- stepAIC(basicModel, direction = "both")
vsummary(model1)

#Removing multicollinearity through VIF check.
vif(model1)

#Creating another model after removing collinear columns.
model2 <- glm(formula = classLabel ~ variable3+variable4+variable10+variable12
              +variable14+variable18+variable8_c+variable8_d+variable8_i+variable8_q
              +variable9_j+variable16_s,data = training_data,
              family=binomial(link = "logit"), 
              na.action = na.pass)
summary(model2)

model2 <- stepAIC(model2, direction = "both")
summary(model2)

pred = predict(model2, type="response") #This returns the probability scores on the training data.
predObj = prediction(pred, training_data$classLabel) #Prediction object needed by ROCR.

rocObj = performance(predObj, measure="tpr", x.measure="fpr")  #Creates ROC curve obj.
aucObj = performance(predObj, measure="auc")  #auc object.

auc = aucObj@y.values[[1]]
auc  #The auc score: tells you how well the model predicts.

plot(rocObj, main = paste("Area under the curve:", auc))

#Removing the labels from the validation data, then predicting the labels using model2.
test_data_unlabeled <- test_data[ , !(names(test_data) %in% c("classLabel"))]
test_data_unlabeled$classLabel <- predict (model1,newdata=test_data_unlabeled,type="response")

#Thresholding the predicted values.
test_data_unlabeled$classLabel <- factor(ifelse(test_data_unlabeled$classLabel >= 0.5, 1, 0))

#Performance Metrics:
#Calculating the confusion matrix.
confusionMatrix <- table(test_data_unlabeled$classLabel, test_data$classLabel)
confusionMatrix

#Calculating accuracy, percission, recall, and F1 score.
accuracy <- 100*(sum(diag(confusionMatrix))/sum(confusionMatrix))
accuracy

precision <- 100*(confusionMatrix[1,1]/sum(confusionMatrix[1,]))
precision

recall <- 100*(confusionMatrix[1,1]/sum(confusionMatrix[,1]))
recall

F1score <- (2*precision*recall)/(precision+recall)
F1score
