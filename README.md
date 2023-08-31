# FoodDeliveryApp
Developed models that predicts customer behavior and applied it to the rest of the customer base. It maximized the profit of the campaign

## 1. Introduction  
Marketing analytics already involves a wide range of data collection and transformation techniques. Social media and
web driven marketing have given a big push in the digitalization of the space; counting the number of visits, the 
number of likes, the minutes of viewing, the number of returning customers, and so on is common practice. However,
we can move one level up and apply machine learning and statistics algorithms to the available data to get a better 
picture of not just the current but also the future situation.We wanted something classification to be able to use the
decision tree tool because it is something that we haven’t done before. Also several of us are in the marketing 
concentration and we have not gotten to apply a lot of analysis to marketing in our courses. Some of the reasons we
chose marketing would be to diversify our background in education because some of us have backgrounds in different 
subjects like finance, information systems, and statistics, we would like to add to that portfolio. Since our data 
set is also used as a test in the hiring process of a company, it can give us experience as to what to expect from 
other companies who are also looking to hire analysts. 
                                 
### 1.2. Background
iFood is a leading food delivery app in Brazil that mainly sells 5 categories of products through 3 sales channels
which can be seen in the following variables. Because of the uncertainty of the future profit growth, our project
focuses on improving the marketing efficiency of iFood. We will explore the characteristic features of customers, 
and make a customer segmentation of customer behaviors.
                                 
### 1.3. Objectives
The first objective of this project is to predict whether a customer will accept the last offer (“Response”). The second
objective of this project is to find which are more powerful, the channel choices or the product choices Aamong the
behavioral variables?

## 2. Data and Methods
### 2.1. Dataset description
The dataset gives an insight into the past acceptance of advertisement campaigns, the result of the most recent ad 
campaign in connection with certain demographic information and some behavioral characteristics of the customer group. 
Whether the consumers have accepted the most recent offer of the campaign is the target variable, whereas the remaining
variables are the explanatory variables.   

| **Variable** | **Description** |
| --- | ----- |
| AcceptedCmp1 | 1 if customer accepted the offer in the 1st campaign, 0 otherwise |
| AcceptedCmp2 | 1 if customer accepted the offer in the 2nd campaign, 0 otherwise |
| AcceptedCmp3 | 1 if customer accepted the offer in the 3rd campaign, 0 otherwise |
| AcceptedCmp4 | 1 if customer accepted the offer in the 4th campaign, 0 otherwise |
| AcceptedCmp5 | 1 if customer accepted the offer in the 5th campaign, 0 otherwise |
| Response(target) | 1 if customer accepted the offer in the last campaign, 0 otherwise |
| Complain | 1 if customer complained in the last 2 years |
| DtCustomer | data of the customer's enrollment with the company |
| Education | customer's level of education |
| Marital | customer's marital status |
| Kidhome | number of small children in the customer's household |
| Teenhome | number of teenagers in customer's household |
| Income | customer's yearly household income |
| MntFishProducts | amount spend on fish products in the last 2 years |
| MntMeatProducts | amount spend on meat products in the last 2 years |
| MntFruits | amount spend on fruits products in the last 2 years |
| MntSweetProdcuts | amount spend on sweet products in the last 2 years |
| MntWines | amount spend on wine products in the last 2 years |
| MntGoldProds | amount spend on gold products in the last 2 years |
| NumDealsPurchases | number of purchases made with discount |
| NumCatalogPurchases | number of purchases made using catalog |
| NumStorePurchases | number of purchases made directly in stores |
| NumWebPurchases | number of purchases made through the company's website |
| NumWebVisitsMonth | number of visits to the company's web site in the last month |
| Recency | number of days since the last purchase |
 
### 2.2. Exploratory Data Analysis
#### 2.2.1. Data processing
setwd("C:/Users/yuanb/Downloads")

ifood <- read.csv("ifood_df.csv")
ifood

require(caret)
#install.packages("knn")
library(leaps)
library(forecast)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(MASS)
library(InformationValue)
library(ISLR)
library(tidyverse)
library(randomForest) # Random Forests
#install.packages("mlbench")
library(mlbench)
library(e1071)
#library(knn)
library(class)

table(ifood$AcceptedCmpOverall)
table(ifood$AcceptedCmp1)
table(ifood$AcceptedCmp2)
table(ifood$AcceptedCmp3)
table(ifood$AcceptedCmp4)
table(ifood$AcceptedCmp5)
table(ifood$Response)

nrow(ifood)
ncol(ifood)

#### 2.2.2. Subset selection
We did a best subset of variables selection for make the better model predictive performace.
### GLM FIT
We started by fitting a logistic regression model with all predictors in the dataset.
glm.fit.1 <- glm(Response ~ . , 
                 data = ifood, family = binomial(link = logit)) 
summary(glm.fit.1)

### BACKWARDS SELECTION
We impleted a backward selection.
glm.fit.1%>%
  stepAIC(trace=FALSE, direction="backward") #Backward selection for variables

#### 2.2.3. Null values checking
which(is.na(ifood))

#### 2.2.4. Data Visualization 
To visualize the relationship between the significant predicors and response variable
library(ggplot2)

#Income
ggplot(data = ifood)+
  geom_density(kernel = "gaussian", mapping = aes(x = Income))+
  facet_wrap(~Response)

#TEEN AT HOME
mosaicplot(Response~Teenhome,data=ifood, col=c("lightblue","lightpink"))

#AcceptedCmp1
ggplot(data = ifood)+
  geom_density(kernel = "gaussian", mapping = aes(x = AcceptedCmp1))+
  facet_wrap(~Response)


#KID AT HOME
mosaicplot(Response~Kidhome,data=ifood, col=c("lightblue","lightpink"))
  
#ACCEPTED CAMPAIGN 3
mosaicplot(Response~AcceptedCmp3,data=ifood, col=c("lightblue","lightpink"))


#RECENCY
ggplot(data = ifood)+
  geom_density(kernel = "gaussian", mapping = aes(x = Recency))+
  facet_wrap(~Response)


#MARITAL STATUS
mosaicplot(Response~marital_Married,data=ifood, col=c("lightblue","lightpink"))

#EDUCATION
mosaicplot(Response~education_Basic,data=ifood, col=c("lightblue","lightpink"))

## REMOVE CONSTANTS
ifood <- dplyr::select(ifood, -c(Z_Revenue, Z_CostContact))
####

### 2.3. RANDOM FOREST
To predict whether a customer will accept the last offer, our dataset is categorical varaibles, so we started by fitting
a random forest model.

# Piciking the best m
We wrote a for loop that tries m = 1, 2, 3, ... , 10 for Random Forest and calculate the average out of bag error rates. 
Use ntree = 500.
ifood$Response <- as.factor(ifood$Response)
start.time <- Sys.time()
results <- double(10)
for(m in 1:10){
  model.Tree <- randomForest(Response ~ ., data = ifood, mtry = m, ntree = 500)
  results[m] <- mean(model.Tree$err.rate[,1])
}

end.time <- Sys.time()

end.time - start.time

# Visulasiation
We plotted m = 1, 2, ..., 10 and the out of bag error rates.
tibble(m = 1:10, error = results)%>%
  ggplot(aes(x = m , y = error)) +
  geom_point()+
  geom_smooth(se=FALSE)+
  theme_bw()

# We selected the value of m=6 that minimizes the out of bag error rates, created a single randomForest model.
model.RandomForest <- randomForest(Response ~ ., data = ifood, mtry = 6, ntree = 500)

### 2.4. Support Vector Machines
## Kernel selection/SVM TUNE
To Support Vector Machines requires choosing a kernel then a number of additional tuning parameters depending
on the kernel. In this step, we used the tune function to fit two different kernels with these tuning parameters.

start.time <- Sys.time()
svm.tune.poly <- tune(svm, Response ~ ., data = ifood, kernel = "polynomial",ranges = list(cost= seq(1,5,0.5), d = seq(1,4, 1)))

svm.tune.sig <- tune(svm, Response ~., data = ifood, kernel = "sigmoid", ranges = list(cost = seq(1,5,0.5)))
end.time <- Sys.time()
end.time - start.time

## Choose best model
We used the summary() function to find the "best performance" of both the polynomial vs Sigmoid models.

summary(svm.tune.poly)
summary(svm.tune.sig)

# After checking to see if the polynomial model or Sigmoid model performed better, save the best model here
model.SVM <- svm(Response ~ ., data = ifood, kernel = "polynomial",ranges = list(cost= seq(1,5,0.5), d = seq(1,4, 1)))

### 2.5. Choosing between Random Forest and Support Vector Machines
Because the out of bag error rates is not comparable to the cross validation used in SVM, we created a single manual
5 fold cross validation to compare winning Random Forest Model with the winning SVM model.

ifood.Partitioned <-
  ifood %>%
  sample_frac() %>%
  dplyr::mutate(Partition = (row_number() %% 5) + 1)

results <- list()

# Random forest and SVM testing on the five partitions
for(v in 1:5){
  
  Testing <- ifood.Partitioned %>% filter(Partition == v) %>% dplyr::select(-Partition)
  
# No need for training data since we already have two trained models
  Test.Predict <- Testing %>% 
    mutate(Predict_RandomForest = predict(model.RandomForest,newdata=.)) %>% #Generate Random Forest Predictions
    mutate(Predict_SVM = predict(model.SVM,newdata=.)) #Generate Random Forest Predictions
  
# Calculate the accuracy on both Random Forest and SVM
  RandomForest.Table <- Test.Predict %>% dplyr::select(Response, Predict_RandomForest) %>% table()
  SVM.Table<-Test.Predict %>% dplyr::select(Response, Predict_SVM) %>% table()
  
  accuracy.RandomForest <- (RandomForest.Table[1,1]+RandomForest.Table[2,2])/sum(RandomForest.Table)
  accuracy.SVM <- (SVM.Table[1,1]+SVM.Table[2,2])/sum(SVM.Table)
  
  
# No need to change below as long as you used the names from above  
  results <- results %>% bind_rows(tibble(Partition = v, Accuracy.RandomForest = accuracy.RandomForest, Accuracy.SVM = accuracy.SVM))
  
}
# We got the acuracy of randomForest is 99.12%, the accuracy of SVM is 93.5%.
                                                                                                   
### 2.5. k Nearest Neighbors

# method 1 - tune K

normalize <- function(x){return ((x - min(x)) / (max(x) - min(x)))}
ifood_norm <- as.data.frame(sapply(ifood, normalize))
which(colSums(is.na(ifood_norm))>0)

ifood_norm <- subset(ifood_norm, select=-c(Z_CostContact, Z_Revenue))
ifood_norm

set.seed(208)
index <- sample(nrow(ifood_norm), nrow(ifood_norm)*0.75, replace = FALSE)

train <- ifood_norm[index,]
test <- ifood_norm[-index,]

ctrl <- trainControl(method = "cv", number = 100)
grid <- expand.grid(.k = c(1:80))


train_labels <- train[,'Response']
test_labels <-test[,'Response']

i=1
k.optm=1
for (i in 1:80){
  knn.mod <- knn(train=train, test=test, cl=train_labels, k = i)
  k.optm[i] <- 100 * sum(test_labels == knn.mod)/NROW(test_labels)
  k=i
  cat(k,'=', k.optm[i],'')
}

plot(k.optm, type="b", xlab="K-Value", ylab="Accuracy level")


knn.3 <- knn(train=train, test=test, cl=train_labels, k=3)
acc.3 <- 100 * sum(test_labels == knn.3)/NROW(test_labels)
acc.3
table(knn.3 ,test_labels)

# method 2 - cross validation

col <- names(ifood)
col <- col[-c(1,5:10,25,26,37,38)]
ifood[, col] <- lapply(col, function(x) as.factor(ifood[[x]]))

ifood <- subset(ifood, select=-c(Z_CostContact, Z_Revenue))

set.seed(208)
index <- sample(nrow(ifood), nrow(ifood)*0.75, replace = FALSE)

train <- ifood[index,]
test <- ifood[-index,]

myctrl3 <- trainControl(method="cv", number = 10)

mygrid <- expand.grid(.k=c(1:10))
KNN_fit <- train(Response ~ ., data=ifood, method = "knn", trControl=myctrl3, tuneGrid=mygrid)
summary(KNN_fit)

predictedotocopy <- predict(KNN_fit, ifood, type="prob")

predictedotoconfusion <- predictedotocopy$`1`

actual <- ifood$Response
confusionMatrix(actual,predictedotoconfusion)
#confusionMatrix(table(actual,predictedotoconfusion))

accuracy <- (1834+86)/2205

# We got the accuracy of knn is 87.07%

### 2.6. NEURAL NETWORK

library(keras)

NeuralNetwork <- keras_model_sequential() # Create a model object we will configure

NeuralNetwork %>% 
  layer_dense(units = 100, activation= "relu") %>% # First hidden layer
  layer_dropout(rate= 0.5) %>% # Dropout Rate of 50%
  layer_dense(units = 75, activation = "relu") %>% # Second Hidden layer
  layer_dropout(rate = 0.3) %>% # Dropout rate of 30%
  layer_dense(units = 2, activation = "softmax") # Output layer for 2 classification


NeuralNetwork %>%
  compile(loss = "categorical_crossentropy", # Cross Entropy Loss / "mse" for regression
          optimizer = optimizer_rmsprop(), # Gradient Optimization Algorithm
          metrics = "accuracy") # Evaluate the model performance with accuracy

x = model.matrix(Response ~ . - 1, data = ifood) %>% scale() # Sets up an X matrix: Fixes categorical variables
y = as.numeric(ifood$Response) -1 # Changes Yes/No to 0/1
x.NN <- array_reshape(x,c(dim(x)[1],dim(x)[2])) # Formats X Data for NN
y.NN <- to_categorical(y,2) # Formats Y Data for NN

history <- 
  NeuralNetwork %>% 
  fit(x.NN,y.NN, # Plug in formatted Data
      batch_size=30, #  
      epochs = 100,
      validation_split = 0.2)

## 3. Results
# Objective 3 WHICH VARIABLES ARE MORE INFLUENTIAL
## minus channel -> product
resultsprod <- double(10)
for(m in 1:10){
  model.Tree <- randomForest(Response ~ . -NumDealsPurchases - NumWebPurchases - NumCatalogPurchases -NumStorePurchases -NumStorePurchases -NumWebVisitsMonth , data = ifood, mtry = m, ntree = 500)
  resultsprod[m] <- mean(model.Tree$err.rate[,1])
}

tibble(m = 1:10, error = resultsprod)%>%
  ggplot(aes(x = m , y = error)) +
  geom_point()+
  geom_smooth(se=FALSE)+
  theme_bw()


model.RandomForest.prod <- randomForest(Response ~ ., data = ifood, mtry = 9, ntree = 500)
model.RandomForest.prod
model.RandomForest.cha

## minus product -> channel
resultscha <- double(10)
for(m in 1:10){
  model.Tree <- randomForest(Response ~ . -MntWines -MntFruits -MntMeatProducts -MntFishProducts -MntSweetProducts -MntGoldProds -MntTotal -MntRegularProds , data = ifood, mtry = m, ntree = 500)
  resultscha[m] <- mean(model.Tree$err.rate[,1])
}

tibble(m = 1:10, error = resultscha)%>%
  ggplot(aes(x = m , y = error)) +
  geom_point()+
  geom_smooth(se=FALSE)+
  theme_bw()


model.RandomForest.cha <- randomForest(Response ~ ., data = ifood, mtry = 6, ntree = 500)

# Random forest minus channel and Random forest minus product on the five partitions
ifood.Partitioned <-
  ifood %>%
  sample_frac() %>%
  dplyr::mutate(Partition = (row_number() %% 5) + 1)

results.new <- list()

for(v in 1:5){
  
  Testing <- ifood.Partitioned %>% filter(Partition == v) %>% dplyr::select(-Partition)
  
  # No need for training data since we already have two trained models
  Test.Predict <- Testing %>% 
    mutate(Predict_RandomForest_prod = predict(model.RandomForest.prod,newdata=.)) %>% #Generate Random Forest Predictions
    mutate(Predict_RandomForest_cha = predict(model.RandomForest.cha,newdata=.)) #Generate Random Forest Predictions
  
  # Calculate the accuracy on both Random Forest and SVM
  RandomForest.Table.prod <- Test.Predict %>% dplyr::select(Response, Predict_RandomForest_prod) %>% table()
  RandomForest.Table.cha<-Test.Predict %>% dplyr::select(Response, Predict_RandomForest_cha) %>% table()
  
  accuracy.RandomForest.prod <- (RandomForest.Table.prod[1,1]+RandomForest.Table.prod[2,2])/sum(RandomForest.Table.prod)
  accuracy.RandomForest.cha <- (RandomForest.Table.cha[1,1]+RandomForest.Table.cha[2,2])/sum(RandomForest.Table.cha)
  
  
  # No need to change below as long as you used the names from above  
  results.new <- results.new %>% bind_rows(tibble(Partition = v, Accuracy.RandomForest.prod = accuracy.RandomForest.prod, Accuracy.RandomForest.cha = accuracy.RandomForest.cha))
  
}

results.new
