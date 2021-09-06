
library(psych)
library(caTools)
library(dplyr)
library(caret)
library(ggplot2)
library(ggthemes)
library(ROCR)
library(car)
library(MASS)
library(gridExtra)
library(data.table)
library(scales)

setwd("~/Documents/PhD2/project")
newdata=read.csv('newdata.csv', header=TRUE)
data_train=read.csv('data_train.csv', header=TRUE)
data_test=read.csv('data_test.csv', header=TRUE)
psych::describeBy(newdata, newdata$bullied)

glm.fits=glm(bullied~height+weight+frstgr_age+age+hhincome+hhsize+mother_age+skipped_gr,
             data=data_train, family=binomial)
summary(glm.fits) 

# prediction
data_train$prediction=predict(glm.fits, newdata = data_train, type = "response" )
data_test$prediction=predict(glm.fits, newdata = data_test , type = "response" )
glm.pred=ifelse(data_test$prediction > 0.5, 1, 0)
table(glm.pred, data_test$bullied)
mean(glm.pred == data_test$bullied)

# finding the cutoff for the imbalanced data
# user-defined different cost for false negative and false positive
source("Additional code.R")
cm_info=ConfusionMatrixInfo( data = data_test, predict = "prediction", 
                             actual = "bullied", cutoff = 0.5 )
cost_fp=100
cost_fn=200
roc_info=ROCInfo( data = cm_info$data, predict = "predict", 
                  actual = "actual", cost.fp = cost_fp, cost.fn = cost_fn )
grid.draw(roc_info$plot)

# plot the confusion matrix plot with the cutoff value
cm_info=ConfusionMatrixInfo( data = data_test, predict = "prediction", 
                                actual = "bullied", cutoff = roc_info$cutoff )

cm_info$plot

# LDA
lda.fit=lda(bullied~height+weight+frstgr_age+age+hhincome+hhsize+mother_age+skipped_gr, 
            data=data_train)
lda.fit
lda.pred=predict(lda.fit, newdata=data_test) 
lda.class=lda.pred$class
table(lda.class, data_test$bullied) 
accuracy.lda=mean(lda.class == data_test$bullied)
accuracy.lda

# QDA
qda.fit=qda(bullied~height+weight+frstgr_age+age+hhincome+hhsize+mother_age+skipped_gr, 
            data=data_train)
qda.fit
qda.pred=predict(qda.fit, newdata=data_test) 
qda.class=qda.pred$class 
table(qda.class, data_test$bullied) 
accuracy.qda=mean(qda.class == data_test$bullied)
accuracy.qda

# create predictions when cutoff=0.21 taken from the logistic model above
lda.pred.adj = ifelse(lda.pred$posterior[, 2] > .27, 1, 0)
qda.pred.adj = ifelse(qda.pred$posterior[, 2] > .27, 1, 0)

# create new confusion matrices
list(LDA_model = table(lda.pred.adj, data_test$bullied),
     QDA_model = table(qda.pred.adj, data_test$bullied))
accuracy.lda=mean(lda.pred.adj == data_test$bullied)
accuracy.lda
accuracy.qda.adj=mean(qda.pred.adj == data_test$bullied)
accuracy.qda

#linear discrimininant
par(mfrow = c(1,2))

# Evaluate the model
pred=prediction(lda.pred$posterior[,2], data_test$bullied)
roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
auc = performance(pred, measure = "auc")
auc = auc@y.values
# Plot
plot(roc.perf, main = 'ROC curve of LDA')
abline(a=0, b= 1)
text(x = .25, y = .65 ,paste("AUC = ", round(auc[[1]],3), sep = ""))

#nonlinear discriminant model
pred=prediction(qda.pred$posterior[,2], data_test$bullied)
roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
auc = performance(pred, measure = "auc")
auc = auc@y.values
# Plot
plot(roc.perf, main = 'ROC curve of QDA')
abline(a=0, b= 1)
text(x = .25, y = .65 ,paste("AUC = ", round(auc[[1]],3), sep = ""))
