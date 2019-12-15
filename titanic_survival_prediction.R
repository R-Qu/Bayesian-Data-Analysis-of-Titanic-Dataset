# Library
library(corrplot)
library(brms)
library(stringr)
library(bayesplot)
library(projpred)
library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
library(loo)
require(ggplot2)
library(titanic)

# Data preparation
fulldata <- titanic_train
testdata <- titanic_test
data <- fulldata
data$Sex <- ifelse(data$Sex=="male", 1, 0)
testdata$Sex <- ifelse(testdata$Sex=="male", 1, 0)
data$Embarked <- ifelse(data$Embarked=="C", 0, ifelse(data$Embarked=="Q", 1, 2))
testdata$Embarked <- ifelse(testdata$Embarked=="C", 0, ifelse(testdata$Embarked=="Q", 1, 2))
data = na.omit(data)
testdata = na.omit(testdata)

# Data visualization

## Age
counts <- table(data$Survived,data$Age)
barplot(table(data$Survived,data$Age),,main="Survivals in terms of age",
  xlab="Age", ylab="Number of People",legend=rownames(table(fulldata$Survived,fulldata$Age)))

## Passenger Class
counts <- table(data$Pclass,data$Survived)
barplot(counts,main="Survivals in terms of Passenger Class",
  xlab="Survival or not", ylab="Number of survivals",
  legend = rownames(counts), beside=TRUE)
  
## Gender
counts <- table(data$Sex,data$Survived)
barplot(counts,main="Survivals in terms of gender",
  xlab="Survival or not", ylab="Number of survivals",
  legend = rownames(counts), beside=TRUE)
  
# Correlation plot
pred <- c("Pclass", "Age", "Sex", "SibSp", "Parch", "Embarked")
target <- c("Survived")
formula <- paste("Survived ~", paste(pred, collapse = "+"))
p <- length(pred)
n <- nrow(data)
x = cor(data[, c(target,pred)])
corrplot(x)

# Use brms to generate stan for 3 models
TitanicSelection<- brm(Survived ~ Age + Sex + Pclass, family = bernoulli(),
    data = data, prior = set_prior('normal(0, 1000000)'),save_all_pars = T)
TitanicFull <- brm(Survived ~ Age + Sex + Pclass + SibSp + Parch + Embarked,  family = bernoulli(),
    data = data, prior = set_prior('normal(0, 1000000)'), save_all_pars = T )
TitanicHier <- brm(Survived ~ Age + Sex + (Age + Sex |Pclass), family = bernoulli(),
    data = data, prior = set_prior('normal(0, 1000000)'), save_all_pars = T )
    
    
# Model results
summary(TitanicSelection)
check_divergences(TitanicSelection$fit)
check_treedepth(TitanicSelection$fit)

summary(TitanicFull)
check_divergences(TitanicFull$fit)
check_treedepth(TitanicFull$fit)

summary(TitanicHier)
check_divergences(TitanicHier$fit)
check_treedepth(TitanicHier$fit)


#  Pareto_k checking
looSelection <- loo(TitanicSelection)
looFull <- loo(TitanicFull)
looHier <- loo(TitanicHier)
hist(looSelection$diagnostics$pareto_k, 
     main = "Diagnostic histogram of Parteto k",
     xlab = "k-values",
     ylab = "Frequency",
     freq = FALSE)

hist(looFull$diagnostics$pareto_k,
     main = "Diagnostic histogram of Parteto k",
     xlab = "k-values",
     ylab = "Frequency",
     freq = FALSE)

hist(looHier$diagnostics$pareto_k,
     main = "Diagnostic histogram of Parteto k",
     xlab = "k-values",
     ylab = "Frequency",
     freq = FALSE)

# LOO-CV leave one out cross validation  
cat("\nFull Model PSIS_LOO:" ,looFull$estimates[2], "\n");
cat("\nSelection Model PSIS_LOO:" , looSelection$estimates[2], "\n");
cat("\nHierarchical Model PSIS_LOO:" ,looHier$estimates[2], "\n\n");
loo_compare(list(looSelection, looFull, looHier))

# Sensitivity analysis
TitanicSelectionDP<- brm(Survived ~ Age + Sex + Pclass, family = bernoulli(), 
    data = data,save_all_pars = T)
TitanicFullDP <- brm(Survived ~ Age + Sex + Pclass + SibSp + Parch + Embarked, family = bernoulli(),
    data = data, save_all_pars = T )
TitanicHierDP <- brm(Survived ~ Age + Sex + (Age + Sex |Pclass), family = bernoulli(),
    data = data, save_all_pars = T )
summary(TitanicSelectionDP)
check_divergences(TitanicSelectionDP$fit)
check_treedepth(TitanicSelectionDP$fit)

summary(TitanicFullDP)
check_divergences(TitanicFullDP$fit)
check_treedepth(TitanicFullDP$fit)

summary(TitanicHierDP)
check_divergences(TitanicHierDP$fit)
check_treedepth(TitanicHierDP$fit)

loo_compare(looSelection, looFull, loo(TitanicHier), loo(TitanicSelectionDP),loo(TitanicFullDP), loo(TitanicHierDP))

