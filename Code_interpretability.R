#######################################################################
# "Interpretability/Explanability Methods of Machine Learning Models" #
######## Predicting positive response to lung cancer treatment ########
#######################################################################

library(tidyverse)
library(arsenal)
library(mice)
library(ggpubr)
library(caret)
library(ROCR) 
library(DALEX)
library(rpart)
library(pdp)
library(iml)
library(lime)

# Data 
load(file = "/Users/jenniferjara/Documents/Code R/Covid19/Code/data_tutoriel.RData")
colnames(data)[which(names(data) == "Y_sim")] <- "Treatment.response"


## Summary of radiomic features
olnames(data)[which(names(data) == "Y_sim")] <- "Treatment.response"
tab1.Treat <-
  tableby(Treatment.response ~ ., data = data, numeric.test = "kwt", cat.test = "fe")

tab2.Treat <-
  cbind(as.data.frame(summary(tab1.Treat)), "Adjusted p-value" = as.data.frame(
    summary(padjust(tab1.Treat, method = "BH")))$`p value`)

kableExtra::kable(
  tab2.Treat,
  col.names = c("", "Unfavorable (N=175)", "Favorable (N=125)", "Total (N=300)",
                "p-valeur", "p-valeur Ajustée"),
  format = "markdown",
  caption = "Description of the study population")



#######################
#### Modeling data ####
#######################

## Separation of data training and data test
set.seed(12345)
validation_index <- sample(seq(1, nrow(data), by=1), size = (nrow(data)*(2/3)), replace = F)
apprentissage <- data[validation_index, ]
test <- data[-validation_index, ]


## Training model based on train dataset
set.seed(1234)
control <- trainControl(method="repeatedcv", number=10, repeats=5,
                        savePredictions = "final", 
                        summaryFunction =  twoClassSummary, classProbs = TRUE)

tunegrid_tunning <- expand.grid(.mtry = (1:10))
rf_tunning <-  caret::train(Treatment.response ~ ., data = data,
                            method = "rf", ntrees = 1000, trControl = control, 
                            tuneGrid = tunegrid_tunning, metric = "ROC")


## Model Evaluation based on test dataset
set.seed(1234)
rf_fitted <-  caret::train(Treatment.response ~ ., data = apprentissage,
                           method = "rf", ntrees = 1000, 
                           trControl = control, tuneGrid = expand.grid(.mtry = 5),
                           metric = "ROC")

matrice_conf <- confusionMatrix(predict(object = rf_fitted, newdata = test), 
                                reference = test$Treatment.response, positive = "Yes")



#################################
#### Global Interpretability ####
#################################

## Feature Importance (FI)
explain_model <- DALEX::explain(model = rf_fitted,
                                data = apprentissage[,-which(colnames(apprentissage) 
                                                      %in% c("Treatment.response"))],
                                y = apprentissage$Treatment.response == "Yes")

set.seed(1234)
var_imp_model <- DALEX::variable_importance(explain_model, 
                                          loss_function = DALEX::loss_root_mean_square, 
                                          B = 100, type = "ratio")
plot(var_imp_model)


##  Predictions - radiomic features relationship 
pred.prob.pdp <- function(object, newdata) {
  pred <- predict(object, newdata, type = "prob")
  return(mean(pred[,2])) }

pred.prob.ice <- function(object, newdata) {
  pred <- predict(object, newdata, type = "prob")
  return(pred[,2]) }

pred.prob.ale <- iml::Predictor$new(rf_fitted, data = apprentissage, 
                                 y = "Treatment.response", class = "Yes", type = "prob")
ale_rf_var1 <- iml::FeatureEffect$new(pred.prob.ale, feature = "Standard.deviation.stat")


ggarrange(
    pdp::partial(rf_fitted, pred.var = "Standard.deviation.stat", pred.fun =pred.prob.pdp,
                plot = T, rug = TRUE, plot.engine = "ggplot", title = "PDP"),
    pdp::partial(rf_fitted, pred.var = "Standard.deviation.stat", pred.fun =pred.prob.ice,
                plot = T, rug = T, alpha = 0.3, plot.engine = "ggplot", title="ICE Plot"),
    ggplot(ale_rf_var1$results, aes(x = Standard.deviation.stat, y = .value)) + 
      geom_line(size = 0.8) +  geom_rug(sides="b") + labs(x = "Standard.deviation.stat", 
                                                     y = "ALE", title = "ALE Plot"),
    nrow = 1, ncol = 3)


## Surrogate model
set.seed(1234)
tree_surrogate_rf <- iml::TreeSurrogate$new(pred.prob.ale, maxdepth = 4)
tree_surrogate_rf$r.squared

tree_surrogate <- 
    partykit::ctree(explain_model$y_hat ~ ., 
      data = apprentissage[,-which(colnames(apprentissage) %in% c("Treatment.response"))],
      control = partykit::ctree_control(maxdepth = 4))
plot(tree_surrogate, gp = gpar(fontsize = 8), ip_args = list(abbreviate = F, id = F))



################################
#### Local Interpretability ####
################################

# We are interested in explaining the individual prediction of third patient of the test dataset 

## Local surrogate model LIME
explainer_local <- lime::lime(x = apprentissage[,-which(colnames(apprentissage) 
                                                 %in% c("Treatment.response"))], 
                              model = rf_fitted, n_bins = 4)

set.seed(1234)
lime_explain_local <- 
  lime::explain(x = test[3, -which(colnames(test)=="Treatment.response")],  
                explainer = explainer_local, # explainer on discretised data
                n_permutations = 1000,       # Permutations to create 
                dist_fun = "euclidean",      # Distance function
                kernel_width = 0.75,         # Define the neighborhood (size local region)
                n_features = 10,             # Features to best describe predicted outcomes
                feature_select = "highest_weights", # Algorithm to select features 
                labels = "Yes")
plot_features(lime_explain_local)


pred_exp <- cbind(Case = unique(lime_explain_local$case),  
                   prob_ML = unique(lime_explain_local$label_prob),    # probability predicted by rf model
                   prob_explainer = unique(lime_explain_local$model_prediction), # probability predicted by lime model
                   R2_explainer = unique(lime_explain_local$model_r2)) # R2 of lime model



## Shapley values
set.seed(1234)
shapley_values <- iml::Shapley$new(pred.prob.ale, x.interest = test[3, ])
plot(shapley_values)

phi_high_values <- shapley_values$results[with(shapley_values$results, order(-abs(phi))),] 
phi_high_values[,1:2]