# set seed for reproducibility 
set.seed(1360)

###########################################
# Step 0: Load necessary libraries
#
library(glmnet)
library(xgboost)
library(caret)

###########################################
# Step 1: Preprocess training data
#         and fit two models
time_start = proc.time()

train <- read.csv("train.csv")

# First, remove the following imbalanced categorical variables due to the concern that an indicator variable for just a small set of samples tends to overfit those samples. 
remove.var <- c('Street', 'Utilities', 'Condition_2', 'Roof_Matl', 'Heating', 'Pool_QC', 'Misc_Feature', 'Low_Qual_Fin_SF', 'Pool_Area', 'Longitude','Latitude')

train <- train[ , -which(names(train) %in% remove.var)] 


# Second, apply winsorization on some numerical variables: compute the upper 95% quantile of that variable based on the train data, denoted by M; then replace all values in the train (and also test) that are bigger than M by M. Winsorization is the process of replacing the extreme values of statistical data in order to limit the effect of the outliers on the calculations or the results obtained by using that data.
# train data without "PID" and "Sale_Price"
train.x <- train[,!(colnames(train) %in% c("PID", "Sale_Price"))] 
# replace missing by zero
train.x$Garage_Yr_Blt[is.na(train.x$Garage_Yr_Blt)] = 0

winsor.vars <- c("Lot_Frontage", "Lot_Area", "Mas_Vnr_Area", "BsmtFin_SF_2", "Bsmt_Unf_SF", "Total_Bsmt_SF", "Second_Flr_SF", 'First_Flr_SF', "Gr_Liv_Area", "Garage_Area", "Wood_Deck_SF", "Open_Porch_SF", "Enclosed_Porch", "Three_season_porch", "Screen_Porch", "Misc_Val")

quan.value <- 0.95
for(var in winsor.vars){
  tmp <- train.x[, var]
  myquan <- quantile(tmp, probs = quan.value, na.rm = TRUE)
  tmp[tmp > myquan] <- myquan
  train.x[, var] <- tmp
}

# Third, handle categorical features. When fitting a linear regression model, for categorical variables with K levels, we usually generate (K-1) binary dummy variables. However, for tree models, we need to generate K binary categorical variables when K>2.
categorical.vars <- colnames(train.x)[
  which(sapply(train.x,
               function(x) mode(x)=="character"))]
train.matrix <- train.x[, !colnames(train.x) %in% categorical.vars, 
                        drop=FALSE]
n.train <- nrow(train.matrix)
for(var in categorical.vars){
  mylevels <- sort(unique(train.x[, var]))
  m <- length(mylevels)
  m <- ifelse(m>2, m, 1)
  tmp.train <- matrix(0, n.train, m)
  col.names <- NULL
  for(j in 1:m){
    tmp.train[train.x[, var]==mylevels[j], j] <- 1
    col.names <- c(col.names, paste(var, '_', mylevels[j], sep=''))
  }
  colnames(tmp.train) <- col.names
  train.matrix <- cbind(train.matrix, tmp.train)
}

####### Train two models
# Use Lasso with lamda.min to select variables and then fit a Ridge regression model on the selected variables using lambda.min.
train.y <- train[,"Sale_Price"]
train.y <- log(as.matrix(train.y))
# Train lasso/ridge model on train split
cv.out <- cv.glmnet(as.matrix(train.matrix), train.y, alpha = 0.2)

# XGBoost
# Function to train xgboost model with `train` data
train_model_xgb = function(train.matrix, train.y){
  params <- list(
    colsample_bytree=0.2,
    gamma=0.0,
    eta=0.01,
    max_depth=4,
    min_child_weight=1.5,
    alpha=0.9,
    lambda=0.6,
    subsample=0.2,
    seed=1360
  )
  
  xgb.model <- xgboost(data = as.matrix(train.matrix), 
                       label = train.y, nrounds = 10000,
                       params = params,
                       verbose = FALSE)
  return(xgb.model)
}

# Train xgboost model on train split
# Model object  are returned from train_model_xgb
trained_model_xgb = train_model_xgb(train.matrix, train.y)



###########################################
# Step 2: Preprocess test data
#         and output predictions into two files
#
test <- read.csv("test.csv")
# Removing remove the following imbalanced categorical variables:   
test <- test[ , -which(names(test) %in% remove.var)]  

# winsorization
# test data without "PID" and "Sale_Price"
test.x <- test[,!(colnames(test) %in% c("PID", "Sale_Price"))] 
# replace missing by zero
test.x$Garage_Yr_Blt[is.na(test.x$Garage_Yr_Blt)] = 0

winsor.vars <- c("Lot_Frontage", "Lot_Area", "Mas_Vnr_Area", "BsmtFin_SF_2", "Bsmt_Unf_SF", "Total_Bsmt_SF", "Second_Flr_SF", 'First_Flr_SF', "Gr_Liv_Area", "Garage_Area", "Wood_Deck_SF", "Open_Porch_SF", "Enclosed_Porch", "Three_season_porch", "Screen_Porch", "Misc_Val")

quan.value <- 0.95
for(var in winsor.vars){
  tmp <- test.x[, var]
  myquan <- quantile(tmp, probs = quan.value, na.rm = TRUE)
  tmp[tmp > myquan] <- myquan
  test.x[, var] <- tmp
}

# handle with categorical features
categorical.vars <- colnames(test.x)[
  which(sapply(test.x,
               function(x) mode(x)=="character"))]
test.matrix <- test.x[, !colnames(test.x) %in% categorical.vars, 
                      drop=FALSE]
n.test <- nrow(test.matrix)
for(var in categorical.vars){
  mylevels <- sort(unique(train.x[, var]))
  m <- length(mylevels)
  m <- ifelse(m>2, m, 1)
  tmp.test <- matrix(0, n.test, m)
  col.names <- NULL
  for(j in 1:m){
    tmp.test[test.x[, var]==mylevels[j], j] <- 1
    col.names <- c(col.names, paste(var, '_', mylevels[j], sep=''))
  }
  colnames(tmp.test) <- col.names
  test.matrix <- cbind(test.matrix, tmp.test)
}


# find out different columns in train.matrix and test.matrix
# library(waldo)
# compare(names(train.matrix), names(test.matrix))
# Make sure the column names of test.matrix are the same, including the order, as the column names of train.matrix
# drop columns (which correspond to new levels) and add all zero columns (which correspond to levels in train but not in test)
# remove.var1 <- c('MS_SubClass_One_and_Half_Story_PUD_All_Ages', 
#                  'Neighborhood_Landmark', 'Exterior_1st_PreCast',
#                  'Exterior_2nd_PreCast', 'Mas_Vnr_Type_CBlock',
#                  'Kitchen_Qual_Poor','Sale_Type_VWD')
# test.matrix1 <- test.matrix[, !colnames(test.matrix) %in% remove.var1]
# 
# remove.var2 <- c('MS_Zoning_A_agr', 'Lot_Config_FR3', 'Condition_1_RRNn',
#                   'Exterior_1st_ImStucc', 'Exterior_2nd_Other', 'Heating_QC_Poor',
#                  'Heating_QC_Poor', 'Electrical_Mix', 'Functional_Sal', 'Functional_Sev',
#                  'Electrical_Unknown'
#                  )
# train.matrix1 <- train.matrix[, !colnames(train.matrix) %in% remove.var2]

# compare(names(train.matrix1), names(test.matrix1))



# make predictions on test split from trained lasso/ridge model
tmp <- predict(cv.out, s = cv.out$lambda.min, newx = as.matrix(test.matrix))

mysubmission1 <- data.frame(PID = test[,"PID"], Sale_Price = exp(tmp))
colnames(mysubmission1) <- c('PID','Sale_Price')
write.table(mysubmission1, file = "mysubmission1.txt", sep = ",", row.names = FALSE)


# Function to make predictions on test data from xgboost model
make_pred_xgb = function(xgb.model, test.matrix, test){
  tmp = predict(xgb.model, data.matrix(test.matrix))
  pred_out = cbind(test$PID, exp(tmp))
  colnames(pred_out) = c("PID","Sale_Price")
  write.table(pred_out, "mysubmission2.txt", sep = ",", row.names = FALSE)
}
# make predictions on test split from trained xgboost model
make_pred_xgb(trained_model_xgb, test.matrix, test)

proc.time() - time_start
