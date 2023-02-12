library(lubridate)
library(tidyverse)

mypredict <- function(){
  
  train <- train %>%
    select(Store, Dept, Date, Weekly_Sales) %>%
    spread(Date, Weekly_Sales)
  train[is.na(train)] <- 0
  
  train_svd <- NULL
  d <- 8
  for(dept in unique(train$Dept)){
    dept_data <- train %>% filter(Dept == dept)
    X <- dept_data[, -c(1,2)]
    if(nrow(X) > d){
      store.mean <- rowMeans(X)
      X <- X - store.mean
      s <- svd(X)
      X <- s$u[,1:d] %*% diag(s$d[1:d]) %*% t(s$v[,1:d]) + store.mean
    }
    dept_data[, -c(1,2)] <- X
    train_svd <- bind_rows(train_svd, dept_data)
  }
  train <- train_svd %>% gather(Date, Weekly_Sales, -Store, -Dept)

  start_date <- ymd("2011-03-01") %m+% months(2 * (t - 1))
  end_date <- ymd("2011-05-01") %m+% months(2 * (t - 1))
  test_current <- test %>%
    filter(Date >= start_date & Date < end_date) %>%
    select(-IsHoliday)
  test_current <- test_current %>%
    mutate(Wk = week(Date))
  
  # find the unique pairs of (Store, Dept) combo that appeared in both training and test sets
  train_pairs <- train[, 1:2] %>% count(Store, Dept) %>% filter(n != 0)
  test_pairs <- test_current[, 1:2] %>% count(Store, Dept) %>% filter(n != 0)
  unique_pairs <- intersect(train_pairs[, 1:2], test_pairs[, 1:2])
  
  # pick out the needed training samples, convert to dummy coding, then put them into a list
  train_split <- unique_pairs %>% 
    left_join(train, by = c('Store', 'Dept')) %>% 
    mutate(Wk = factor(ifelse(year(Date) == 2010, week(Date) - 1, week(Date)), levels = 1:52)) %>% 
    mutate(Yr = year(Date))
  train_split = as_tibble(model.matrix(~ Weekly_Sales + Store + Dept + Yr + I(Yr^2) + Wk, train_split)) %>% group_split(Store, Dept)
  
  # do the same for the test set
  test_split <- unique_pairs %>% 
    left_join(test_current, by = c('Store', 'Dept')) %>% 
    mutate(Wk = factor(ifelse(year(Date) == 2010, week(Date) - 1, week(Date)), levels = 1:52)) %>% 
    mutate(Yr = year(Date))
  test_split = as_tibble(model.matrix(~ Store + Dept + Yr + I(Yr^2) + Wk, test_split)) %>% mutate(Date = test_split$Date) %>% group_split(Store, Dept)
  
  # pre-allocate a list to store the predictions
  test_pred <- vector(mode = "list", length = nrow(unique_pairs))
  
  # perform regression for each split, note we used lm.fit instead of lm
  for (i in 1:nrow(unique_pairs)) {
    tmp_train <- train_split[[i]]
    tmp_test <- test_split[[i]]
    
    # shift for fold 5
    if(t == 5){
      shift <- 1/7
      tmp_test[, "Wk51"] <- tmp_test[, "Wk51"] * (1 - shift) + tmp_test[, "Wk52"] * shift
    }
    
    mycoef <- lm.fit(as.matrix(tmp_train[, -(2:4)]), tmp_train$Weekly_Sales)$coefficients
    mycoef[is.na(mycoef)] <- 0
    tmp_pred <- mycoef[1] + as.matrix(tmp_test[, 4:56]) %*% mycoef[-1]
    
    test_pred[[i]] <- cbind(tmp_test[, 2:3], Date = tmp_test$Date, Weekly_Pred = tmp_pred[,1])
  }
  
  # turn the list into a table at once; this is much more efficient then keep concatenating small tables
  test_pred <- bind_rows(test_pred)
  
  return(test_pred)
}


