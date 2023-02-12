library(pROC)
library(glmnet)
library(text2vec)

set.seed(1306)

myvocab = scan(file= "myvocab.txt", what = character())  

# split the data set
# data <- read.table("alldata.tsv", stringsAsFactors = FALSE,
#                    header = TRUE)
# 
# testIDs <- read.csv("project3_splits.csv", header = TRUE)
# 
# for(j in 1:5){
#   dir.create(paste("split_", j, sep=""))
#   train <- data[-testIDs[,j], c("id", "sentiment", "review") ]
#   test <- data[testIDs[,j], c("id", "review")]
#   test.y <- data[testIDs[,j], c("id", "sentiment", "score")]
#   
#   tmp_file_name <- paste("split_", j, "/", "train.tsv", sep="")
#   write.table(train, file=tmp_file_name, 
#               quote=TRUE, 
#               row.names = FALSE,
#               sep='\t')
#   tmp_file_name <- paste("split_", j, "/", "test.tsv", sep="")
#   write.table(test, file=tmp_file_name, 
#               quote=TRUE, 
#               row.names = FALSE,
#               sep='\t')
#   tmp_file_name <- paste("split_", j, "/", "test_y.tsv", sep="")
#   write.table(test.y, file=tmp_file_name, 
#               quote=TRUE, 
#               row.names = FALSE,
#               sep='\t')
# }


## setwd("~/Desktop/STAT542/PJ3/split_1")

#####################################
# Load libraries
# Load your vocabulary and training data
#####################################
trainpath = "train.tsv"
testpath = "test.tsv"
#testypath = "test_y.tsv"
train = read.table(trainpath,
                   stringsAsFactors = FALSE,
                   header = TRUE)
  
train$review <- gsub('<.*?>', ' ', train$review)
it_train = itoken(train$review,
                  preprocessor = tolower, 
                  tokenizer = word_tokenizer)
vectorizer = vocab_vectorizer(create_vocabulary(myvocab, 
                                                ngram = c(1L, 2L)))
dtm_train = create_dtm(it_train, vectorizer)
  
#####################################
# Train a binary classification model
#####################################
logit.cv = cv.glmnet(x = dtm_train, 
                     y = train$sentiment, 
                     alpha = 1,
                     family='binomial', 
                     type.measure = "auc")
  

#####################################
# Load test data, and
# Compute prediction
#####################################
test = read.table(testpath,
                  stringsAsFactors = FALSE,
                  header = TRUE)
test$review <- gsub('<.*?>', ' ', test$review)
it_test = itoken(test$review,
                 preprocessor = tolower, 
                 tokenizer = word_tokenizer)
dtm_test = create_dtm(it_test, vectorizer)
  
pred.y = predict(logit.cv, s = logit.cv$lambda.min, newx = dtm_test, type = "response")
output = data.frame(id = test$id, prob = as.vector(pred.y))

# coef(logit.cv, s = logit.cv$lambda.min)

#####################################
# Store your prediction for test data in a data frame
# "output": col 1 is test$id
#           col 2 is the predicted probs
#####################################
write.table(output, file = "mysubmission.txt", 
            row.names = FALSE, sep='\t')


#5. Evaluation (will be removed before submitting)
# test.y = read.table("test_y.tsv", header = TRUE)
# pred = read.table("mysubmission.txt", header = TRUE)
# pred = merge(pred, test.y, by="id")
# roc_obj = roc(pred$sentiment, pred$prob)
# tmp = pROC::auc(roc_obj)
# print(tmp)