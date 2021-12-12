
# rm(list = ls())
setwd("D:/R_Python/ML/Project/")
load('data_train_test.RData')

# ___________________________________________________________

ctrl_cv5 <- trainControl(method = "cv",
                         number = 5)
ctrl_nocv <- trainControl(method = "none")


data_linear_nocv <- train(log_DEPARTURE_DELAY ~ .,
                          data = data_train,
                          method = "lm",
                          trControl = ctrl_nocv)
data_linear_nocv_is <- regressionMetrics(data_train$log_DEPARTURE_DELAY, predict(data_linear_nocv, data_train))
data_linear_nocv_oss <- regressionMetrics(data_test$log_DEPARTURE_DELAY, predict(data_linear_nocv, data_test))


lambdas <- exp(log(10)*seq(-2, 9, length.out = 200))

tgrid_nocv <- expand.grid(alpha = 0, lambda=exp(log(10)*3.5))
tgrid <- expand.grid(alpha = 0, lambda=lambdas)

# RIDGE
data_ridge_nocv <- train(log_DEPARTURE_DELAY ~ .,
                         data = data_train,
                         method = "glmnet",
                         tuneGrid = tgrid_nocv,
                         trControl = ctrl_nocv)

data_ridge_nocv_is <- regressionMetrics(data_train$log_DEPARTURE_DELAY, predict(data_ridge_nocv, data_train))
data_ridge_nocv_oss <- regressionMetrics(data_test$log_DEPARTURE_DELAY, predict(data_ridge_nocv, data_test))


data_ridge_cv <- train(log_DEPARTURE_DELAY ~ .,
                       data = data_train,
                       method = "glmnet",
                       tuneGrid = tgrid,
                       trControl = ctrl_cv5)

data_ridge_cv_is <- regressionMetrics(data_train$log_DEPARTURE_DELAY, predict(data_ridge_cv, data_train))
data_ridge_cv_oss <- regressionMetrics(data_test$log_DEPARTURE_DELAY, predict(data_ridge_cv, data_test))


# LASSO
tgrid_l <- expand.grid(alpha = 1, lambda=lambdas)
tgrid_nocv_l <- expand.grid(alpha = 1, lambda=exp(log(10)*3.5))

data_lasso_nocv <- train(log_DEPARTURE_DELAY ~ .,
                         data = data_train,
                         method = "glmnet",
                         tuneGrid = tgrid_nocv_l,
                         trControl = ctrl_nocv)

data_lasso_nocv_is <- regressionMetrics(data_train$log_DEPARTURE_DELAY, predict(data_lasso_nocv, data_train))
data_lasso_nocv_oss <- regressionMetrics(data_test$log_DEPARTURE_DELAY, predict(data_lasso_nocv, data_test))


data_lasso_cv5 <- train(log_DEPARTURE_DELAY ~ .,
                        data = data_train,
                        method = "glmnet",
                        tuneGrid = tgrid_l,
                        trControl = ctrl_cv5)

data_lasso_cv_is <- regressionMetrics(data_train$log_DEPARTURE_DELAY, predict(data_lasso_cv5, data_train))
data_lasso_cv_oss <- regressionMetrics(data_test$log_DEPARTURE_DELAY, predict(data_lasso_cv5, data_test))



save(data_linear_nocv_is, data_linear_nocv_oss, data_ridge_nocv_is, data_ridge_nocv_oss, data_ridge_cv_is,
     data_ridge_cv_oss, data_lasso_nocv_is, data_lasso_nocv_oss, data_lasso_cv_is, data_lasso_cv_oss,
     file = "results.Rdata")



