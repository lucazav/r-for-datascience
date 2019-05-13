
# Install needed packages
pkgs <- c('dplyr','readr','funModeling',
          'parsnip', 'dials', 'yardstick',
          'randomForest', 'ranger', 'ggplot2')

for (pkg in pkgs) {
  if (! (pkg %in% rownames(installed.packages()))) { install.packages(pkg) }
}


# Load packages to read from csv and manipulate data in a 'tidy' way
library(readr)
library(dplyr)


project_folder <- "Z:/SolidQ/Speaker/2019 - R for Data Science/"


bank <- paste0(project_folder,'bank-marketing.zip') %>%
  unzip('bank-additional-full.csv') %>%
  read_delim(delim = ';')


# rename target class value 'yes' for better interpretation and convert to factor
bank <- bank %>%
  mutate(y = ifelse(y == 'yes', 'term.deposit', y))


# transform strings in factors
bank <- bank %>%
  mutate_if(is.character, as.factor)

# Summary info about dataset
bank %>%
  summary()

# For example, "month" has more than 6 factors. To show all of them,
# use maxsum
bank %>%
  summary(maxsum = 15)


# Let's check the y distribution.
# Classes are unbalanced
bank %>%
  group_by(y) %>%
  summarise(n = n()) %>%
  mutate(perc = n / sum(n))


# "duration" is not known before a call is performed.
# Also, after the end of the call y is obviously known.
# We remove the duration to avoid leaks
bank <- bank %>%
  select(-duration)


#
# Let's do a few of feature engineering
#

# Create a dummy variable
bank$contacted_before <- ifelse(bank$pdays == 999, 0, 1)

# Let's check how many 999 there are
bank %>%
  select(pdays) %>%
  group_by(pdays) %>%
  count()

# Let's see all the results
bank %>%
  select(pdays) %>%
  group_by(pdays) %>%
  count() %>%
  View()

# Let's check the distribution of pdays removing the 999s
bank %>%
  select(pdays) %>%
  filter(pdays != 999) %>%
  pull() %>%
  summary()

# We want to check also standard deviation
bank %>%
  select(pdays) %>%
  filter(pdays != 999) %>%
  summarise(min = min(pdays),
            q25 = quantile(pdays, .25),
            median = median(pdays),
            mean = mean(pdays),
            q75 = quantile(pdays, .75),
            max = max(pdays),
            sd = sd(pdays),
            n = n())
            

# Let's check the pdays distribution replacing the 999s with
# the mean (without the 999s)
bank %>%
  select(pdays) %>%
  mutate(pdays_mean = mean( ifelse(pdays < 999, pdays, NA),
                            na.rm = TRUE )) %>%
  mutate(pdays = ifelse(pdays == 999, pdays_mean, pdays)) %>%
  summarise(min = min(pdays),
            q25 = quantile(pdays, .25),
            median = median(pdays),
            mean = mean(pdays),
            q75 = quantile(pdays, .75),
            max = max(pdays),
            sd = sd(pdays),
            n = n())

# Substituting with the mean will change the distribution.
# For simplicity I'll do that, but in this case I'd prefer to use
# a regression to find missing (999) pdays values
bank <- bank %>%
  mutate(pdays_mean = mean( ifelse(pdays < 999, pdays, NA),
                            na.rm = TRUE )) %>%
  mutate( pdays = ifelse(pdays == 999, pdays_mean, pdays) ) %>%
  select(-pdays_mean)



# Data  binning is also known as  bucketing, discretization,
# categorization or quantization. It is a way to simplify and compress
# a column of data, by reducing the number of possible values or
# levels represented in the data.

# Let's Bin numerical variables with a lot of unique values
# Let's check how many distinct count values have numerical variables
numeric_distinct_count <- bank %>%
  select_if(is.numeric) %>%
  mutate_all(~ n_distinct(.) ) %>%
  distinct()

print(numeric_distinct_count)

# Get only the variables having more than 15 distinct values
numeric_distinct_count %>%
  select_if(~ . > 10)


# Vector containing the names of the variables with more than 10
# distinct values
vect_names_discr <- numeric_distinct_count %>%
  select_if(~ . > 10) %>%
  names()


# Let's define the equal frequency binning rules for each input
# variable
library(funModeling)

d_bins <- bank %>%
  discretize_get_bins(input = vect_names_discr, n_bins = 10)

# Apply the binning rules to the tibble
bank <- bank %>%
  discretize_df(data_bins = d_bins, stringsAsFactors = TRUE)

# Let's check the numeric variables distribution after binning
bank %>%
  select(vect_names_discr) %>%
  summary()



###############################################################
## MODELING PHASE


# Let's prepare data for training and train models (hold out)
train_size <- 0.8
bank <- bank %>% mutate(id = row_number())

# To better test our model we need a train set which has the
# same distribution of the all set for the target y. We can't
# simply do a random sampling, since it'll change the training
# set distribution over which the model will be trained
# (remeber our data set is imbalanced!).

# So, not a simple sample like this one:
# bank.train <- bank %>% sample_frac(train_size)

# ... but we need a strata sampling:
bank.train <- bank %>%
  group_by(y) %>%
  mutate(num_rows = n()) %>%
  sample_frac(train_size, weight = num_rows) %>%
  ungroup() %>%
  select(-num_rows)

# Let's check the y distribution from the train set
bank.train %>%
  group_by(y) %>%
  summarise(n = n()) %>%
  mutate(perc = n / sum(n))

# Let's get the test set getting all those rows from the
# complete data set having the row_id different from the
# row_ids contained in the train dataset
bank.test <- anti_join(bank, bank.train, by = 'id')

# Let's check the y distribution from the test set
bank.test %>%
  group_by(y) %>%
  summarise(n = n()) %>%
  mutate(perc = n / sum(n))

# Remove the id from the train and test data sets
bank.train$id <- NULL
bank.test$id <- NULL


# Many functions have different interfaces and arguments names and
# parsnip standardizes the interface for fitting models as well as
# the return values
library(parsnip)

# Tools to create and manage values of tuning parameters integrated
# with parsnip
library(dials)

# Yardstick is a package to estimate how well models are working using
# tidy data principles. It contains many of the performance measurement methods
# used in machine learning
library(yardstick)

# Change the class of interest to 1 (0 is the default in yardstick)
options(yardstick.event_first = FALSE)

# Let's prepare the model specifications (like a template)
# We'll use a random forest, passing some of its hyperparameters.
#  - mtry:  Number of variables randomly sampled as candidates
#           at each split.
#  - ntree: Number of trees to grow.
rf_with_seed <- 
  rand_forest(trees = 500, mtry = 1, mode = "classification") %>%
  set_engine(
    # A fast implementation of Random Forests, particularly
    # suited for high dimensional data
    'ranger',
    
    #Classification and regression based on a forest of trees
    # using random inputs, based on Breiman (2001)
    #'randomForest',
    
    seed = 12345)

# Let's prepare a grid containing all the combinations of the selected
# hyperparameters, each of which may vary in a defined range of values.
# From this complete grid, let's select a sample of 10 combinations.
rf_grid <- grid_random(
  trees %>%       range_set(c( 500, 2000)), 
  mtry  %>%       range_set(c( 1,  floor(sqrt(ncol(bank))))),
  size = 10
)


# Let's define the objects used to store all the models results
# while looping throught the grid
models <- list()
pr_threshold_values <- list()
roc_threshold_values <- list()
model_hyperparams <- tibble()

# Let's fit our models for each grid row
# (this for loop takes more than 5 minutes)
for(i in 1:nrow(rf_grid)) {
  # i = 3   # used to debug the inner code
  
  # Inject the hyperparameters values of the current grid row
  # into the model specification
  rf <- merge(rf_with_seed, rf_grid[i, ])[[1]]
  
  # Let's fit the model using the current hyperparameters
  rf_fit <- rf %>% 
    fit(y ~ ., data = bank.train)
  
  # Let's calculate the the predictions for the test set,
  # using the upon fitted model. We need the probability
  # of each prediction, so that we can use the threshold later
  predictions <- predict(rf_fit, new_data = bank.test, type = 'prob')
  
  # Let's store the labels "true" values and both the probabilities
  # of the "no" and "term.deposit" occurrences
  results <- tibble(
    actual = bank.test$y,
    prob_no = predictions$.pred_no,
    prob_term.deposit = predictions$.pred_term.deposit)
  
  
  # A fast way to measure the goodness of a model is to
  # get the value of the Area Under the Curve (AUC). Usually AUC is
  # calculated from the ROC curve (ROC_AUC). When the target variable
  # is umbalanced, it's better to get it from the Precision-Recall
  # curve (PR_AUC), 
  
  # Let's store the Precision-Recall curve values (threshold,
  # precision, recall) in a list in order to use it later
  roc_threshold_values[[i]] <- results %>%
    roc_curve(truth = actual, prob_term.deposit)
  
  # Let's store the Precision-Recall curve values (threshold,
  # precision, recall) in a list in order to use it later
  pr_threshold_values[[i]] <- results %>%
    pr_curve(truth = actual, prob_term.deposit)
  
  # Let's get the ROC_AUC
  rocauc <- results %>%
    roc_auc(truth = actual, prob_term.deposit)
  
  # Let's get the PR_AUC
  prauc <- results %>%
    pr_auc(truth = actual, prob_term.deposit)
  
  # Let's store the id, the hyperparameters values and the AUC
  # in a tibble in order to use it later
  if (i == 1) {
    model_hyperparams <- tibble(id = i, trees = rf_grid[i,]$trees,
                                mtry = rf_grid[i,]$mtry,
                                roc_auc = rocauc$.estimate,
                                pr_auc = prauc$.estimate)
  } else {
    model_hyperparams <- add_row(model_hyperparams, id = i,
                                 trees = rf_grid[i,]$trees,
                                 mtry = rf_grid[i,]$mtry,
                                 roc_auc = rocauc$.estimate,
                                 pr_auc = prauc$.estimate)
  }
  
  # Let's store the fitted model in a list in order to use it later
  models[[i]] <- rf_fit
  
}

# Let's check all the 10 AUC calculated for each hyperparameters
# combination. It seems lot of them are equal
print(model_hyperparams)

# Tell to the engine to print more decimal numbers
options(pillar.sigfig = 6)

# Now it's evident the AUCs are different
print(model_hyperparams)


# Get the best model ID (the one having the max ROC_AUC)
best_model_id <- model_hyperparams %>%
  filter(roc_auc == max(roc_auc)) %>%
  pull(id)

# Retrieve the best model using the upon calculated ID
best_model <- models[[best_model_id]]

# Get the best model predicitons for the test set
best_model_predictions <- predict(best_model, new_data = bank.test,
                                  type = 'prob')

# Let's fix a threshold to determine the positive and negative classes
# (if the predicted probability for the target of an osservation is
# equal or greater than the threshold, the predicted class is positive)
threshold <- 0.4

# 
best_model_results <- tibble(
  actual = bank.test$y,
  prob_no = best_model_predictions$.pred_no,
  prob_term.deposit = best_model_predictions$.pred_term.deposit,
  predicted = factor(
    ifelse(best_model_predictions$.pred_term.deposit >= threshold,
           'term.deposit', 'no'))
)

# Get the confusion matrix object
cross_tab_actual_predicted <- conf_mat(best_model_results,
         truth = actual,
         estimate = predicted)

# Plot the confusion matrix
autoplot(cross_tab_actual_predicted, type = "heatmap") +
  ggtitle('Confusion Matrix',
          subtitle = paste0('Threshold = ', threshold))
  

# Retrieve the ROC curve values for the best model
best_model_roc_threshold_values <- roc_threshold_values[[best_model_id]]

# Plot the ROC curve
roc_auc_value <- model_hyperparams %>%
  filter(id == best_model_id) %>%
  pull(roc_auc)

autoplot(best_model_roc_threshold_values) +
  ggtitle('ROC Curve',
          subtitle = paste0('AUC = ', round(roc_auc_value, digits = 4)))



# Retrieve the Precision-Recall curve values for the best model
best_model_pr_threshold_values <- pr_threshold_values[[best_model_id]]

# Plot the Precision-Recall curve
pr_auc_value <- model_hyperparams %>%
  filter(id == best_model_id) %>%
  pull(pr_auc)

autoplot(best_model_pr_threshold_values) +
  ggtitle('Precision-Recall Curve',
          subtitle = paste0('AUC = ', round(pr_auc_value, digits = 4)))
