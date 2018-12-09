#Learning Objectives
#   - Calculate the size of a model
#   - Apply L1 regularization to reduce the size of a model by increasing sparsity

#One way to reduce complexity is to use a regularization function that encourages weights to be exactly zero. For linear models such as reression, a zero weight is equivalent
#to not using the corresponding feature at all. In addition to avoiding overfitting, the resulting model will be more efficient.
#L1 regularization is a good way to increase sparsity.

from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")

california_housing_dataframe["population"].hist()
california_housing_dataframe = shuffle(california_housing_dataframe)

def preprocess_features(california_housing_dataframe):
    selected_features = california_housing_dataframe[
    ["latitude",
     "longitude",
     "housing_median_age",
     "total_rooms",
     "total_bedrooms",
     "population",
     "households",
     "median_income"]
    ]
    preprocessed_features = selected_features.copy()
    #Create synthetic features
    preprocessed_features["rooms_per_person"]=california_housing_dataframe["total_rooms"] / california_housing_dataframe["population"]
    return preprocessed_features

def preprocess_targets(california_housing_dataframe):
    """Prepares target features (i.e., labels) from California housing data set.

      Args:
        california_housing_dataframe: A Pandas DataFrame expected to contain data
          from the California housing data set.
      Returns:
        A DataFrame that contains the target feature.
      """
    output_targets = pd.DataFrame()
    output_targets["median_house_value_is_high"] = (california_housing_dataframe["median_house_value"] > 265000).astype(float)
    return output_targets

#Choose input examples and output targets both for trainging and validation
training_examples = preprocess_features(california_housing_dataframe.head(12000))
training_targets = preprocess_targets(california_housing_dataframe.head(12000))

validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model.

    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
    features = {key:np.array(value) for key,value in dict(features).items()}

    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)
    if(shuffle):
        ds = ds.shuffle(10000)

    features, labels = ds.make_one_shot_iterator().get_next()

    return features, labels

def get_quantile_based_buckets(feature_values, num_buckets):
    quantiles = feature_values.quantile([(i+1.)/(num_buckets+ 1.) for i in range(num_buckets)])
    return [quantiles[q] for q in quantiles.keys()]


def construct_feature_columns():
        """
        Construct the Tensorflow feature columns

        returns:
        -A set of feature columns
        """

        bucketized_longitude = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("longitude"),
                                                                    boundaries=get_quantile_based_buckets(training_examples["longitude"],50))
        bucketized_latitude = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("latitude"),
                                                                    boundaries=get_quantile_based_buckets(training_examples["latitude"],50))
        bucketized_housing_median_age = tf.feature_column.bucketized_column(
                                tf.feature_column.numeric_column("housing_median_age"),
                                boundaries=get_quantile_based_buckets(training_examples["housing_median_age"], 10))
        bucketized_total_rooms = tf.feature_column.bucketized_column(
                                tf.feature_column.numeric_column("total_rooms"),
                                boundaries=get_quantile_based_buckets(training_examples["total_rooms"], 10))
        bucketized_total_bedrooms = tf.feature_column.bucketized_column(
                                tf.feature_column.numeric_column("total_bedrooms"),
                                boundaries=get_quantile_based_buckets(training_examples["total_bedrooms"], 10))
        bucketized_population = tf.feature_column.bucketized_column(
                                tf.feature_column.numeric_column("population"),
                                boundaries=get_quantile_based_buckets(training_examples["population"], 10))
        bucketized_median_income = tf.feature_column.bucketized_column(
                                tf.feature_column.numeric_column("median_income"),
                                boundaries=get_quantile_based_buckets(training_examples["median_income"], 10))
        bucketized_rooms_per_person = tf.feature_column.bucketized_column(
                                tf.feature_column.numeric_column("rooms_per_person"),
                                boundaries=get_quantile_based_buckets(
                                training_examples["rooms_per_person"], 10))
        bucketized_households = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("households"),
                                boundaries=get_quantile_based_buckets(training_examples["households"], 10))

        long_x_lat = tf.feature_column.crossed_column(set([bucketized_longitude, bucketized_latitude]), hash_bucket_size=1000)

        feature_columns = set([
        long_x_lat,
        bucketized_longitude,
        bucketized_latitude,
        bucketized_housing_median_age,
        bucketized_total_rooms,
        bucketized_total_bedrooms,
        bucketized_population,
        bucketized_households,
        bucketized_median_income,
        bucketized_rooms_per_person
        ])
        return feature_columns

#Calculate the model size
#To calculate the model size, we simply count the number of parameters that are non-zero. WE provide a helper function below to do that.
#The function uses intimate kwnoledge of the Estimators API - don't worry about understanding how it works.
def model_size(estimator):
    variables = estimator.get_variable_names()
    size = 0
    for variable in variables:
        if not any(x in variable for x in ['global_step', 'centered_bias_weight', 'bias_weight', 'Ftrl']):
                    size += np.count_nonzero(estimator.get_variable_value(variable))
    return size

#Reduce the Model Size
#Your team needs to build a highly accurate Logistic Regression model on the SmartRing, a ring that is so mart it can sense the
#demographics of a city block ('median_income', 'avg_rooms', 'households',...,etc) and tell you whether the given city block is high cost
#city block or not.
#Since the SmartRing is small, the engineering team has determined that it can only handle a model that has no more than 600 parameters.
#On the other hand, the product managment team has determined that the model is not launchable unless the LogLoss is less than 0.35
#on the houldout test set.
#Can you user your secret weapon - L1 regularization - to tune the model to satisfy both the size and accuracy constraints?

#TASK 1: Find a good regularization coefficient
#Find an L1 regularization strength parameter which satifies both constraints - model size is less than 600 and log-loss is less than 0.35 on validation set.
#There are many ways to apply  L1 regularization in our model. Here, we do it using FtrlOptimizer, which is designed to give better results than standard gradient descent

def train_linear_classifier_model_reg(
    learning_rate,
    select_optimizer,
    regularization_strength,
    steps,
    batch_size,
    feature_columns,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):

    periods = 10
    steps_per_period = steps / periods

    #Create a linear classifier object
    if(select_optimizer==1):
        my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate, l1_regularization_strength=regularization_strength)
    elif (select_optimizer==2):
        my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate, l2_regularization_strength = regularization_strength)
    else:
        print("Invalid select optimizer value")
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_classifier = tf.estimator.LinearClassifier(feature_columns=feature_columns,optimizer=my_optimizer)

    #Create input functions
    training_input_fn = lambda: my_input_fn(training_examples, training_targets["median_house_value_is_high"],batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples, training_targets["median_house_value_is_high"],num_epochs=1, shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples, validation_targets["median_house_value_is_high"], num_epochs=1, shuffle=False)

    #Train the model, but do so inside a loop so that we can periodically
    # assess loss metrics
    print("Training model..")
    print("LogLoss (on trainig data):")
    training_log_losses = []
    validation_log_losses = []
    for period in range(0,periods):
      #Train the model, starting from the prior state.
      #For each step, calls input_fn, which returns one batch of data...
      linear_classifier.train(input_fn=training_input_fn, steps=steps_per_period)

      #Compute probabilities as we are using logistic regression for binary classification (binary targets)
      training_probabilities = linear_classifier.predict(input_fn=predict_training_input_fn)
      training_probabilities = np.array([item['probabilities'] for item in training_probabilities])
      validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
      validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])

      #Compute training and validation loss:
      training_log_loss = metrics.log_loss(training_targets, training_probabilities)
      validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)
      #Print out validation log loss
      print("Period %02d: %.2f" %(period, validation_log_loss))

      #Add Log Losses from this period to our list
      training_log_losses.append(training_log_loss)
      validation_log_losses.append(validation_log_loss)
    print("Model training finished")
    # Output a graph of loss metrics over periods.
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.tight_layout()
    plt.plot(training_log_losses, label="training")
    plt.plot(validation_log_losses, label="validation")
    plt.legend()

    return linear_classifier

def train_linear_classifier_model(
    learning_rate,
    steps,
    batch_size,
    feature_columns,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  """Trains a linear classification model.

  In addition to training, this function also prints training progress information,
  as well as a plot of the training and validation loss over time.

  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    training_examples: A `DataFrame` containing one or more columns from
      `california_housing_dataframe` to use as input features for training.
    training_targets: A `DataFrame` containing exactly one column from
      `california_housing_dataframe` to use as target for training.
    validation_examples: A `DataFrame` containing one or more columns from
      `california_housing_dataframe` to use as input features for validation.
    validation_targets: A `DataFrame` containing exactly one column from
      `california_housing_dataframe` to use as target for validation.

  Returns:
    A `LinearClassifier` object trained on the training data.
  """

  periods = 10
  steps_per_period = steps / periods

  # Create a linear classifier object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_classifier = tf.estimator.LinearClassifier(feature_columns=feature_columns, optimizer=my_optimizer)

  # Create input functions.
  training_input_fn = lambda: my_input_fn(training_examples,
                                          training_targets["median_house_value_is_high"],
                                          batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(training_examples,
                                                  training_targets["median_house_value_is_high"],
                                                  num_epochs=1,
                                                  shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                                    validation_targets["median_house_value_is_high"],
                                                    num_epochs=1,
                                                    shuffle=False)

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("LogLoss (on training data):")
  training_log_losses = []
  validation_log_losses = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    linear_classifier.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.
    training_probabilities = linear_classifier.predict(input_fn=predict_training_input_fn)
    training_probabilities = np.array([item['probabilities'] for item in training_probabilities])

    validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
    validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])

    training_log_loss = metrics.log_loss(training_targets, training_probabilities)
    validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)
    # Occasionally print the current loss.
    print("  period %02d : %0.2f" % (period, training_log_loss))
    # Add the loss metrics from this period to our list.
    training_log_losses.append(training_log_loss)
    validation_log_losses.append(validation_log_loss)
  print("Model training finished.")

  # Output a graph of loss metrics over periods.
  plt.ylabel("LogLoss")
  plt.xlabel("Periods")
  plt.title("LogLoss vs. Periods")
  plt.tight_layout()
  plt.plot(training_log_losses, label="training")
  plt.plot(validation_log_losses, label="validation")
  plt.legend()

  return linear_classifier



linear_classifier_no_reg = train_linear_classifier_model(
                    learning_rate=0.1,
                    steps=300,
                    batch_size=40,
                    feature_columns=construct_feature_columns(),
                    training_examples=training_examples,
                    training_targets=training_targets,
                    validation_examples=validation_examples,
                    validation_targets=validation_targets)

linear_classifierl1 = train_linear_classifier_model_reg(learning_rate=0.1,
                                                    select_optimizer=1,
                                                    regularization_strength=0.5,
                                                    steps=300,
                                                    batch_size=40,
                                                    feature_columns=construct_feature_columns(),
                                                    training_examples=training_examples,
                                                    training_targets=training_targets,
                                                    validation_examples=validation_examples,
                                                    validation_targets=validation_targets)

linear_classifierl2 = train_linear_classifier_model_reg(learning_rate=0.1,
                                                    select_optimizer=2,
                                                    regularization_strength=0.5,
                                                    steps=300,
                                                    batch_size=40,
                                                    feature_columns=construct_feature_columns(),
                                                    training_examples=training_examples,
                                                    training_targets=training_targets,
                                                    validation_examples=validation_examples,
                                                    validation_targets=validation_targets)


print("Model Size: %02d" %(model_size(linear_classifier_no_reg)))
print("Model Size: %02d" %(model_size(linear_classifierl2)))
print("Model Size: %02d" %(model_size(linear_classifierl1)))

model_size_no_reg = model_size(linear_classifier_no_reg)
model_size_l2 = model_size(linear_classifierl2)
model_size_l1 = model_size(linear_classifierl1)

reg_labels = np.array(['No regularization', 'L2', 'L1'])
model_sizes = np.array([model_size_no_reg,model_size_l2,model_size_l1])

plt.figure(figsize=(15,6))
plt.title('Model evaluation - Size', fontsize=20)
plt.xlabel('Regularization', fontsize=16)
plt.ylabel('Model size', fontsize=16)
plt.bar([1,2,3],height=model_sizes, width=0.5, bottom=None, align='center', tick_label=reg_labels, edgecolor='black')
plt.grid()


model_sizes = {0:[model_size(linear_classifierl1)], 1:[model_size(linear_classifierl2)]}
model_sizes[0]
