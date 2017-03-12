# Import the pandas library.
import pandas
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
# Read in the data.
from sklearn.metrics import mean_squared_error
from flask import Flask

# app = Flask(__name__)

'''# Make a histogram of all the ratings in the average_rating column.
plt.hist(predict["traffic_status"])

# Show the plot.
plt.show();

# Initialize the model with 2 parameters -- number of clusters and random state.
kmeans_model = KMeans(n_clusters=5, random_state=1)
# Get only the numeric columns from games.
good_columns = predict._get_numeric_data()
# Fit the model using the good columns.
kmeans_model.fit(good_columns)
# Get the cluster assignments.
labels = kmeans_model.labels_

# Create a PCA model.
pca_2 = PCA(2)
# Fit the PCA model on the numeric columns from earlier.
plot_columns = pca_2.fit_transform(good_columns)
# Make a scatter plot of each game, shaded according to cluster assignment.
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
# Show the plot.
plt.show();'''


# Read the CSV file with pandas. Use predict.[column name] to get column values.
predict = pandas.read_csv("trafficData.csv")

columns = predict.columns.tolist()
target = "traffic_status"
# road_id = predict.road_id
# direction = predict.direction
# dayOfWeek = predict.dayOfWeek
# timeOfDay = predict.timeOfDay
# traffic = predict.traffic_status
# Generate the training set.  Set random_state to be able to replicate results.
train = predict.sample(frac=.8, random_state=1)
# # Select anything not in the training set and put it in the testing set.
test = predict.loc[~predict.index.isin(train.index)]

model = LinearRegression()
# Fit the model to the training data.
model.fit(train[columns], train[target])

# Print the shapes of both sets.
# print(train.shape)
# print(test.shape)
# print(test)
# print(train)

# Generate our predictions for the test set.
predictions = model.predict(test[columns])
print(predictions) # predicting traffic status 5,1,6,18,9
# outputs 8.85 which is pretty close to 9

# Compute error between our test predictions and the actual values.
# percentError = mean_squared_error(predictions, test[target]) * 100
# print(percentError)

#
#
# @app.route("/")
# def hello():
#     return "testing"
#
# if __name__ == "__main__":
#     app.run()

