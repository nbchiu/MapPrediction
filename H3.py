# Import the pandas library.
import pandas
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error



# Read in the data.
predict = pandas.read_csv("trafficData.csv")
# Print the names of the columns in games.
print(predict.columns)
print(predict.road_id)

# Make a histogram of all the ratings in the average_rating column.
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
plt.show();


columns = predict.columns.tolist()
target = "traffic_status"

# Generate the training set.  Set random_state to be able to replicate results.
train = predict.sample(frac=0.8, random_state=1)
# Select anything not in the training set and put it in the testing set.
test = predict.loc[~predict.index.isin(train.index)]
# Print the shapes of both sets.
print(train.shape)
print(test.shape)

# Initialize the model class.
model = LinearRegression()
# Fit the model to the training data.
model.fit(train[columns], train[target])

print(model.predict(test[columns]))

# Generate our predictions for the test set.
predictions = model.predict(test[columns])


