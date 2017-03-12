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

app = Flask(__name__)
traffic_Data = []
outputs = []

@app.route("/")
def index():
    return app.send_static_file('index.html')

#trains for prediction
@app.route('/train', methods=['POST'])
def train():
    road_ID = request.form['road_ID']
    direction = request.form['direction']
    dayOfWeek = request.form['dayOfWeek']
    timeOfDay = request.form['timeOfDay']
    traffic_Status = request.form['traffic_Status']
    datapoints.append([str(road_ID), str(direction), str(dayOfWeek), str(timeOfDay)])
    outputs.append(int(traffic_Status))

    print "TRAFFIC DATA: "
    print traffic_Data
    print "OUTPUTS: "
    print outputs

    return app.send_static_file('index.html')

#make a prediction
@app.route('/predict', methods=['POST'])
def predict():
    if(len(traffic_Data) < 2):
        return("ERROR: NEED AT LEAST TWO DATA", 400)
    model = SVC(gamma = 0.001, C=100.)
    model.fit(traffic_Data, outputs)
    road_ID = request.form['road_ID']
    direction = request.form['direction']
    dayOfWeek = request.form['dayOfWeek']
    timeOfDay = request.form['timeOfDay']
    prediction = model.predict([[str(road_ID), str(direction), str(dayOfWeek), str(timeOfDay)]])

    print "TRAFFIC DATA: "
    print traffic_Data
    print "OUTPUTS: "
    print outputs
    print "PREDICTION: "
    print prediction

    return("PREDICTION: " + str(prediction), 200)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


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
