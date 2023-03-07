from flask import Flask, request, jsonify
import pandas as pd
from sklearn.datasets import fetch_california_housing
import numpy as np

app = Flask(__name__)

@app.route('/old_predict', methods=['GET'])
def old_predict():
    data = request.get_json()
    X_pred = pd.DataFrame.from_dict(data)
    breakpoint()
    model=LinearRegression()
    variables = X_pred[['HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude', 'MedHouseVal']]
    y_pred = model.predict(variables)
    return jsonify({'predicted_values': y_pred.tolist()})

@app.route('/predict', methods=['GET'])
def predict():
    housing=fetch_california_housing()

    housing.keys()

    housing['feature_names']
    housing['target_names']
    df = pd.concat([pd.DataFrame(housing.data, columns=housing.feature_names), 
                    pd.DataFrame(housing.target, columns=['MedHouseVal'])], axis=1)
    df.head()

    df.columns
    X=df[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
        'Latitude', 'Longitude']]
    y=df['MedHouseVal']
    

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    from sklearn.linear_model import LinearRegression
    lm = LinearRegression()

    lm.fit(X_train,y_train)

    print(lm.intercept_)
    print(lm.coef_)

    X_train.columns
    cdf = pd.DataFrame(lm.coef_, X_train.columns, columns=['Coeff'])
    cdf

    # Getting predictions from our model
    predictions=lm.predict(X_test)
    predictions

    y_test


    from sklearn import metrics
    mae=metrics.mean_absolute_error(y_test,predictions)
    mse=metrics.mean_squared_error(y_test,predictions)
    rmse=np.sqrt(metrics.mean_squared_error(y_test,predictions))
    
    # return jsonify({'predicted_values': predictions.tolist(), 'mae': mae, 'mse': mse, 'rmse': rmse})
     # create a table with actual and predicted values
    table = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
    
    # generate an HTML table from the pandas DataFrame
    html_table = table.to_html()
    
    # return the HTML table as a string
    return html_table
if __name__ == '__main__':
    app.run()
