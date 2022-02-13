from flask import Flask, request, jsonify
import utils

app = Flask(__name__)


@app.route('/get_car_model_names', methods=['GET'])
def get_car_model_names():
    response = jsonify({
        'Car Models' : utils.get_car_model_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


@app.route('/predict_car_price', methods=['GET', 'POST'])
def predict_car_price():
    name = str(request.form['name'])
    km_driven = int(request.form['km_driven'])
    fuel = int(request.form['fuel'])
    seller_type = int(request.form['seller_type'])
    transmission = int(request.form['transmission'])
    owner = int(request.form['owner'])
    mileage = float(request.form['mileage'])
    engine = float(request.form['engine'])
    max_power = float(request.form['max_power'])
    seats = float(request.form['seats'])
    age = int(request.form['seats'])

    response = jsonify({
        'car_price':utils.predict_price(name, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats, age)
    })

    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


if __name__ == "__main__":
    print("Starting Python Flask Server")
    utils.load_saved_artifacts()
    app.run()
