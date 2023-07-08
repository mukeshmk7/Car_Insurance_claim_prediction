from flask import Flask, render_template, request
from joblib import load
import pandas as pd
import warnings 
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

app = Flask(__name__)

def data_preprocessor(df):

    df["max_torque_Nm"] = df["max_torque"].str.extract(r"([-+]?[0-9]*\.?[0-9]+)(?=\s*Nm)").astype('float64')
    df["max_torque_rpm"] = df["max_torque"].str.extract(r"([-+]?[0-9]*\.?[0-9]+)(?=\s*rpm)").astype('float64')

    df["max_power_bhp"] = df["max_power"].str.extract(r"([-+]?[0-9]*\.?[0-9]+)(?=\s*bhp)").astype('float64')
    df["max_power_rpm"] = df["max_power"].str.extract(r"([-+]?[0-9]*\.?[0-9]+)(?=\s*rpm)").astype('float64')

    df.drop(["max_torque","max_power"],axis=1,inplace=True)
    return df

@app.route("/")
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    policy_tenure = float(request.form.get('policy_tenure'))
    age_of_car = float(request.form.get('age_of_car'))
    age_of_policyholder = float(request.form.get('age_of_policyholder'))
    area_cluster = str(request.form.get('area_cluster'))
    population_density = int(request.form.get('population_density'))
    make = int(request.form.get('make'))
    segment = str(request.form.get('segment'))
    model_car = str(request.form.get('model'))
    fuel_type = str(request.form.get('fuel_type'))
    max_torque = str(request.form.get('max_torque'))
    max_power = str(request.form.get('max_power'))
    engine_type = str(request.form.get('engine_type'))
    airbags = int(request.form.get('airbags'))
    is_esc = str(request.form.get('is_esc'))
    is_adjustable_steering = str(request.form.get('is_adjustable_steering'))
    is_tpms = str(request.form.get('is_tpms'))
    is_parking_sensors = str(request.form.get('is_parking_sensors'))
    is_parking_camera = str(request.form.get('is_parking_camera'))
    rear_brakes_type = str(request.form.get('rear_brakes_type'))
    cylinder = int(request.form.get('cylinder'))
    displacement = int(request.form.get('displacement'))
    transmission_type = str(request.form.get('transmission_type'))
    gear_box = int(request.form.get('gear_box'))
    steering_type = str(request.form.get('steering_type'))
    turning_radius = float(request.form.get('turning_radius'))
    length = int(request.form.get('length'))
    width = int(request.form.get('width'))
    height = int(request.form.get('height'))
    gross_weight = int(request.form.get('gross_weight'))
    is_front_fog_lights = str(request.form.get('is_front_fog_lights'))
    is_rear_window_wiper = str(request.form.get('is_rear_window_wiper'))
    is_rear_window_washer = str(request.form.get('is_rear_window_washer'))
    is_rear_window_defogger = str(request.form.get('is_rear_window_defogger'))
    is_brake_assist = str(request.form.get('is_brake_assist'))
    is_power_door_locks = str(request.form.get('is_power_door_locks'))
    is_central_locking = str(request.form.get('is_central_locking'))
    is_power_steering = str(request.form.get('is_power_steering'))
    is_driver_seat_height_adjustable = str(request.form.get('is_driver_seat_height_adjustable'))
    is_day_night_rear_view_mirror = str(request.form.get('is_day_night_rear_view_mirror'))
    is_ecw = str(request.form.get('is_ecw'))
    is_speed_alert = str(request.form.get('is_speed_alert'))
    ncap_rating = int(request.form.get('ncap_rating'))
    df_sample = pd.DataFrame({'policy_tenure': [policy_tenure],'age_of_car': [age_of_car], 'age_of_policyholder': [age_of_policyholder], 'area_cluster': [area_cluster], 'population_density': [population_density], 'make': [make],
                              'segment': [segment], 'model_car': [model_car], 'fuel_type': [fuel_type],'max_torque': [max_torque], 'max_power': [max_power],
                              'engine_type': [engine_type], 'airbags': [airbags], 'is_esc': [is_esc],'is_adjustable_steering': [is_adjustable_steering], 'is_tpms': [is_tpms],
                              'is_parking_sensors': [is_parking_sensors], 'is_parking_camera': [is_parking_camera], 'rear_brakes_type': [rear_brakes_type],'cylinder': [cylinder],
                              'displacement': [displacement], 'transmission_type': [transmission_type], 'gear_box': [gear_box],'steering_type': [steering_type],
                              'turning_radius': [turning_radius], 'length': [length], 'width': [width],'height': [height], 'gross_weight': [gross_weight],
                              'is_front_fog_lights': [is_front_fog_lights], 'is_rear_window_wiper': [is_rear_window_wiper],
                              'is_rear_window_washer': [is_rear_window_washer], 'is_rear_window_defogger': [is_rear_window_defogger],
                              'is_brake_assist': [is_brake_assist], 'is_power_door_locks': [is_power_door_locks], 'is_central_locking': [is_central_locking],
                              'is_power_steering': [is_power_steering], 'is_driver_seat_height_adjustable': [is_driver_seat_height_adjustable],
                              'is_day_night_rear_view_mirror': [is_day_night_rear_view_mirror], 'is_ecw': [is_ecw],
                              'is_speed_alert': [is_speed_alert], 'ncap_rating': [ncap_rating]})
    df_sample = data_preprocessor(df_sample)
    df_sample['volume'] = df_sample['length'] * df_sample['width'] * df_sample['height']
    df_sample.drop(['length', 'width', 'height'], axis=1, inplace=True)
    X = df_sample
    num = X[['policy_tenure', 'age_of_car', 'age_of_policyholder', 'population_density']]
    num_binned = pd.DataFrame(kbin.transform(num), index=num.index, columns=num.columns, ).add_suffix('_rank')
    X.drop(['age_of_car', 'age_of_policyholder', 'population_density'], axis=1, inplace=True)
    X['make'] = X['make'].astype('object')
    X['airbags'] = X['airbags'].astype('object')
    X['displacement'] = X['displacement'].astype('object')
    X['cylinder'] = X['cylinder'].astype('object')
    X['gear_box'] = X['gear_box'].astype('object')
    X['turning_radius'] = X['turning_radius'].astype('object')
    X['gross_weight'] = X['gross_weight'].astype('object')
    X['ncap_rating'] = X['ncap_rating'].astype('object')
    X['max_torque_Nm'] = X['max_torque_Nm'].astype('object')
    X['max_torque_rpm'] = X['max_torque_rpm'].astype('object')
    X['max_power_bhp'] = X['max_power_bhp'].astype('object')
    X['max_power_rpm'] = X['max_power_rpm'].astype('object')
    X['volume'] = X['volume'].astype('object')
    X = pd.get_dummies(X, )
    X_all = pd.concat([num_binned, X], axis=1, join="inner")
    X_all = X_all.reindex(columns=col_names, fill_value=0)
    df_sample = X_all.reindex(columns=col_names)
    output = model.predict(df_sample)[0]
    if output == 0:
        result = "not claimed"
    elif output == 1:
        result = "claimed"
    return render_template('predict.html', output= result)


if __name__  ==  '__main__':
    model = load('model.pkl')
    kbin = load('kbin.pkl')
    col_names = load('column_names.pkl')
    app.run(debug=True)


