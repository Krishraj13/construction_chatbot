from flask import Flask, render_template, request, redirect, session, url_for
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.secret_key = 'super_secret_key'

# Load dataset and train model
df = pd.read_csv('construction.csv')
drop_cols = ['project_id', 'location', 'sequence_id', 'estimated_cost_usd', 'materials_sufficient', 'shortfall_details']
df.drop(columns=drop_cols, inplace=True)

feature_cols = [
    'building_type', 'room_type', 'total_plot_area_sqft', 'building_footprint_sqft',
    'floor_area_sqft', 'total_floors', 'floor_number',
    'inventory_bricks_available', 'inventory_cement_bags_available',
    'inventory_sand_tons_available', 'inventory_steel_kg_available',
    'inventory_labor_hours_available'
]

target_cols = [
    'bricks_required', 'cement_bags_required', 'sand_tons_required',
    'steel_kg_required', 'labor_hours_required'
]

X = df[feature_cols]
y = df[target_cols]

categorical = ['building_type', 'room_type']
numeric = [col for col in feature_cols if col not in categorical]

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
    ('num', StandardScaler(), numeric)
])

X_processed = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y.values, test_size=0.2, random_state=42)

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(target_cols), activation='linear')
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=256, verbose=0)

def clamp_predictions(preds):
    return np.clip(np.round(preds), 0, 1e7).astype(int)

def to_native(obj):
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_native(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return to_native(obj.tolist())
    else:
        return obj

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        btype = request.form['building_type']
        rtype = request.form['room_type']
        tpa = float(request.form['total_plot_area'])
        bfa = float(request.form['building_footprint'])
        tf = int(request.form['total_floors'])
        hpf = int(request.form.get('units_per_floor', 0))

        avg_inventory = df[[
            'inventory_bricks_available', 'inventory_cement_bags_available',
            'inventory_sand_tons_available', 'inventory_steel_kg_available',
            'inventory_labor_hours_available'
        ]].mean().to_dict()
        inventory = {k: int(v) for k, v in avg_inventory.items()}

        session['constructed_floors'] = []
        session['project_data'] = {
            'btype': btype, 'rtype': rtype, 'tpa': tpa,
            'bfa': bfa, 'tf': tf, 'hpf': hpf
        }
        session['inventory'] = to_native(inventory)
        session['floor_cache'] = {}

        return redirect(url_for('floor_input'))

    return render_template('home.html')

@app.route('/floor', methods=['GET', 'POST'])
def floor_input():
    project = session.get('project_data')
    inventory = session.get('inventory')
    constructed = session.get('constructed_floors', [])
    floor_cache = session.get('floor_cache', {})

    # Stop further input if construction is complete
    if len(constructed) >= project['tf']:
        return redirect(url_for('summary'))

    if request.method == 'POST':
        # Inventory update flow
        if 'update_inventory' in request.form:
            for key in inventory.keys():
                add_qty = request.form.get(key)
                if add_qty:
                    inventory[key] += int(add_qty)
            session['inventory'] = to_native(inventory)
            return redirect(url_for('floor_input'))

        floor = int(request.form['floor'])

        # Validate floor number
        if floor in constructed or floor >= project['tf']:
            return render_template('floor.html', error="Invalid or already constructed floor", floor=None)

        # Prediction flow
        if floor not in [0, project['tf'] - 1] and 'mid_floor' in floor_cache:
            result = floor_cache['mid_floor']
        else:
            floor_area = project['bfa']
            input_dict = {
                'building_type': [project['btype']],
                'room_type': [project['rtype']],
                'total_plot_area_sqft': [project['tpa']],
                'building_footprint_sqft': [project['bfa']],
                'floor_area_sqft': [floor_area],
                'total_floors': [project['tf']],
                'floor_number': [floor],
                'inventory_bricks_available': [inventory['inventory_bricks_available']],
                'inventory_cement_bags_available': [inventory['inventory_cement_bags_available']],
                'inventory_sand_tons_available': [inventory['inventory_sand_tons_available']],
                'inventory_steel_kg_available': [inventory['inventory_steel_kg_available']],
                'inventory_labor_hours_available': [inventory['inventory_labor_hours_available']]
            }
            df_input = pd.DataFrame(input_dict)
            transformed = preprocessor.transform(df_input)
            preds = model.predict(transformed, verbose=0)[0]
            clamped = clamp_predictions(preds)
            result = dict(zip(target_cols, clamped))

            if floor not in [0, project['tf'] - 1]:
                floor_cache['mid_floor'] = to_native(result)
                session['floor_cache'] = floor_cache

        result = to_native(result)

        # Check shortfall
        shortfall = {}
        for k in result:
            inv_k = 'inventory_' + k.split('_required')[0] + '_available'
            if result[k] > inventory.get(inv_k, 0):
                shortfall[inv_k] = result[k] - inventory.get(inv_k, 0)

        if shortfall:
            return render_template('add_inventory.html', shortfall=shortfall, floor=floor)

        # Deduct inventory
        for k in result:
            inv_k = 'inventory_' + k.split('_required')[0] + '_available'
            inventory[inv_k] -= result[k]

        constructed.append(floor)

        session['inventory'] = to_native(inventory)
        session['constructed_floors'] = to_native(constructed)

        # Check again if all floors are done
        if len(constructed) >= project['tf']:
            return redirect(url_for('summary'))

        return render_template('result.html', result=result, floor=floor)

    return render_template('floor.html')

@app.route('/summary')
def summary():
    constructed = to_native(session.get('constructed_floors', []))
    inventory = to_native(session.get('inventory', {}))

    return render_template('summary.html',
                           constructed=constructed,
                           inventory=inventory)

if __name__ == '__main__':
    app.run(debug=True)
