from flask import Flask, render_template, request, redirect, session, url_for
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Set random seed for consistent behavior (optional)
import os, random, tensorflow as tf
os.environ['PYTHONHASHSEED'] = '42'
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

app = Flask(__name__)
app.secret_key = 'super_secret_key'

# Load saved model and preprocessor
model = load_model('model.h5')
preprocessor = joblib.load('preprocessor.pkl')

# Load dataset just for inventory averages
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

    if len(constructed) >= project['tf']:
        return redirect(url_for('summary'))

    if request.method == 'POST':
        if 'update_inventory' in request.form:
            for key in inventory.keys():
                add_qty = request.form.get(key)
                if add_qty:
                    inventory[key] += int(add_qty)
            session['inventory'] = to_native(inventory)
            return redirect(url_for('floor_input'))

        floor = int(request.form['floor'])

        if floor in constructed or floor >= project['tf']:
            return render_template('floor.html', error="Invalid or already constructed floor", floor=None)

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

        shortfall = {}
        for k in result:
            inv_k = 'inventory_' + k.split('_required')[0] + '_available'
            if result[k] > inventory.get(inv_k, 0):
                shortfall[inv_k] = result[k] - inventory.get(inv_k, 0)

        if shortfall:
            return render_template('add_inventory.html', shortfall=shortfall, floor=floor)

        for k in result:
            inv_k = 'inventory_' + k.split('_required')[0] + '_available'
            inventory[inv_k] -= result[k]

        constructed.append(floor)

        session['inventory'] = to_native(inventory)
        session['constructed_floors'] = to_native(constructed)

        if len(constructed) >= project['tf']:
            return redirect(url_for('summary'))

        return render_template('result.html', result=result, floor=floor)

    return render_template('floor.html')

@app.route('/summary')
def summary():
    constructed = to_native(session.get('constructed_floors', []))
    inventory = to_native(session.get('inventory', {}))
    return render_template('summary.html', constructed=constructed, inventory=inventory)

if __name__ == '__main__':
    app.run(debug=True)
