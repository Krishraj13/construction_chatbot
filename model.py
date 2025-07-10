import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score

# Globals
constructed_floors = set()
floor_material_cache = {}

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

def load_dataset():
    df = pd.read_csv('construction.csv')
    drop_cols = ['project_id', 'location', 'sequence_id', 'estimated_cost_usd', 'materials_sufficient', 'shortfall_details']
    df.drop(columns=drop_cols, inplace=True)
    avg_inventory = df[[  # Used for resetting
        'inventory_bricks_available', 'inventory_cement_bags_available',
        'inventory_sand_tons_available', 'inventory_steel_kg_available',
        'inventory_labor_hours_available'
    ]].mean().to_dict()
    return df, avg_inventory

def train_and_save_model(df):
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

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=100, batch_size=256, callbacks=[reduce_lr, early_stop], verbose=2)

    # 🔍 Evaluate and print accuracy
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("\n📊 Model Evaluation Metrics:")
    print(f" - Mean Squared Error: {mse:.2f}")
    print(f" - R² Score: {r2:.4f}")

    # 💾 Save model and preprocessor
    model.save('model.h5')
    joblib.dump(preprocessor, 'preprocessor.pkl')
    print("✅ Model and preprocessor saved.\n")

def load_model_and_preprocessor():
    model = load_model('model.h5')
    preprocessor = joblib.load('preprocessor.pkl')
    return model, preprocessor

def clamp_predictions(preds):
    return np.clip(np.round(preds), 0, 1e7).astype(int)

def get_user_input():
    print("\nEnter project details:")
    btype = input("Building Type (residential/office/mixed): ").strip().lower()
    tpa = float(input("Total Plot Area (in sqft): "))
    bfa = float(input("Building Footprint Area (in sqft): "))
    tf = int(input("Total number of floors: "))

    if btype == 'residential':
        rtype = input("Room type (2bed/3bed/others): ").strip().lower()
        hpf = int(input("Number of houses per floor: "))
    elif btype == 'office':
        rtype = input("Room type (number of rooms/cabins): ").strip()
        hpf = None
    elif btype == 'mixed':
        rtype = input("Room type (residential/commercial): ").strip().lower()
        hpf = int(input("Number of units per floor: "))
    else:
        print("Invalid building type.")
        exit()
    return btype, rtype, tpa, bfa, tf, hpf

def predict_materials(floor_num, btype, rtype, tpa, bfa, tf, hpf, model, preprocessor, inventory):
    floor_area = bfa

    if floor_num not in [0, tf - 1] and 'mid_floor' in floor_material_cache:
        return floor_material_cache['mid_floor']

    input_dict = {
        'building_type': [btype],
        'room_type': [rtype],
        'total_plot_area_sqft': [tpa],
        'building_footprint_sqft': [bfa],
        'floor_area_sqft': [floor_area],
        'total_floors': [tf],
        'floor_number': [floor_num],
        'inventory_bricks_available': [inventory['inventory_bricks_available']],
        'inventory_cement_bags_available': [inventory['inventory_cement_bags_available']],
        'inventory_sand_tons_available': [inventory['inventory_sand_tons_available']],
        'inventory_steel_kg_available': [inventory['inventory_steel_kg_available']],
        'inventory_labor_hours_available': [inventory['inventory_labor_hours_available']],
    }

    df_input = pd.DataFrame(input_dict)
    transformed = preprocessor.transform(df_input)
    preds = model.predict(transformed, verbose=0)[0]
    clamped = clamp_predictions(preds)
    result = dict(zip(target_cols, clamped))

    if floor_num not in [0, tf - 1]:
        floor_material_cache['mid_floor'] = result

    return result

def check_inventory(materials, inventory):
    shortfall = {}
    for k in materials:
        inv_k = 'inventory_' + k.split('_required')[0] + '_available'
        if materials[k] > inventory.get(inv_k, 0):
            shortfall[inv_k] = materials[k] - inventory.get(inv_k, 0)

    if shortfall:
        print("\n❗ Insufficient materials:")
        for k, v in shortfall.items():
            print(f" - {k}: need {v} more")

        choice = input("Do you want to manually update inventory to proceed? (yes/no): ").strip().lower()
        if choice == 'yes':
            for k in shortfall:
                try:
                    extra = int(input(f"Enter amount to add for {k}: "))
                    inventory[k] += extra
                except:
                    print("Invalid input.")
            return True
        else:
            print("❌ Cannot proceed without required materials.")
            return False
    return True

def deduct_inventory(materials, inventory):
    for k in materials:
        inv_k = 'inventory_' + k.split('_required')[0] + '_available'
        inventory[inv_k] -= materials[k]

def chatbot_loop(model, preprocessor, avg_inventory):
    global constructed_floors, floor_material_cache
    constructed_floors = set()
    floor_material_cache = {}
    inventory = {k: int(v) for k, v in avg_inventory.items()}

    print("\n🏗️  Welcome to Construction Planning Chatbot\n")
    btype, rtype, tpa, bfa, tf, hpf = get_user_input()

    while True:
        entry = input("\nEnter floor number or range (e.g., 0 or 1-3) or type 'exit': ").strip().lower()
        if entry == 'exit':
            print("\n✅ Exiting. Construction summary:")
            print("Constructed Floors:", sorted(constructed_floors))
            print("Remaining Inventory:")
            for k, v in inventory.items():
                print(f" - {k}: {v}")
            break

        if '-' in entry:
            start, end = map(int, entry.split('-'))
            floors = list(range(start, end + 1))
        else:
            floors = [int(entry)]

        max_constructed = max(constructed_floors) if constructed_floors else -1
        if any(f > max_constructed + 1 for f in floors):
            print(f"❗ You must construct floor {max_constructed + 1} first.")
            continue

        existing = [f for f in floors if f in constructed_floors]
        if existing:
            print(f"⚠️ Floor(s) {existing} already constructed. Skipping.")
            floors = [f for f in floors if f not in constructed_floors]
            if not floors:
                continue

        total_mats = {k: 0 for k in target_cols}
        floor_mats = {}

        for f in floors:
            mats = predict_materials(f, btype, rtype, tpa, bfa, tf, hpf, model, preprocessor, inventory)
            print(f"\n🔧 Floor {f} material prediction:")
            for k, v in mats.items():
                print(f" - {k}: {v}")
                total_mats[k] += v
            floor_mats[f] = mats

        print("\n📦 Total materials required:")
        for k, v in total_mats.items():
            print(f" - {k}: {v}")

        if not check_inventory(total_mats, inventory):
            continue

        deduct_inventory(total_mats, inventory)
        constructed_floors.update(floors)

        print(f"\n✅ Constructed floor(s): {floors}")
        print("📉 Remaining inventory:")
        for k, v in inventory.items():
            print(f" - {k}: {v}")

        # Optional logging of results
        log_df = pd.DataFrame([{'floor': f, **floor_mats[f]} for f in floors])
        log_df.to_csv('floor_log.csv', mode='a', index=False, header=not os.path.exists('floor_log.csv'))

        next_f = max(constructed_floors) + 1
        if next_f < tf:
            cont = input(f"\n➡️ Construct next floor {next_f}? (yes/no): ").strip().lower()
            if cont != 'yes':
                break
        else:
            print("\n🎉 All floors constructed! Project completed.")
            break

def main():
    df, avg_inventory = load_dataset()

    if not os.path.exists('model.h5') or not os.path.exists('preprocessor.pkl'):
        print("Training model since no saved model found...")
        train_and_save_model(df)
    else:
        print("Loading saved model and preprocessor...")

    model, preprocessor = load_model_and_preprocessor()
    chatbot_loop(model, preprocessor, avg_inventory)

if __name__ == '__main__':
    main()
