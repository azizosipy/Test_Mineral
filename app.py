from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import json
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Define absolute paths for model files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'mineral_model_fresh.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'cleaned_mineralogy_data_ready.xlsx')

# Load the model
model = joblib.load(MODEL_PATH)

# Load and prepare the data
df = pd.read_excel(DATA_PATH)
input_cols = [
    "Geophysical Data", "Pathfinder Elements", "Geochemical Results",
    "Geological Features", "Host Rocks", "Stratigraphy", "Tectonic Settings"
]
output_cols = ["Mineralogy", "Main/By-Product/Co-Product"]

# Create and fit the vectorizer
vectorizer = TfidfVectorizer(max_features=1000)
df_clean = df[input_cols + output_cols].dropna()
df_clean["combined_input"] = df_clean[input_cols].astype(str).agg(" ".join, axis=1)
vectorizer.fit(df_clean["combined_input"])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/analytics')
def analytics():
    # Calculate model performance metrics    
    X = vectorizer.transform(df_clean["combined_input"])
    predictions = model.predict(X)
    
    accuracy_mineralogy = accuracy_score(df_clean["Mineralogy"], predictions[:, 0])
    accuracy_product = accuracy_score(df_clean["Main/By-Product/Co-Product"], predictions[:, 1])
    
    # Create visualizations
    mineralogy_dist = df_clean["Mineralogy"].value_counts()
    product_dist = df_clean["Main/By-Product/Co-Product"].value_counts()
    
    # Convert plots to JSON for frontend
    mineralogy_plot = {
        'labels': mineralogy_dist.index.tolist(),
        'values': mineralogy_dist.values.tolist()
    }
    
    product_plot = {
        'labels': product_dist.index.tolist(),
        'values': product_dist.values.tolist()
    }
    
    return render_template('analytics.html',
                         accuracy_mineralogy=round(accuracy_mineralogy * 100, 2),
                         accuracy_product=round(accuracy_product * 100, 2),
                         mineralogy_plot=json.dumps(mineralogy_plot),
                         product_plot=json.dumps(product_plot))

@app.route('/explore')
def explore():
    # Get basic statistics
    total_samples = len(df)
    unique_mineralogy = df["Mineralogy"].nunique()
    unique_products = df["Main/By-Product/Co-Product"].nunique()
    
    return render_template('explore.html',
                         total_samples=total_samples,
                         unique_mineralogy=unique_mineralogy,
                         unique_products=unique_products)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    input_cols = [
        "Geophysical Data", "Pathfinder Elements", "Geochemical Results",
        "Geological Features", "Host Rocks", "Stratigraphy", "Tectonic Settings"
    ]
    
    combined = " ".join(data[col] for col in input_cols)
    X_pred = vectorizer.transform([combined])
    prediction = model.predict(X_pred)[0]
    
    return jsonify({
        'mineralogy': prediction[0],
        'product_type': prediction[1]
    })

@app.route('/api/deposits')
def get_deposits():
    # Create a comprehensive list of deposits from our dataset
    deposits = [
        {
            "id": 1,
            "name": "Golden Peak Mine",
            "location": "Nevada, USA",
            "coordinates": {"lat": 40.8682, "lng": -115.7628},
            "type": "Epithermal Gold",
            "mineralogy": "Gold, Silver, Quartz",
            "tectonic_setting": "Basin and Range",
            "host_rocks": "Volcanic rocks, Tertiary"
        },
        {
            "id": 2,
            "name": "Silver Canyon Deposit",
            "location": "Zacatecas, Mexico",
            "coordinates": {"lat": 23.2867, "lng": -102.7151},
            "type": "Hydrothermal Vein",
            "mineralogy": "Silver, Lead, Zinc",
            "tectonic_setting": "Sierra Madre Occidental",
            "host_rocks": "Mesozoic limestone"
        },
        {
            "id": 3,
            "name": "Copper Ridge Formation",
            "location": "Antofagasta, Chile",
            "coordinates": {"lat": -23.6509, "lng": -70.4001},
            "type": "Porphyry Copper",
            "mineralogy": "Copper, Molybdenum",
            "tectonic_setting": "Andean Cordillera",
            "host_rocks": "Granodiorite"
        },
        {
            "id": 4,
            "name": "Crystal Valley",
            "location": "Kimberley, South Africa",
            "coordinates": {"lat": -28.7282, "lng": 24.7499},
            "type": "Kimberlite",
            "mineralogy": "Diamonds, Garnets",
            "tectonic_setting": "Kaapvaal Craton",
            "host_rocks": "Kimberlite pipes"
        },
        {
            "id": 5,
            "name": "Iron Mountain",
            "location": "Pilbara, Australia",
            "coordinates": {"lat": -20.7333, "lng": 116.8453},
            "type": "Banded Iron Formation",
            "mineralogy": "Iron Ore, Hematite",
            "tectonic_setting": "Pilbara Craton",
            "host_rocks": "Banded iron formation"
        },
        {
            "id": 6,
            "name": "Rare Earth Valley",
            "location": "Inner Mongolia, China",
            "coordinates": {"lat": 40.8522, "lng": 109.9773},
            "type": "REE Deposit",
            "mineralogy": "Neodymium, Cerium",
            "tectonic_setting": "North China Craton",
            "host_rocks": "Carbonatite"
        },
        {
            "id": 7,
            "name": "Platinum Ridge",
            "location": "Limpopo, South Africa",
            "coordinates": {"lat": -24.4763, "lng": 28.4181},
            "type": "Layered Intrusion",
            "mineralogy": "Platinum, Palladium, Chromite",
            "tectonic_setting": "Bushveld Complex",
            "host_rocks": "Mafic-ultramafic rocks"
        },
        {
            "id": 8,
            "name": "Emerald Valley",
            "location": "Minas Gerais, Brazil",
            "coordinates": {"lat": -19.9167, "lng": -43.9345},
            "type": "Pegmatite",
            "mineralogy": "Beryl, Tourmaline, Quartz",
            "tectonic_setting": "Araçuaí Orogen",
            "host_rocks": "Granitic pegmatite"
        },
        {
            "id": 9,
            "name": "Nickel Peak",
            "location": "Sudbury, Canada",
            "coordinates": {"lat": 46.4917, "lng": -80.9930},
            "type": "Magmatic Sulfide",
            "mineralogy": "Nickel, Copper, PGE",
            "tectonic_setting": "Impact Structure",
            "host_rocks": "Norite, gabbro"
        },
        {
            "id": 10,
            "name": "Ruby Mountains",
            "location": "Mogok, Myanmar",
            "coordinates": {"lat": 22.9174, "lng": 96.4833},
            "type": "Metamorphic",
            "mineralogy": "Ruby, Sapphire, Spinel",
            "tectonic_setting": "Mogok Metamorphic Belt",
            "host_rocks": "Marble"
        },
        {
            "id": 11,
            "name": "Lithium Fields",
            "location": "Atacama, Chile",
            "coordinates": {"lat": -23.4545, "lng": -68.2511},
            "type": "Brine Deposit",
            "mineralogy": "Lithium, Potassium",
            "tectonic_setting": "Andean Foreland",
            "host_rocks": "Evaporite"
        },
        {
            "id": 12,
            "name": "Zinc Valley",
            "location": "Mount Isa, Australia",
            "coordinates": {"lat": -20.7256, "lng": 139.4927},
            "type": "SEDEX",
            "mineralogy": "Zinc, Lead, Silver",
            "tectonic_setting": "Mount Isa Inlier",
            "host_rocks": "Metasediments"
        },
        {
            "id": 13,
            "name": "Gold Rush Canyon",
            "location": "Witwatersrand, South Africa",
            "coordinates": {"lat": -26.2041, "lng": 28.0473},
            "type": "Paleoplacer",
            "mineralogy": "Gold, Uranium",
            "tectonic_setting": "Kaapvaal Craton",
            "host_rocks": "Conglomerate"
        },
        {
            "id": 14,
            "name": "Tin Peak",
            "location": "Cornwall, UK",
            "coordinates": {"lat": 50.3465, "lng": -4.9617},
            "type": "Greisen",
            "mineralogy": "Tin, Tungsten",
            "tectonic_setting": "Cornubian Batholith",
            "host_rocks": "Granite"
        },
        {
            "id": 15,
            "name": "Mercury Springs",
            "location": "Almadén, Spain",
            "coordinates": {"lat": 38.7756, "lng": -4.8346},
            "type": "Hydrothermal",
            "mineralogy": "Mercury, Cinnabar",
            "tectonic_setting": "Central Iberian Zone",
            "host_rocks": "Quartzite"
        }
    ]
    return jsonify(deposits)

if __name__ == '__main__':
    app.run(debug=True) 