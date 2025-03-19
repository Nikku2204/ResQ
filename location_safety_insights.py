from flask import Flask, render_template, request, jsonify
import pandas as pd
from geopy.distance import geodesic
import openai
import os
from datetime import datetime
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Load safety datasets with error handling
try:
    safety_grid = pd.read_csv('safety_grid.csv')
    time_safety = pd.read_csv('time_safety_data.csv')
    incident_data = pd.read_csv('cleaned_safety_data.csv')
    print(f"Successfully loaded data files. Rows: Safety Grid={len(safety_grid)}, Time Safety={len(time_safety)}, Incidents={len(incident_data)}")
    
    # Print column names to help debug
    print(f"Safety Grid columns: {safety_grid.columns.tolist()}")
    print(f"Time Safety columns: {time_safety.columns.tolist()}")
except Exception as e:
    print(f"Error loading data: {e}")
    safety_grid, time_safety, incident_data = None, None, None

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    print("WARNING: OpenAI API key not found in environment variables")

# Helper functions
def get_safety_level(score):
    if score >= 80:
        return "Very Safe"
    elif score >= 60:
        return "Safe"
    elif score >= 40:
        return "Moderate Risk"
    elif score >= 20:
        return "High Risk"
    else:
        return "Very High Risk"

def create_safety_context(lat, lon, time_period=None):
    if safety_grid is None:
        return {"error": "Safety data not available"}

    safety_grid['distance'] = safety_grid.apply(
        lambda row: geodesic((lat, lon), (row['cell_center_y'], row['cell_center_x'])).kilometers, axis=1
    )

    closest_cell = safety_grid.loc[safety_grid['distance'].idxmin()]
    grid_id = closest_cell['grid_id']

    # Check if the column exists before trying to access it
    # For time_safety data
    if time_period and not time_safety.empty and 'time_period' in time_safety.columns:
        time_data = time_safety[(time_safety['grid_id'] == grid_id) & (time_safety['time_period'] == time_period)]
        # Use the correct column name from your dataset 
        if not time_data.empty:
            if 'safety_score' in time_data.columns:
                safety_score = time_data.iloc[0]['safety_score']
            else:
                # Try alternative columns
                for possible_column in ['score', 'Safety_Score', 'incident_score']:
                    if possible_column in time_data.columns:
                        safety_score = time_data.iloc[0][possible_column]
                        break
                else:
                    # Default to 50 (neutral safety score)
                    safety_score = 50
        else:
            # Get from closest_cell if time data not available
            safety_score = get_safety_score_from_cell(closest_cell)
    else:
        # Get from closest_cell directly
        safety_score = get_safety_score_from_cell(closest_cell)

    safety_level = get_safety_level(safety_score)

    radius = 0.01
    nearby = incident_data[
        (incident_data['Latitude'] > lat - radius) &
        (incident_data['Latitude'] < lat + radius) &
        (incident_data['Longitude'] > lon - radius) &
        (incident_data['Longitude'] < lon + radius)
    ] if incident_data is not None else pd.DataFrame()

    top_incidents = nearby['Offense'].value_counts().head(3).to_dict() if 'Offense' in nearby.columns else {}

    return {
        "location": {"latitude": lat, "longitude": lon},
        "time_period": time_period,
        "safety_score": safety_score,
        "safety_level": safety_level,
        "incident_analysis": {
            "total_incidents": len(nearby),
            "top_incidents": top_incidents
        }
    }

def get_safety_score_from_cell(cell):
    """Helper function to extract safety score from a cell using multiple possible column names."""
    for possible_column in ['safety_score', 'score', 'Safety_Score', 'incident_score', 'safety_score']:
        if possible_column in cell:
            return cell[possible_column]
    
    # If we get here, none of the expected columns were found
    print(f"Warning: No safety score column found in data. Available columns: {cell.index.tolist()}")
    
    # Check if we can compute a safety score from other columns
    if 'incident_score' in cell and 'severity_score' in cell:
        return (cell['incident_score'] * 100 + cell['severity_score'] * 100) / 2
    
    # Default to middle score if nothing suitable found
    return 50

def generate_safety_alert(context):
    system_prompt = """
    You are a personal safety assistant for the ResQ app. Provide practical and supportive safety recommendations.
    Avoid alarmist language.
    """

    safety_level = context.get('safety_level', 'Unknown')
    incident_types = list(context.get('incident_analysis', {}).get('top_incidents', {}).keys())
    incident_types_str = ", ".join(incident_types) if incident_types else "no specific incidents"

    user_prompt = f"""
    Safety Level: {safety_level} (Safety Score: {context.get('safety_score', 'N/A')}/100)
    Incident Types: {incident_types_str}
    Total Incidents Nearby: {context.get('incident_analysis', {}).get('total_incidents', 0)}
    """

    try:
        # Check if we're using the newer client format or older version
        try:
            # Try newer OpenAI client format
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=350
            )
            return response.choices[0].message.content.strip()
        except (AttributeError, ImportError):
            # Fall back to older format
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=350
            )
            return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return "Unable to generate a response at this time."

def getCurrentTimePeriod():
    hour = datetime.now().hour
    if hour >= 0 and hour < 6:
        return "Night (12AM-6AM)"
    elif hour >= 6 and hour < 12:
        return "Morning (6AM-12PM)"
    elif hour >= 12 and hour < 18:
        return "Afternoon (12PM-6PM)"
    else:
        return "Evening (6PM-12AM)"

# Flask routes
@app.route('/location_safety')
def location_safety_page():
    incidents = []
    if incident_data is not None:
        # Sample a subset of incidents for map display
        try:
            sample_size = min(500, len(incident_data))
            incident_sample = incident_data.sample(sample_size)
            incidents = incident_sample[['Latitude', 'Longitude', 'Offense']].dropna().to_dict(orient='records')
        except Exception as e:
            print(f"Error preparing incident data: {str(e)}")
    
    return render_template('location_safety.html', incidents=incidents)

@app.route('/api/get_location_safety', methods=['POST'])
def api_location_safety():
    data = request.json
    lat = data.get('latitude')
    lon = data.get('longitude')
    time_hour = data.get('time_hour')

    if lat is None or lon is None:
        return jsonify({"error": "Missing latitude or longitude"}), 400

    context = create_safety_context(lat, lon, time_hour)

    if "error" in context:
        return jsonify(context), 500

    recommendations = generate_safety_alert(context)

    return jsonify({
        'location': {'latitude': lat, 'longitude': lon},
        'safety': {
            'score': context['safety_score'],
            'level': context['safety_level'],
            'incident_summary': context['incident_analysis'],
        },
        'recommendations': recommendations
    })

@app.route('/api/get_safety_insight', methods=['POST'])
def api_safety_insight():
    data = request.json
    lat = data.get('latitude')
    lon = data.get('longitude')
    safety_level = data.get('safety_level')
    score = data.get('score')
    incident_summary = data.get('incident_summary', {})
    
    if lat is None or lon is None:
        return jsonify({"error": "Missing location data"}), 400
    
    try:
        system_prompt = """
        You are an AI safety assistant for the ResQ app. Create a brief, helpful safety insight for a user 
        concerned about their safety at a specific location. Your response should be personalized, practical, 
        and reassuring without minimizing risks. Focus on actionable advice that helps them make informed decisions.
        
        Format your response with:
        1. A brief assessment of the area (1-2 sentences)
        2. 2-3 specific, practical safety tips relevant to the time of day and incident types
        3. A reassuring closing note
        
        Keep your entire response under 150 words and focus on being helpful rather than alarming.
        """
        
        # Format incident data for better context
        incident_types = list(incident_summary.get('top_incidents', {}).keys())
        incident_types_str = ", ".join(incident_types) if incident_types else "no specific incidents"
        
        user_prompt = f"""
        Location: Coordinates ({lat}, {lon})
        Safety Level: {safety_level}
        Safety Score: {score}/100
        Incident Types: {incident_types_str}
        Total Incidents Nearby: {incident_summary.get('total_incidents', 0)}
        Current Time: {datetime.now().strftime('%H:%M')} ({getCurrentTimePeriod()})
        """
        
        # Try both new and old OpenAI API formats
        try:
            # Newer OpenAI client
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            insight = response.choices[0].message.content.strip()
        except (AttributeError, ImportError):
            # Fall back to older format
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            insight = response.choices[0].message.content.strip()
        
        return jsonify({"insight": insight})
    
    except Exception as e:
        print(f"Error generating safety insight: {e}")
        return jsonify({"insight": "Unable to generate a safety insight at this time. Please rely on the safety score and recommendations provided above."}), 500

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/test')
def test():
    return "<h1>Location Safety Testing Page</h1><p>This page confirms the server is running.</p>"

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Using a different port to avoid conflicts