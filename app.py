from flask import Flask, request, render_template, url_for, redirect, jsonify
from google.cloud import vision
import requests
import os
import base64
import cv2
import openai
from dotenv import load_dotenv
from deepface import DeepFace
import imghdr
import concurrent.futures
from functools import partial
import json

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
SCRAPINGDOG_API_KEY = os.getenv("SCRAPINGDOG_API_KEY")
IMGUR_CLIENT_ID = os.getenv("IMGUR_CLIENT_ID")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS

app = Flask(__name__)
client = vision.ImageAnnotatorClient()

# Directories
upload_folder = "static/uploads"
fbi_image_folder = "static/fugitives/fbi"
interpol_image_folder = "static/fugitives/interpol"
static_images_folder = "static/images"  # Added for new static resources

os.makedirs(upload_folder, exist_ok=True)
os.makedirs(fbi_image_folder, exist_ok=True)
os.makedirs(interpol_image_folder, exist_ok=True)
os.makedirs(static_images_folder, exist_ok=True)  # Ensure images directory exists

# OpenAI analysis function
def analyze_search_results(search_results):
    """
    Analyze search results from Scrapingdog using OpenAI to generate a summary.
    
    Parameters:
    search_results (list): List of dictionaries containing search result data
    
    Returns:
    str: A quirky short summary of the person based on the search results
    """
    if not search_results or len(search_results) == 0:
        return "No information available for analysis."
    
    # Format the search results
    formatted_results = ""
    for i, result in enumerate(search_results[:10]):  # Limit to first 10 results
        formatted_results += f"{i+1}. Title: {result.get('title', 'No Title')}\n"
        formatted_results += f"   Link: {result.get('link', 'No Link')}\n\n"
    
    try:
        # Create client and make API call
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing profile images and search results. Provide concise information about the person in the image."},
                {"role": "user", "content": f"Here are the search results:\n{formatted_results}\n\nProvide a brief 2-3 sentence summary about the person in the image and a short recommendation regarding whether to trust/meet this person based ONLY on these search results."}
            ],
            max_tokens=150  # Reduced for shorter response
        )
        
        # Extract and return the response
        summary = response.choices[0].message.content
        return summary
    
    except Exception as e:
        print(f"Error calling OpenAI API: {str(e)}")
        return "Unable to generate analysis at this time. Please try again later."
    
# Convert any image to JPG format with better error handling
def convert_to_jpg(image_path):
    """Convert any image to JPG format with better error handling."""
    try:
        # If file doesn't exist or is empty, return the original path
        if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
            print(f"⚠️ Warning: Unable to convert {image_path} - File doesn't exist or is empty")
            return image_path
            
        # If already a JPG file, return the original path
        if image_path.lower().endswith(('.jpg', '.jpeg')):
            return image_path
            
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️ Warning: Unable to read {image_path} with OpenCV")
            return image_path
            
        # Create output JPG path
        jpg_path = os.path.splitext(image_path)[0] + ".jpg"
        
        # Save as JPG
        cv2.imwrite(jpg_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        print(f"✅ Converted {image_path} to {jpg_path}")
        return jpg_path
        
    except Exception as e:
        print(f"❌ Error converting {image_path} to JPG: {str(e)}")
        return image_path  # Return original if conversion fails

# Upload image to Imgur
def upload_to_imgur(image_path):
    headers = {"Authorization": f"Client-ID {IMGUR_CLIENT_ID}"}
    with open(image_path, "rb") as img:
        encoded_image = base64.b64encode(img.read()).decode('utf-8')
    response = requests.post("https://api.imgur.com/3/upload", headers=headers, data={'image': encoded_image})
    if response.status_code == 200:
        return response.json()['data']['link']
    return None

# Check if uploaded image matches FBI or Interpol fugitive images
def check_database(image_path, database_folder):
    """Checks the uploaded image against stored FBI or Interpol images using DeepFace."""
    print(f"🔍 Checking against database: {database_folder}")
    
    # Store candidates with their scores for better selection
    candidates = []
    
    for fugitive_image in os.listdir(database_folder):
        fugitive_image_path = os.path.join(database_folder, fugitive_image)
        
        # Skip if it's not a valid file
        if not os.path.isfile(fugitive_image_path):
            continue
            
        # Check file extension instead of using imghdr
        file_ext = os.path.splitext(fugitive_image_path)[1].lower()
        if file_ext not in ['.jpg', '.jpeg', '.png']:
            print(f"🚨 Skipping invalid file extension: {fugitive_image}")
            continue  
            
        # Extract name properly using splitext
        fugitive_name = os.path.splitext(fugitive_image)[0].replace("_", " ")
        
        # Skip if file doesn't exist or is empty
        if not os.path.exists(fugitive_image_path) or os.path.getsize(fugitive_image_path) == 0:
            print(f"❌ ERROR: Image not found or empty - {fugitive_image_path}")
            continue
        
        try:
            # Perform DeepFace Verification
            print(f"🔎 Comparing {image_path} with {fugitive_image_path}")
            result = DeepFace.verify(
                img1_path=image_path,
                img2_path=fugitive_image_path,
                model_name="VGG-Face",  # Try VGG-Face for better compatibility
                enforce_detection=False,
                detector_backend="opencv"
            )
            
            # Get the distance score (lower = more similar)
            distance = result.get("distance", 1.0)
            
            # Add to candidates list
            candidates.append({
                "name": fugitive_name,
                "distance": distance,
                "image_path": fugitive_image
            })
            
            print(f"Distance for {fugitive_name}: {distance}")
            
        except Exception as e:
            print(f"❌ DeepFace error for {fugitive_name}: {e}")
    
    # Sort candidates by distance (lowest first = best match)
    candidates.sort(key=lambda x: x["distance"])
    
    # Use a stricter threshold for better matching
    threshold = 0.65  # Adjusted threshold to reduce false positives
    
    # Check if best match is below threshold
    if candidates and candidates[0]["distance"] < threshold:
        best_match = candidates[0]
        print(f"✅ Match Found: {best_match['name']} (Confidence: {best_match['distance']})")
        
        # Return the image filename instead of the full URL
        return True, best_match["name"], best_match["image_path"], best_match["distance"]
        
    return False, None, None, None  # No match found

# Wrapper function for parallel processing
def check_database_wrapper(database_folder, image_path):
    """Wrapper function to make check_database work with concurrent.futures"""
    return check_database(image_path, database_folder)

# Scrapingdog Reverse Image Search
def perform_reverse_image_search(image_url):
    """Uses Scrapingdog to perform a reverse image search and retrieve matched pages."""
    params = {"api_key": SCRAPINGDOG_API_KEY, "url": image_url}
    response = requests.get("https://api.scrapingdog.com/google_lens", params=params)
    if response.status_code == 200:
        data = response.json()
        results = []
        if "lens_results" in data:
            for r in data["lens_results"]:
                results.append({
                    "title": r.get("title", "No Title"),
                    "link": r.get("link", "#"),
                    "thumbnail": r.get("thumbnail", "https://via.placeholder.com/150")
                })
        return results
    print("❌ Reverse Image Search failed:", response.status_code, response.text)
    return []

# Function to detect faces using Google Vision API
def detect_faces(filepath):
    with open(filepath, "rb") as image_file:
        image = vision.Image(content=image_file.read())
        response = client.face_detection(image=image)
    return len(response.face_annotations)

# Home Page
@app.route('/')
def home():
    return render_template("index.html")

# Profile Verification Page
@app.route('/profile_verification')
def profile_verification():
    return render_template("profile_verification.html")

# NEW ROUTES FOR TEAMMATE'S PAGES
@app.route('/hydrate')
def hydrate():
    return render_template("hydrate.html")

@app.route('/aware_ai_demo')
def aware_ai_demo():
    return render_template("aware_ai_demo.html")

# Main Upload Function: Handles image verification, searches & matching
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return render_template("result.html", message="No file uploaded.")
    
    file = request.files['file']
    if file.filename == '':
        return render_template("result.html", message="No file selected.")

    # Save uploaded image
    filepath = os.path.join(upload_folder, file.filename)
    file.save(filepath)
    filepath = convert_to_jpg(filepath)
    filename = os.path.basename(filepath)

    # Face detection
    faces_detected = detect_faces(filepath)

    # Upload to Imgur for Reverse Image Search
    imgur_url = upload_to_imgur(filepath)

    # Run FBI and Interpol checks in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Create partial functions for each database
        fbi_check = partial(check_database_wrapper, fbi_image_folder)
        interpol_check = partial(check_database_wrapper, interpol_image_folder)
        
        # Submit tasks to the executor
        fbi_future = executor.submit(fbi_check, filepath)
        interpol_future = executor.submit(interpol_check, filepath)
        
        # Get results when both are complete
        is_fbi_match, fbi_name, fbi_image_filename, fbi_confidence = fbi_future.result()
        is_interpol_match, interpol_name, interpol_image_filename, interpol_confidence = interpol_future.result()

    # Generate URLs in the main thread where we have access to the Flask app context
    fbi_image_url = url_for('static', filename=f"fugitives/fbi/{fbi_image_filename}") if is_fbi_match else None
    interpol_image_url = url_for('static', filename=f"fugitives/interpol/{interpol_image_filename}") if is_interpol_match else None

    # Perform Reverse Image Search
    search_results = perform_reverse_image_search(imgur_url) if imgur_url else []

    return render_template(
        "result.html",
        uploaded_image=filename,
        faces_detected=faces_detected,
        is_fbi_match=is_fbi_match,
        fbi_name=fbi_name,
        fbi_image_url=fbi_image_url,
        fbi_confidence=fbi_confidence,
        is_interpol_match=is_interpol_match,
        interpol_name=interpol_name,
        interpol_image_url=interpol_image_url,
        interpol_confidence=interpol_confidence,
        search_results=search_results
    )

# New route to analyze search results
@app.route('/analyze_results', methods=['POST'])
def analyze_results():
    # Get search results from the form
    search_results_json = request.form.get('search_results')
    
    # Convert JSON string to Python list
    search_results = json.loads(search_results_json)
    
    # Analyze the results
    analysis = analyze_search_results(search_results)
    
    return jsonify({"analysis": analysis})

# Debug route to test FBI matching and inspect fugitive images
@app.route('/debug_fbi_matching')
def debug_fbi_matching():
    """Debug route to test FBI matching and inspect fugitive images."""
    fbi_images = []
    
    # Get list of FBI images
    for img in os.listdir(fbi_image_folder):
        img_path = os.path.join(fbi_image_folder, img)
        if os.path.isfile(img_path):
            img_type = imghdr.what(img_path) or "unknown"
            file_size = os.path.getsize(img_path)
            
            fbi_images.append({
                "name": img,
                "path": img_path,
                "type": img_type,
                "size": file_size,
                "url": url_for('static', filename=f"fugitives/fbi/{img}")
            })
    
    # Render debug template with FBI image info
    return render_template(
        "debug.html",
        fbi_images=fbi_images,
        fbi_folder=fbi_image_folder
    )

# Test route to directly test matching against the FBI database
@app.route('/test_match', methods=['POST'])
def test_match():
    """Test route to directly test matching against the FBI database."""
    if 'test_image' not in request.files:
        return render_template("debug.html", error="No file uploaded.")
    
    file = request.files['test_image']
    if file.filename == '':
        return render_template("debug.html", error="No file selected.")

    # Save uploaded test image
    test_filepath = os.path.join(upload_folder, "test_" + file.filename)
    file.save(test_filepath)
    
    # Convert to JPG for consistency
    test_filepath = convert_to_jpg(test_filepath)
    
    # Run matching test
    is_match, name, image_filename, confidence = check_database(test_filepath, fbi_image_folder)
    
    # Generate URL in the main thread
    image_url = url_for('static', filename=f"fugitives/fbi/{image_filename}") if is_match else None
    
    # Get FBI images for display
    fbi_images = []
    for img in os.listdir(fbi_image_folder):
        img_path = os.path.join(fbi_image_folder, img)
        if os.path.isfile(img_path):
            img_type = imghdr.what(img_path) or "unknown"
            file_size = os.path.getsize(img_path)
            
            fbi_images.append({
                "name": img,
                "path": img_path,
                "type": img_type,
                "size": file_size,
                "url": url_for('static', filename=f"fugitives/fbi/{img}")
            })
    
    # Render debug template with test results
    return render_template(
        "debug.html",
        fbi_images=fbi_images,
        fbi_folder=fbi_image_folder,
        test_results={
            "is_match": is_match,
            "name": name,
            "image_url": image_url,
            "confidence": confidence,
            "test_image": url_for('static', filename=f"uploads/{os.path.basename(test_filepath)}")
        }
    )

# Image Repair Utility
@app.route('/repair_images')
def repair_images():
    """Utility route to repair corrupt image files in the FBI directory."""
    results = []
    
    # First, check and fix FBI images
    for image_name in os.listdir(fbi_image_folder):
        image_path = os.path.join(fbi_image_folder, image_name)
        
        if not os.path.isfile(image_path):
            continue
            
        # Check if OpenCV can read the image
        img = cv2.imread(image_path)
        if img is None:
            results.append(f"❌ Error reading {image_name} - Attempting repair...")
            
            # Try to fix the image by opening and resaving it
            try:
                # Create temporary name for fixed image
                fixed_path = os.path.join(fbi_image_folder, "fixed_" + image_name)
                
                # Open and resave using Pillow (which can sometimes fix corrupted headers)
                from PIL import Image
                try:
                    with Image.open(image_path) as img:
                        img.save(fixed_path)
                    
                    # If successful, replace the original with fixed version
                    os.remove(image_path)
                    os.rename(fixed_path, image_path)
                    results.append(f"✅ Successfully repaired {image_name}")
                except Exception as e:
                    results.append(f"⚠️ Failed to repair {image_name}: {str(e)}")
            except Exception as e:
                results.append(f"⚠️ Error during repair of {image_name}: {str(e)}")
        else:
            # Image is readable but might not be in optimal format, convert to JPG
            if not image_path.lower().endswith(('.jpg', '.jpeg')):
                try:
                    jpg_path = os.path.splitext(image_path)[0] + ".jpg"
                    cv2.imwrite(jpg_path, img)
                    os.remove(image_path)  # Remove original
                    results.append(f"✅ Converted {image_name} to JPG format")
                except Exception as e:
                    results.append(f"⚠️ Failed to convert {image_name} to JPG: {str(e)}")
            else:
                # Ensure proper JPG encoding by resaving
                try:
                    temp_path = os.path.join(fbi_image_folder, "temp_" + image_name)
                    cv2.imwrite(temp_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    os.remove(image_path)
                    os.rename(temp_path, image_path)
                    results.append(f"✅ Optimized {image_name}")
                except Exception as e:
                    results.append(f"⚠️ Failed to optimize {image_name}: {str(e)}")
    
    return render_template("repair_results.html", results=results)

# Custom route for handling the upload form directly from the homepage
@app.route('/direct_upload', methods=['POST'])
def direct_upload():
    # This route is specifically for handling uploads from the homepage form
    return upload_image()  # Reuse the existing upload_image function

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
