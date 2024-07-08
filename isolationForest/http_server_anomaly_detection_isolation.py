import os
import argparse
import json
import threading
import time
import numpy as np
from os.path import join, dirname, abspath
from io import BytesIO
from functools import partial
from http.server import HTTPServer, BaseHTTPRequestHandler
import scipy as sp
from scipy import stats
from joblib import load
import RPi.GPIO as GPIO
from sklearn.metrics import confusion_matrix, classification_report

# Configuración de la biblioteca RPi.GPIO
GPIO.setmode(GPIO.BCM)  # Usa la numeración de pines BCM
ANOMALY_PIN = 21        # Define el pin que deseas usar (por ejemplo, GPIO 18)
GPIO.setup(ANOMALY_PIN, GPIO.OUT)  # Configura el pin como salida

# Settings
DEFAULT_PORT = 1337
MODELS_PATH = 'models'
JOBLIB_MODEL_FILE = 'isolation.joblib'
MAX_MEASUREMENTS = 128      # Truncate measurements to this number
ANOMALY_THRESHOLD = 0.05    # An MSE over this will be considered an anomaly
TIMING_LOG_FILE = 'inference_times.txt'
TIMING_BATCH_SIZE = 20

# Global flag
server_ready = 0
inference_times = []

################################################################################
# Functions

# Function: extract specified features (MAD) from sample
def extract_features(sample, max_measurements=0):
    features = []

    # Truncate sample
    if max_measurements == 0:
        max_measurements = sample.shape[0]
    sample = sample[:max_measurements]

    # Median absolute deviation (MAD)
    features.append(stats.median_abs_deviation(sample))

    return np.array(features).flatten()

# Decode string to JSON and save measurements in a file
def parseSamples(json_str):
    global inference_times
    
    # Create a browsable JSON document
    try:
        json_doc = json.loads(json_str)
    except Exception as e:
        print('ERROR: Could not parse JSON |', str(e))
        return

    # Parse sample
    sample = []
    num_meas = len(json_doc['x'])
    for i in range(num_meas):
        sample.append([float(json_doc['x'][i]),
                       float(json_doc['y'][i]),
                       float(json_doc['z'][i])])

    # Calculate MAD for each axis
    feature_set = extract_features(np.array(sample), max_measurements=MAX_MEASUREMENTS)
    print("MAD:", feature_set)

    # Measure inference time
    start_time = time.time()
    # Make prediction from model
    pred = model.predict([feature_set])
    end_time = time.time()
    inference_time = end_time - start_time
    inference_times.append(inference_time)
    print("Inference Time:", inference_time)
    
    # Check if we have reached the batch size for logging
    if len(inference_times) >= TIMING_BATCH_SIZE:
        save_inference_times()
    
    print("Prediction:", pred)

    # Calculate decision function score (anomaly score)
    anomaly_score = model.decision_function([feature_set])
    print("Anomaly Score:", anomaly_score)

    # Compare the anomaly score with the threshold
    if anomaly_score < -ANOMALY_THRESHOLD:  # Thresholds may need to be adjusted based on decision_function output
        print("ANOMALY DETECTED!")
        GPIO.output(ANOMALY_PIN, GPIO.HIGH)  # Activa el pin
    else:
        print("Normal")
        GPIO.output(ANOMALY_PIN, GPIO.LOW)   # Desactiva el pin

    return

def save_inference_times():
    global inference_times
    with open(TIMING_LOG_FILE, 'a') as f:
        for t in inference_times:
            f.write(f"{t}\n")
    inference_times = []
    print(f"Saved {TIMING_BATCH_SIZE} inference times to {TIMING_LOG_FILE}")

def cleanup():
    GPIO.cleanup()

# Handler class for HTTP requests
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_GET(self):
        # Tell client if server is ready for a new sample
        self.send_response(200)
        self.end_headers()
        self.wfile.write(str(server_ready).encode())

    def do_POST(self):
        # Read message
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)

        # Respond with 204 "no content" status code
        self.send_response(204)
        self.end_headers()

        # Decode JSON and compute anomaly score
        parseSamples(body.decode('ascii'))

# Server thread
class ServerThread(threading.Thread):
    
    def __init__(self, *args, **kwargs):
        super(ServerThread, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def is_stopped(self):
        return self._stop_event.is_set()

################################################################################
# Main

# Parse arguments
parser = argparse.ArgumentParser(description='Server that saves data from IoT sensor node.')
parser.add_argument('-p', action='store', dest='port', type=int, default=DEFAULT_PORT, help='Port number for server')
args = parser.parse_args()
port = args.port

# Print versions
print('Numpy ' + np.__version__)
print('SciPy ' + sp.__version__)

# Load model
model = load(join(MODELS_PATH, JOBLIB_MODEL_FILE))

# Create server
handler = partial(SimpleHTTPRequestHandler)
server = HTTPServer(('', port), handler)
server_addr = server.socket.getsockname()
print('Server running at: ' + str(server_addr[0]) + ':' + str(server_addr[1]))

# Create thread running server
server_thread = ServerThread(name='server_daemon', target=server.serve_forever)
server_thread.daemon = True
server_thread.start()

# Store samples for given time
server_ready = 1
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print('Server shutting down')
finally:
    server.shutdown()
    server_thread.stop()
    server_thread.join()
    cleanup()
    save_inference_times()
    print('Cleanup complete')
