#!/usr/bin/env python3

import os
import tempfile
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, render_template, request, send_file, flash, redirect, url_for
from flask_cors import CORS
from scipy.signal import savgol_filter
import io
import csv

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'

# Enable CORS for all routes (permissive for development)
CORS(app, origins='*', 
     methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization'])

# Model classes (same as inference script)
class ResidualBlock2(nn.Module):
    def __init__(self, dim: int, p: float = 0.2) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        return self.relu(out + identity)

class ResNetMLP2(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_blocks: int, dropout_rate: float) -> None:
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResidualBlock2(hidden_dim, p=dropout_rate) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.input_layer(x))
        x = self.blocks(x)
        x = self.output_layer(x)
        return x

# Global model variables
model = None
scaler_mean = None
scaler_scale = None
model_loaded = False

def load_model():
    global model, scaler_mean, scaler_scale, model_loaded
    
    try:
        ckpt_path = "/Users/nachu/space25/best_model_airs (1).pth"
        
        # Load checkpoint
        obj = torch.load(ckpt_path, map_location="cpu")
        if isinstance(obj, dict) and any(k.startswith("input_layer.") or k.startswith("output_layer.") for k in obj.keys()):
            ckpt = {"model_state_dict": obj}
        elif isinstance(obj, dict) and "model_state_dict" in obj:
            ckpt = obj
        else:
            raise ValueError("Unsupported checkpoint format")
        
        sd = ckpt["model_state_dict"]
        
        # Infer architecture from state dict
        if "input_layer.weight" in sd:
            in_w = sd["input_layer.weight"]
            hidden_dim = int(in_w.shape[0])
            input_dim = int(in_w.shape[1])
        else:
            hidden_dim = 256
            input_dim = 3
            
        if "output_layer.weight" in sd:
            output_dim = int(sd["output_layer.weight"].shape[0])
        else:
            output_dim = 282
            
        # Count blocks
        block_indices = []
        for k in sd.keys():
            if k.startswith("blocks."):
                try:
                    idx = int(k.split(".")[1])
                    block_indices.append(idx)
                except:
                    continue
        num_blocks = (max(block_indices) + 1) if block_indices else 80
        
        # Load scaler if available
        if "scaler_mean" in ckpt and "scaler_scale" in ckpt:
            scaler_mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
            scaler_scale = np.asarray(ckpt["scaler_scale"], dtype=np.float32)
        
        # Create and load model
        model = ResNetMLP2(input_dim, hidden_dim, output_dim, num_blocks, 0.3)
        model.load_state_dict(sd, strict=True)
        model.eval()
        
        model_loaded = True
        print(f"Model loaded successfully: input_dim={input_dim}, hidden_dim={hidden_dim}, output_dim={output_dim}, num_blocks={num_blocks}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        model_loaded = False

def predict_spectrum(transit_depth, rs, i):
    """Generate spectrum prediction for given features"""
    global model, scaler_mean, scaler_scale
    
    if not model_loaded:
        raise ValueError("Model not loaded")
    
    # Prepare input
    X = np.array([[transit_depth, rs, i]], dtype=np.float32)
    
    # Apply scaling if available
    if scaler_mean is not None and scaler_scale is not None:
        X = (X - scaler_mean) / np.where(scaler_scale == 0, 1.0, scaler_scale)
    
    # Predict
    with torch.no_grad():
        preds = model(torch.from_numpy(X)).numpy()
    
    # Apply smoothing and scaling
    if preds.shape[1] > 2:
        w = 13 if 13 % 2 == 1 else 15
        preds_smooth = savgol_filter(preds, window_length=w, polyorder=2, axis=1)
        preds = 0.7 * preds + 0.3 * preds_smooth
    
    # Scale output
    preds = preds * 0.0001
    
    return preds[0]  # Return first (and only) prediction

def calculate_instrumental_noise(spectrum_data):
    """Calculate realistic instrumental noise based on spectrum characteristics"""
    # FGS1 channel (first bin) - typically noisier
    fgs1_signal = spectrum_data[0]
    airs_signals = spectrum_data[1:]
    
    # Estimate noise from spectral scatter and instrumental characteristics
    # Use median absolute deviation for robust noise estimation
    mad_airs = np.median(np.abs(airs_signals - np.median(airs_signals)))
    airs_noise = 1.4826 * mad_airs  # Convert MAD to standard deviation
    
    # FGS1 noise is typically 2-3x higher due to shorter integration
    fgs1_noise = airs_noise * 2.5
    
    # Add photon noise component (Poisson statistics)
    photon_noise = np.sqrt(np.abs(fgs1_signal)) * 0.1
    fgs1_noise = np.sqrt(fgs1_noise**2 + photon_noise**2)
    
    return fgs1_signal, fgs1_noise, airs_noise

def detect_transit_signal(spectrum_data, transit_depth_input):
    """Multi-criteria transit detection algorithm"""
    fgs1_signal, fgs1_noise, airs_noise = calculate_instrumental_noise(spectrum_data)
    
    # Criterion 1: Signal strength relative to noise
    snr = fgs1_signal / fgs1_noise if fgs1_noise > 0 else 0
    
    # Criterion 2: Consistency across wavelength channels
    airs_mean = np.mean(spectrum_data[1:])
    airs_std = np.std(spectrum_data[1:])
    consistency_score = 1.0 - (airs_std / np.abs(airs_mean)) if airs_mean != 0 else 0
    
    # Criterion 3: Expected vs observed depth correlation
    expected_depth = transit_depth_input / 10000  # Convert ppm to fractional
    depth_correlation = 1.0 - abs(fgs1_signal - expected_depth) / max(expected_depth, 0.001)
    
    # Criterion 4: Spectral coherence (transit should be consistent across wavelengths)
    spectral_coherence = 1.0 - (np.std(spectrum_data) / np.mean(np.abs(spectrum_data)))
    
    # Criterion 5: Statistical significance using chi-squared test
    chi_squared = np.sum((spectrum_data - np.mean(spectrum_data))**2 / (airs_noise**2))
    p_value = 1.0 - (chi_squared / len(spectrum_data))  # Simplified p-value
    
    # Weighted detection score (0-1 scale)
    detection_score = (
        0.3 * min(snr / 5.0, 1.0) +           # SNR component (30%)
        0.25 * max(consistency_score, 0) +     # Consistency (25%)
        0.2 * max(depth_correlation, 0) +      # Depth correlation (20%)
        0.15 * max(spectral_coherence, 0) +    # Spectral coherence (15%)
        0.1 * min(p_value, 1.0)                # Statistical significance (10%)
    )
    
    # Detection threshold based on multiple criteria
    detected = detection_score > 0.6  # 60% confidence threshold
    
    # Confidence based on detection score
    confidence = min(detection_score * 100, 100)
    
    return {
        'snr': snr,
        'detection_score': detection_score,
        'detected': detected,
        'confidence': confidence,
        'consistency_score': consistency_score,
        'depth_correlation': depth_correlation,
        'spectral_coherence': spectral_coherence,
        'p_value': p_value,
        'fgs1_signal': fgs1_signal,
        'fgs1_noise': fgs1_noise,
        'airs_noise': airs_noise
    }

def analyze_atmospheric_signature_advanced(spectrum_data):
    """Advanced atmospheric signature analysis using spectral decomposition"""
    from scipy.signal import detrend, find_peaks
    from scipy.stats import skew, kurtosis
    
    # Remove baseline trend
    detrended = detrend(spectrum_data)
    
    # Calculate spectral statistics
    spectral_mean = np.mean(detrended)
    spectral_std = np.std(detrended)
    spectral_skew = skew(detrended)
    spectral_kurt = kurtosis(detrended)
    
    # Detect absorption features (negative peaks)
    negative_peaks, _ = find_peaks(-detrended, height=spectral_std*0.5, distance=3)
    absorption_strength = np.sum(-detrended[negative_peaks]) if len(negative_peaks) > 0 else 0
    
    # Detect emission features (positive peaks)
    positive_peaks, _ = find_peaks(detrended, height=spectral_std*0.5, distance=3)
    emission_strength = np.sum(detrended[positive_peaks]) if len(positive_peaks) > 0 else 0
    
    # Calculate spectral complexity (number of significant features)
    all_peaks = np.concatenate([negative_peaks, positive_peaks])
    spectral_complexity = len(all_peaks) / len(spectrum_data)
    
    # Atmospheric signature strength based on multiple criteria
    signature_score = (
        0.4 * min(abs(spectral_skew) / 2.0, 1.0) +      # Skewness (40%)
        0.3 * min(spectral_kurt / 3.0, 1.0) +           # Kurtosis (30%)
        0.2 * min(spectral_complexity * 10, 1.0) +      # Complexity (20%)
        0.1 * min((absorption_strength + emission_strength) / spectral_std, 1.0)  # Feature strength (10%)
    )
    
    # Classify atmospheric signature
    if signature_score > 0.7:
        signature_strength = 'Strong'
    elif signature_score > 0.4:
        signature_strength = 'Moderate'
    elif signature_score > 0.2:
        signature_strength = 'Weak'
    else:
        signature_strength = 'None'
    
    # Determine atmospheric type
    if absorption_strength > emission_strength * 1.5:
        atmospheric_type = 'Absorption-dominated'
    elif emission_strength > absorption_strength * 1.5:
        atmospheric_type = 'Emission-dominated'
    else:
        atmospheric_type = 'Mixed'
    
    return {
        'spectral_skewness': spectral_skew,
        'spectral_kurtosis': spectral_kurt,
        'spectral_complexity': spectral_complexity,
        'absorption_strength': absorption_strength,
        'emission_strength': emission_strength,
        'signature_score': signature_score,
        'signature_strength': signature_strength,
        'atmospheric_type': atmospheric_type,
        'num_absorption_features': len(negative_peaks),
        'num_emission_features': len(positive_peaks)
    }

def analyze_exoplanet_detection(spectrum_data, transit_depth_input):
    """Advanced multi-criteria exoplanet detection analysis"""
    
    # Multi-criteria transit detection
    detection_results = detect_transit_signal(spectrum_data, transit_depth_input)
    
    # Advanced atmospheric analysis
    atmospheric = analyze_atmospheric_signature_advanced(spectrum_data)
    
    # Calculate additional scientific metrics
    mean_depth = np.mean(spectrum_data)
    depth_variability = np.std(spectrum_data)
    max_depth = np.max(spectrum_data)
    min_depth = np.min(spectrum_data)
    
    # Spectral quality metrics
    spectral_snr = mean_depth / depth_variability if depth_variability > 0 else 0
    spectral_contrast = (max_depth - min_depth) / mean_depth if mean_depth > 0 else 0
    
    # Detection quality assessment based on detection score
    detection_score = detection_results['detection_score']
    if detection_score >= 0.8:
        detection_quality = 'Excellent'
    elif detection_score >= 0.7:
        detection_quality = 'Good'
    elif detection_score >= 0.6:
        detection_quality = 'Marginal'
    else:
        detection_quality = 'Poor'
    
    analysis = {
        'transit_depth_input': float(transit_depth_input),
        'fgs1_depth': float(detection_results['fgs1_signal']),
        'fgs1_sigma': float(detection_results['fgs1_noise']),
        'fgs1_noise': float(detection_results['fgs1_noise']),
        'airs_noise': float(detection_results['airs_noise']),
        'snr': float(detection_results['snr']),
        'detection_score': float(detection_score),
        'transit_detected': bool(detection_results['detected']),
        'confidence_percent': float(detection_results['confidence']),
        'detection_quality': detection_quality,
        'consistency_score': float(detection_results['consistency_score']),
        'depth_correlation': float(detection_results['depth_correlation']),
        'spectral_coherence': float(detection_results['spectral_coherence']),
        'p_value': float(detection_results['p_value']),
        'mean_depth': float(mean_depth),
        'depth_variability': float(depth_variability),
        'max_depth': float(max_depth),
        'min_depth': float(min_depth),
        'spectral_snr': float(spectral_snr),
        'spectral_contrast': float(spectral_contrast),
        'atmospheric_signature': atmospheric['signature_strength'],
        'atmospheric_type': atmospheric['atmospheric_type'],
        'signature_score': float(atmospheric['signature_score']),
        'spectral_skewness': float(atmospheric['spectral_skewness']),
        'spectral_kurtosis': float(atmospheric['spectral_kurtosis']),
        'spectral_complexity': float(atmospheric['spectral_complexity']),
        'absorption_strength': float(atmospheric['absorption_strength']),
        'emission_strength': float(atmospheric['emission_strength']),
        'num_absorption_features': int(atmospheric['num_absorption_features']),
        'num_emission_features': int(atmospheric['num_emission_features']),
        'detection_threshold': 0.6,  # 60% detection score threshold
        'wavelength_bins': len(spectrum_data)
    }
    
    return analysis

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        transit_depth = float(request.form['transit_depth'])
        rs = float(request.form['rs'])
        i = float(request.form['i'])
        
        # Validate inputs
        if transit_depth <= 0 or rs <= 0 or i < 0 or i > 90:
            flash('Invalid input values. Please check: transit_depth > 0, Rs > 0, 0 ≤ i ≤ 90')
            return redirect(url_for('index'))
        
        # Generate prediction
        spectrum = predict_spectrum(transit_depth, rs, i)
        
        # Perform detailed analysis
        analysis = analyze_exoplanet_detection(spectrum, transit_depth)
        
        # Create CSV in memory with analysis
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write spectrum data
        header = [f'y_{i}' for i in range(len(spectrum))]
        writer.writerow(header)
        writer.writerow(spectrum.tolist())
        
        # Add analysis section
        writer.writerow([])  # Empty row
        writer.writerow(['ADVANCED EXOPLANET DETECTION ANALYSIS'])
        writer.writerow(['Metric', 'Value', 'Units'])
        writer.writerow(['Transit Detected', 'YES' if analysis['transit_detected'] else 'NO', ''])
        writer.writerow(['Detection Quality', analysis['detection_quality'], ''])
        writer.writerow(['Detection Score', f"{analysis['detection_score']:.3f}", '0-1 scale'])
        writer.writerow(['SNR', f"{analysis['snr']:.2f}", ''])
        writer.writerow(['Confidence', f"{analysis['confidence_percent']:.1f}", '%'])
        writer.writerow(['Consistency Score', f"{analysis['consistency_score']:.3f}", '0-1 scale'])
        writer.writerow(['Depth Correlation', f"{analysis['depth_correlation']:.3f}", '0-1 scale'])
        writer.writerow(['Spectral Coherence', f"{analysis['spectral_coherence']:.3f}", '0-1 scale'])
        writer.writerow(['P-value', f"{analysis['p_value']:.3f}", 'statistical'])
        writer.writerow(['FGS1 Depth', f"{analysis['fgs1_depth']:.6f}", 'fractional'])
        writer.writerow(['FGS1 Noise', f"{analysis['fgs1_noise']:.6f}", 'fractional'])
        writer.writerow(['AIRS Noise', f"{analysis['airs_noise']:.6f}", 'fractional'])
        writer.writerow(['Mean Depth', f"{analysis['mean_depth']:.6f}", 'fractional'])
        writer.writerow(['Depth Variability', f"{analysis['depth_variability']:.6f}", 'fractional'])
        writer.writerow(['Spectral SNR', f"{analysis['spectral_snr']:.2f}", ''])
        writer.writerow(['Spectral Contrast', f"{analysis['spectral_contrast']:.3f}", ''])
        writer.writerow(['Atmospheric Signature', analysis['atmospheric_signature'], ''])
        writer.writerow(['Atmospheric Type', analysis['atmospheric_type'], ''])
        writer.writerow(['Signature Score', f"{analysis['signature_score']:.3f}", '0-1 scale'])
        writer.writerow(['Spectral Skewness', f"{analysis['spectral_skewness']:.3f}", ''])
        writer.writerow(['Spectral Kurtosis', f"{analysis['spectral_kurtosis']:.3f}", ''])
        writer.writerow(['Spectral Complexity', f"{analysis['spectral_complexity']:.3f}", ''])
        writer.writerow(['Absorption Strength', f"{analysis['absorption_strength']:.6f}", ''])
        writer.writerow(['Emission Strength', f"{analysis['emission_strength']:.6f}", ''])
        writer.writerow(['Num Absorption Features', f"{analysis['num_absorption_features']}", 'count'])
        writer.writerow(['Num Emission Features', f"{analysis['num_emission_features']}", 'count'])
        writer.writerow(['Detection Threshold', f"{analysis['detection_threshold']:.1f}", 'score'])
        writer.writerow(['Wavelength Bins', f"{analysis['wavelength_bins']}", 'count'])
        
        # Prepare file for download
        output.seek(0)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            tmp_file.write(output.getvalue())
            tmp_file_path = tmp_file.name
        
        return send_file(
            tmp_file_path,
            as_attachment=True,
            download_name=f'exoplanet_analysis_td{transit_depth}_rs{rs}_i{i}.csv',
            mimetype='text/csv'
        )
        
    except ValueError as e:
        flash(f'Input error: {str(e)}')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f'Prediction error: {str(e)}')
        return redirect(url_for('index'))

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    """Return detailed analysis as JSON"""
    if request.method == 'OPTIONS':
        response = app.make_response('')
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
    
    try:
        # Get form data
        transit_depth = float(request.form['transit_depth'])
        rs = float(request.form['rs'])
        i = float(request.form['i'])
        
        # Validate inputs
        if transit_depth <= 0 or rs <= 0 or i < 0 or i > 90:
            return {'error': 'Invalid input values'}, 400
        
        # Generate prediction
        spectrum = predict_spectrum(transit_depth, rs, i)
        
        # Perform detailed analysis
        analysis = analyze_exoplanet_detection(spectrum, transit_depth)
        
        # Add spectrum data to response (convert numpy types to Python types)
        analysis['spectrum'] = [float(x) for x in spectrum.tolist()]
        analysis['wavelength_bins'] = int(len(spectrum))
        
        # Convert all numpy types to Python types for JSON serialization
        for key, value in analysis.items():
            if hasattr(value, 'item'):  # numpy scalar
                analysis[key] = value.item()
            elif isinstance(value, np.ndarray):
                analysis[key] = value.tolist()
        
        response = app.make_response(analysis)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
        
    except Exception as e:
        error_response = app.make_response({'error': str(e)})
        error_response.headers['Access-Control-Allow-Origin'] = '*'
        return error_response, 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file uploaded')
            return redirect(url_for('index'))
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(url_for('index'))
        
        # Read CSV
        df = pd.read_csv(file)
        
        # Check required columns
        required_cols = ['transit_depth', 'Rs', 'i']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            flash(f'Missing required columns: {missing_cols}')
            return redirect(url_for('index'))
        
        # Generate predictions
        predictions = []
        for _, row in df.iterrows():
            spectrum = predict_spectrum(row['transit_depth'], row['Rs'], row['i'])
            predictions.append(spectrum)
        
        # Create output DataFrame
        pred_df = pd.DataFrame(predictions, columns=[f'y_{i}' for i in range(len(predictions[0]))])
        
        # Create CSV in memory
        output = io.StringIO()
        pred_df.to_csv(output, index=False)
        output.seek(0)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            tmp_file.write(output.getvalue())
            tmp_file_path = tmp_file.name
        
        return send_file(
            tmp_file_path,
            as_attachment=True,
            download_name='batch_exoplanet_spectra.csv',
            mimetype='text/csv'
        )
        
    except Exception as e:
        flash(f'Batch prediction error: {str(e)}')
        return redirect(url_for('index'))

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    if model_loaded:
        print("Starting Flask app...")
        port = int(os.environ.get('PORT', 5001))
        app.run(debug=False, host='0.0.0.0', port=port)
    else:
        print("Failed to load model. Please check the model file path.")

# For Railway deployment
app = app
