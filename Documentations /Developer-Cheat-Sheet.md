# Technical Cheat Sheet: MVP Implementation
## One-Page Reference for Developers

---

## SENSOR DATA COLLECTION

```
┌─────────────────────────────────────────────────┐
│ ACCELEROMETER (X, Y, Z)                         │
├─────────────────────────────────────────────────┤
│ • Range: ±16g (±156 m/s²)                       │
│ • Rate: 100 Hz (10ms per sample)                │
│ • Noise: ~0.05 m/s² RMS                        │
│ • Window: 2 seconds = 200 samples per axis      │
│ • Output: 3D acceleration vector                │
│ • Useful for: Motion intensity, jitter, tremor │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ GYROSCOPE (X, Y, Z)                             │
├─────────────────────────────────────────────────┤
│ • Range: ±2000 deg/s                            │
│ • Rate: 100 Hz (synchronized with accel)       │
│ • Noise: ~0.05 deg/s typical                   │
│ • Window: 2 seconds = 200 samples per axis     │
│ • Output: 3D angular velocity                   │
│ • Useful for: Rotation, tremor detection       │
└─────────────────────────────────────────────────┘

Sample Code (Flutter):
─────────────────────
import 'package:sensors_plus/sensors_plus.dart';

void collectMotionData() {
  accelerometerEvents.listen((event) {
    double ax = event.x;  // m/s²
    double ay = event.y;
    double az = event.z;
    // Store in buffer
  });
  
  gyroscopeEvents.listen((event) {
    double gx = event.x;  // deg/s or rad/s
    double gy = event.y;
    double gz = event.z;
    // Store in buffer
  });
}
```

---

## FEATURE EXTRACTION (20 TIME-DOMAIN FEATURES)

```
From 2-second window (200 samples per axis):

TIME-DOMAIN (Calculate directly from sensor values):
──────────────────────────────────────────────────
1. accel_mean        = avg(|magnitude|)
2. accel_std         = std(|magnitude|)  ← JITTER INDICATOR
3. accel_max         = max(|magnitude|)
4. accel_min         = min(|magnitude|)
5. accel_peak_peak   = max - min

6. velocity_mean     = integral(accel)    ← SPEED INDICATOR
7. velocity_std      = std(velocity)
8. velocity_max      = max(velocity)

9. gyro_mean         = avg(|magnitude|)
10. gyro_std         = std(|magnitude|)   ← TREMOR
11. gyro_max         = max(|magnitude|)
12. gyro_peak_peak   = max - min

13. accel_energy     = sum(magnitude²)
14. gyro_energy      = sum(magnitude²)

15-17. accel_corr_xy = correlation(ax, ay)  ← Movement coherence
       accel_corr_yz = correlation(ay, az)
       accel_corr_xz = correlation(ax, az)

18. zero_cross_rate  = count(sign changes)

19-20. dominant_accel_freq (FFT - optional for v1)
       spectral_entropy (FFT - optional for v1)

Python Implementation:
─────────────────────
import numpy as np
from scipy import signal

def extract_features(accel, gyro):
    """
    accel, gyro: (3, N) arrays [X, Y, Z accelerations]
    Returns: dict of 20 features
    """
    features = {}
    
    # Magnitude
    accel_mag = np.sqrt(np.sum(accel**2, axis=0))
    gyro_mag = np.sqrt(np.sum(gyro**2, axis=0))
    
    # Time-domain
    features['accel_mean'] = np.mean(accel_mag)
    features['accel_std'] = np.std(accel_mag)
    features['accel_max'] = np.max(accel_mag)
    features['accel_min'] = np.min(accel_mag)
    features['accel_peak_peak'] = np.max(accel_mag) - np.min(accel_mag)
    
    # Velocity (integration)
    velocity = np.cumsum(accel_mag) / 100  # 100 Hz sampling
    features['velocity_mean'] = np.mean(velocity)
    features['velocity_std'] = np.std(velocity)
    features['velocity_max'] = np.max(velocity)
    
    # Gyro features
    features['gyro_mean'] = np.mean(gyro_mag)
    features['gyro_std'] = np.std(gyro_mag)
    features['gyro_max'] = np.max(gyro_mag)
    
    # Energy
    features['accel_energy'] = np.sum(accel_mag**2)
    features['gyro_energy'] = np.sum(gyro_mag**2)
    
    # Correlations
    features['accel_corr_xy'] = np.corrcoef(accel[0], accel[1])[0, 1]
    features['accel_corr_yz'] = np.corrcoef(accel[1], accel[2])[0, 1]
    features['accel_corr_xz'] = np.corrcoef(accel[0], accel[2])[0, 1]
    
    # Zero crossing rate
    zero_crossings = np.sum(np.diff(np.sign(accel_mag)) != 0)
    features['zero_cross_rate'] = zero_crossings / len(accel_mag)
    
    return features
```

---

## COGNITIVE STATE CLASSIFICATION

### Option 1: Random Forest (Recommended for MVP v1)

```
Training:
─────────
Input:  20 features (time-domain)
Label:  cognitive_state ∈ {0: calm, 1: stressed, 2: confused}
Dataset: 500 samples minimum (50 users × 10 sessions)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,      # 100 trees
    max_depth=15,          # Prevent overfitting
    min_samples_split=5,
    random_state=42,
    n_jobs=-1              # Use all cores
)

model.fit(X_train, y_train)  # X: (N, 20), y: (N,)

Feature Importance:
───────────────────
importance = model.feature_importances_
# Typically: velocity_mean (10%), accel_std (8%), gyro_std (7%)

Inference (Real-time):
──────────────────────
features = extract_features(accel_buffer, gyro_buffer)  # (20,)
feature_vector = np.array([features[key] for key in feature_names])

prediction = model.predict([feature_vector])[0]  # 0, 1, or 2
confidence = model.predict_proba([feature_vector])[0]  # [0.7, 0.2, 0.1]

cognitive_state = {0: 'calm', 1: 'stressed', 2: 'confused'}[prediction]
```

### Option 2: SVM (If data < 200 samples)

```
from sklearn.svm import SVC

model = SVC(kernel='rbf', probability=True, C=1.0)
model.fit(X_train, y_train)

prediction = model.predict([feature_vector])[0]
confidence = np.max(model.predict_proba([feature_vector])[0])
```

### Option 3: CNN-LSTM (For v2 with >5000 samples)

```
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(32, 5, activation='relu', input_shape=(200, 6)),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy')
```

---

## HYBRID ADAPTATION LOGIC

```python
class AdaptationEngine:
    def __init__(self, user_prefs, thresholds):
        self.user_prefs = user_prefs      # Manual settings
        self.motion_model = load_model()  # Pre-trained classifier
        self.confidence_threshold = 0.75
    
    def decide_adaptation(self, accel, gyro):
        """
        Returns: dict with UI adaptation commands
        """
        # Step 1: Extract features from sensor data
        features = self.extract_features(accel, gyro)
        
        # Step 2: Predict cognitive state
        pred_state, confidence = self.motion_model.predict(features)
        
        # Step 3: Quality check
        if confidence < self.confidence_threshold:
            # Low confidence → Use manual prefs only
            return self.user_prefs
        
        # Step 4: Generate adaptive recommendations
        adaptations = self._generate_adaptations(pred_state)
        
        # Step 5: Blend with manual prefs (70% motion + 30% manual)
        blended = self._blend(adaptations, self.user_prefs, weight=0.7)
        
        return blended
    
    def _generate_adaptations(self, state):
        """Generate UI changes based on cognitive state"""
        if state == 'calm':
            return {'button_size': 50, 'layout_density': 100}
        
        elif state == 'stressed':
            # Reduce complexity
            return {
                'button_size': 70,
                'layout_density': 50,
                'show_guidance': True,
                'animation_speed': 'slow'
            }
        
        elif state == 'confused':
            # Maximize simplicity
            return {
                'button_size': 80,
                'layout_density': 25,
                'show_help_text': True,
                'steps_visible': 'one_at_a_time'
            }
    
    def _blend(self, auto_adapt, manual_prefs, weight=0.7):
        """Weighted blend of automatic and manual"""
        blended = {}
        for key in manual_prefs:
            if key in auto_adapt:
                auto_val = auto_adapt[key]
                manual_val = manual_prefs[key]
                # Blend: 70% automatic + 30% manual
                if isinstance(manual_val, (int, float)):
                    blended[key] = auto_val * weight + manual_val * (1 - weight)
                else:
                    blended[key] = auto_val  # Use auto for non-numeric
            else:
                blended[key] = manual_prefs[key]
        return blended
```

---

## QUALITY MONITORING & FALLBACK

```
Quality Assessment:
───────────────────
confidence = prediction_confidence_from_model
quality_level = confidence  # 0.0 to 1.0

IF quality_level >= 0.80:
    Mode = "FULL_HYBRID"    → Use all biometric signals
    Weight = 70% motion + 30% manual
    Adaptation = Active

ELSE IF quality_level >= 0.70:
    Mode = "PARTIAL_HYBRID"  → Use biometric with caution
    Weight = 50% motion + 50% manual
    Adaptation = Conservative

ELSE IF quality_level >= 0.60:
    Mode = "FALLBACK"       → Mostly manual, minimal auto
    Weight = 20% motion + 80% manual
    Show warning: "Sensor quality low"

ELSE:
    Mode = "FULL_FALLBACK"  → Manual only
    Weight = 0% motion + 100% manual
    Adaptation = Disabled
    Show error: "Using your settings"

Implementation:
───────────────
def apply_quality_based_fallback(confidence, auto_adapt, manual_prefs):
    if confidence >= 0.80:
        return blend(auto_adapt, manual_prefs, 0.70)
    elif confidence >= 0.70:
        return blend(auto_adapt, manual_prefs, 0.50)
    elif confidence >= 0.60:
        return blend(auto_adapt, manual_prefs, 0.20)
    else:
        return manual_prefs
```

---

## PERFORMANCE TARGETS (v1)

```
Accuracy:
─────────
Motion-Emotion classification: ≥75%
False positive rate: <15%
True positive rate for "confused": >70%

Latency:
────────
Sensor reading → Feature extraction: <100ms
Feature extraction → Prediction: <50ms
Prediction → UI update: <500ms
Total: <1s (acceptable for accessibility)

Battery:
────────
Baseline (no motion sensing): 100%
With 100 Hz sampling + feature extraction: 95-97%
Net additional drain: 3-5% per hour

Device Support:
───────────────
Android 6.0+: >90% of active devices
iOS 11+: >98% of active devices
Budget phones: 85% support (all have accelerometer)
```

---

## MANUAL PREFERENCE SCHEMA

```
{
  "font_size": 16,              // 12-28 pt
  "contrast_level": "normal",   // normal, high, maximum
  "layout_density": 100,        // 100%, 75%, 50%, 25%
  "button_size": 48,            // pixels (40-80)
  "animation_speed": "normal",  // off, slow, normal, fast
  "haptic_feedback": true,      // boolean
  "sound_effects": true,        // boolean
  "auto_adapt_enabled": true,   // boolean
  "last_updated": 1701590400   // timestamp
}

Storage (SQLite):
─────────────────
CREATE TABLE user_preferences (
  id INTEGER PRIMARY KEY,
  key TEXT NOT NULL,
  value TEXT NOT NULL,
  created_at TIMESTAMP,
  updated_at TIMESTAMP
);

Flutter Code:
─────────────
final prefs = await SharedPreferences.getInstance();
await prefs.setInt('font_size', 18);
int fontSize = prefs.getInt('font_size') ?? 16;  // With default
```

---

## TESTING CHECKLIST

```
Sensor Collection:
  [ ] Accelerometer data logged correctly
  [ ] Gyroscope data synchronized with accel
  [ ] 100 Hz sampling rate maintained
  [ ] Tested on 3+ device types

Feature Extraction:
  [ ] All 20 features computed
  [ ] No NaN or Inf values
  [ ] Features normalized (0 mean, unit variance)
  [ ] Feature importance analyzed

Model Performance:
  [ ] Random Forest trained on 500+ samples
  [ ] Cross-validation accuracy ≥75%
  [ ] Confusion matrix checked (no biased errors)
  [ ] Inference time <50ms

Integration:
  [ ] Features → Model → Adaptation flow works
  [ ] Hybrid blending produces expected values
  [ ] Fallback activated at confidence <0.75
  [ ] Quality warning displays correctly

User Testing:
  [ ] 10+ testers, 3+ tasks each
  [ ] SUS score collected
  [ ] No crashes or ANRs
  [ ] Battery drain measured
```

---

## DEPLOYMENT CHECKLIST

```
Before MVP v1 Release:
  [ ] Code reviewed (4-eyes)
  [ ] Privacy policies written
  [ ] Consent flows implemented
  [ ] Data deletion working
  [ ] Settings exported/imported
  [ ] Crash reporting enabled
  [ ] Analytics anonymized
  [ ] Documentation complete
  [ ] Beta tested on 50+ devices
  [ ] IRB approval for research (if needed)
  [ ] Accessibility audit passed
  [ ] Performance profiled (battery, memory)
```

---

## QUICK DECISION MATRIX

```
"Should we include X in MVP v1?"

Motion-Emotion Detection?
→ YES (core feature, all sensors available)

Manual Preferences?
→ YES (non-negotiable for accessibility)

Gaze Tracking?
→ NO (requires fine-tuning, do v2)

Context Awareness?
→ NO (nice-to-have, do v2)

CNN-LSTM Model?
→ NO (Random Forest sufficient + faster)

Cloud Sync?
→ NO (privacy first, local only for v1)

Offline Mode?
→ YES (already local-first)

Auto-Calibration (Gaze)?
→ N/A (gaze not in v1)

ML Personalization?
→ NO (collect data v1, build v2)
```

---

**Print this page. Keep it on your desk during development.**

**Version:** 1.0  
**Last Updated:** December 3, 2025  
**Status:** Ready to Code