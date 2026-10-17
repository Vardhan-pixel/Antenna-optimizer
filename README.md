# Antenna Optimization using AI/ML

An ML-driven tool that predicts the dimensions and performance of a **rectangular microstrip patch antenna**.
Given the operating frequency, substrate dielectric constant and substrate height, it predicts patch width, patch length, gain, bandwidth and S11, and shows the S11 return-loss curve. A PDF design report can also be downloaded.

Industrial training project at IIT (BHU) Varanasi, submitted to SKIT Jaipur.

![Web interface](screenshots/web_interface.png)

## Features
- Multi-output regression: 4 models compared (Linear Regression, Random Forest, Gradient Boosting, Neural Network)
- Best model: **Gradient Boosting** (average R² = 0.9375 on the test set)
- FastAPI backend with input-range validation
- Web interface with an S11 vs frequency plot (Chart.js)
- One-click PDF design report

## Results (test set)
| Output | RMSE | R² |
|---|---|---|
| Patch width (mm) | 1.107 | 0.9964 |
| Patch length (mm) | 0.972 | 0.9959 |
| Gain (dBi) | 0.408 | 0.9354 |
| Bandwidth (%) | 0.240 | 0.9880 |
| S11 min (dB) | 1.969 | 0.7716 |

## Project structure
| File | Purpose |
|---|---|
| `generate_antenna_data.py` | Generates the 2,000-sample synthetic dataset using patch-antenna design equations |
| `train_model.py` | Trains and compares models, saves the best one |
| `main.py` | FastAPI backend (`/health`, `/predict`, `/predict-report`) |
| `report_generator.py` | Builds the PDF design report |
| `index.html` | Web frontend |
| `*.pkl` | Saved model, scaler and metadata |

## How to run
```bash
pip install -r requirements.txt

# optional: regenerate data and retrain
python generate_antenna_data.py
python train_model.py

# start the backend
python main.py
```
Then open `index.html` in your browser (backend runs at http://127.0.0.1:8000, API docs at `/docs`).

## Limitations
- The dataset is **synthetic**, generated from standard design equations (Balanis) with added noise.
- Gain, bandwidth and S11 use empirical approximations, and the S11 curve is a modelled shape, not a full-wave simulation.
- Use results for initial sizing and verify with HFSS, CST or MATLAB before fabrication.

## Future scope
Train on real simulation or measured data, inverse design from target specs, GA/PSO optimization, more antenna types.

## Author
Vardhan Maharishi, B.Tech CSE, SKIT Jaipur
