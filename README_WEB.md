# Insurance Cost Prediction - Web Interface

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Web Application
```bash
streamlit run app.py
```

Or on Windows, double-click `run_web_app.bat`

### 3. Access the Web Interface
The app will automatically open in your browser at `http://localhost:8501`

## 🌐 Web Interface Features

### 🏠 Home Page
- Overview of the project
- Model performance metrics
- Quick navigation to all features

### 📊 Model Performance
- Detailed performance metrics (R², RMSE, MAE)
- Performance interpretation
- Model accuracy visualization

### 🔮 Single Prediction
- Interactive form to input patient details
- Real-time cost prediction
- Input validation and error handling

### 📁 Batch Prediction
- Upload CSV files for bulk predictions
- Download results as CSV
- Data validation and preview
- Summary statistics and visualizations

### 📈 Data Explorer
- Interactive data exploration
- Statistical summaries
- Visualizations (histograms, bar charts, correlation matrix)
- Dataset insights

## 🎯 How to Use

### Single Prediction
1. Go to "Single Prediction" page
2. Adjust the sliders and dropdowns for:
   - Age (18-100)
   - Sex (male/female)
   - BMI (15.0-50.0)
   - Number of children (0-10)
   - Smoking status (yes/no)
   - Region (northeast/northwest/southeast/southwest)
3. Click "Predict Insurance Cost"
4. View the predicted cost

### Batch Prediction
1. Go to "Batch Prediction" page
2. Upload a CSV file with columns: `age`, `sex`, `bmi`, `children`, `smoker`, `region`
3. Click "Generate Predictions"
4. Download the results

### Data Exploration
1. Go to "Data Explorer" page
2. View dataset statistics and visualizations
3. Analyze patterns in the data

## 🔧 Technical Details

- **Framework**: Streamlit
- **Visualization**: Plotly
- **Model**: Random Forest (best performing)
- **Performance**: R² = 0.862, RMSE = $4,625, MAE = $2,534

## 📱 Browser Compatibility
- Chrome (recommended)
- Firefox
- Safari
- Edge

## 🛠️ Troubleshooting

### If the app doesn't start:
1. Make sure all dependencies are installed: `pip install -r requirements.txt`
2. Ensure you're in the correct directory: `cd "AI pro34"`
3. Check that the model files exist in `artifacts/` folder

### If predictions fail:
1. Make sure the model is trained: `python src/train.py`
2. Check that input data has the correct column names
3. Verify data types match the training data

## 📞 Support
If you encounter any issues, check the console output for error messages and ensure all required files are present.
