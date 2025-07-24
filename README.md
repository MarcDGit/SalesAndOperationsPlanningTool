# 🏭 Supply Chain Inventory Planning & Optimization Tool

A comprehensive web application for supply chain inventory planning with advanced analytics, outlier detection, optimization capabilities, and forecast accuracy evaluation. The application features two main modules: **Inventory Planning** for stock optimization and **Demand Planning** for forecast accuracy analysis.

## 🚀 Features

### 📤 Data Upload & Processing
- **Multi-format Support**: Upload data via CSV or Excel files
- **Flexible Schema**: Supports time series data with week, SKU, location, demand, and inventory columns
- **Data Validation**: Automatic validation of required columns and data structure
- **Real-time Preview**: Instant data overview with statistics and visualizations

### 🔍 Advanced Outlier Detection
- **Multiple Detection Methods**:
  - **Isolation Forest**: Machine learning-based anomaly detection
  - **Z-Score Method**: Statistical outlier detection (>3 standard deviations)
  - **IQR Method**: Interquartile range-based detection
- **Interactive Visualizations**: Highlight outliers in time series plots
- **Detailed Analysis**: Export outlier data for further investigation

### ⚙️ Inventory Optimization
- **Safety Stock Calculation**: Dynamic safety stock based on demand variability and service levels
- **Reorder Point Optimization**: Calculate optimal reorder points considering lead times
- **Inventory Turnover Analysis**: Comprehensive turnover metrics and days on hand calculations
- **Service Level Optimization**: Compare different service levels and their inventory impact

### 💡 Intelligent Parameter Proposals
- **Automated Recommendations**: System-generated inventory parameter suggestions
- **Service Level Impact Analysis**: Visualize how different service levels affect inventory requirements
- **Seasonal Decomposition**: Advanced time series analysis for seasonal patterns
- **Demand Pattern Recognition**: Identify trends, seasonality, and residual patterns

### 📊 ABC Analysis
- **Pareto Classification**: Automatic ABC categorization based on demand value
- **Category Analytics**: Detailed analysis by product category
- **Visual Dashboards**: Interactive charts and pie charts for category distribution
- **Strategic Insights**: Focus inventory efforts on high-value items

### 📈 Demand Planning & Forecast Accuracy
- **File Upload Support**: Excel (.xlsx, .xls) and CSV file uploads for forecast data
- **Comprehensive Metrics**: Calculate Bias, MAE, MAPE, RMSE, and industry-standard accuracy percentages
- **Industry-Standard Accuracy**: Forecast Accuracy (0-100%), Weighted Accuracy, and Tracking Signal
- **Time-Based Analysis**: Monthly trends and year-to-date performance tracking with accuracy benchmarks
- **Multi-Dimensional Filtering**: Filter by SKU, location, classification, and date range
- **Interactive Visualizations**: Plotly-based charts for trend analysis, comparisons, and accuracy scales
- **Performance Benchmarking**: Industry-standard forecast accuracy evaluation with quality thresholds

## 📋 Data Requirements

### Inventory Planning Data
#### Required Columns
- `week`: Date in YYYY-MM-DD format (weekly time periods)
- `sku`: Stock Keeping Unit identifier
- `location`: Warehouse or location identifier  
- `demand`: Numerical demand values

#### Optional Columns
- `inventory`: Current inventory levels (enables advanced turnover analysis)

#### Sample Data Format
```csv
week,sku,location,demand,inventory
2024-01-01,SKU001,Warehouse_A,150,500
2024-01-08,SKU001,Warehouse_A,200,350
2024-01-15,SKU001,Warehouse_A,180,170
```

### Demand Planning & Forecast Data
#### Required Columns
- `sku`: Product SKU identifier
- `actual_sales`: Actual sales values
- `forecast`: Forecasted sales values
- `date`: Date in YYYY-MM-DD format
- `location`: Store or location identifier
- `classification`: Product classification/category

#### Sample Data Format
```csv
sku,actual_sales,forecast,date,location,classification
SKU_001,120.5,115.2,2024-01-01,Store_A,Category_A
SKU_002,200.0,210.5,2024-01-01,Store_B,Category_B
SKU_003,85.3,88.7,2024-01-01,Store_C,Category_A
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone or download** the application files
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application**:
   ```bash
   streamlit run main.py
   ```
4. **Access the application** in your browser at `http://localhost:8501`

### Sample Data Files
The application includes sample data files for testing:
- `sample_inventory_data.csv`: Sample inventory planning data with demand and inventory levels
- `sample_forecast_data.csv`: Sample forecast accuracy data with actual vs. forecast comparisons

## 📖 User Guide

### Getting Started

#### For Inventory Planning:
1. **Navigate to Inventory Planning**: Select "🏭 Inventory Planning" from the sidebar
2. **Upload Your Data**: Go to "Data Upload & Overview" and upload your CSV/Excel file
3. **Explore Analytics**: Use the different analysis types in the sidebar navigation

#### For Demand Planning & Forecast Accuracy:
1. **Navigate to Demand Planning**: Select "📈 Demand Planning" from the sidebar
2. **Upload Forecast Data**: Upload your forecast data file (CSV or Excel format)
3. **Analyze Accuracy**: Review comprehensive forecast accuracy metrics and trends
4. **Filter & Drill Down**: Use filters to analyze specific segments or time periods

### Analysis Workflow

#### 1. Data Upload & Overview
- Upload your inventory data file
- Review data statistics and time series visualization
- Validate data structure and completeness

#### 2. Outlier Detection
- Choose detection method (Isolation Forest recommended for most cases)
- Review detected outliers in tabular and visual format
- Export outlier data for further investigation

#### 3. Inventory Optimization
- Set service level (90-99%) and lead time parameters
- Calculate safety stock requirements
- Determine optimal reorder points
- Analyze inventory turnover metrics

#### 4. Parameter Proposals
- Review system-generated parameter recommendations
- Compare different service level scenarios
- Analyze seasonal demand patterns
- Export recommendations for implementation

#### 5. ABC Analysis
- Perform Pareto analysis on your SKUs
- Review category classifications
- Focus on A-category items for maximum impact
- Use insights for strategic inventory planning

#### 6. Demand Planning & Forecast Accuracy Evaluation
- Upload forecast data with actual vs. predicted values
- Calculate comprehensive accuracy metrics (Bias, MAE, MAPE, RMSE)
- Analyze monthly trends and year-to-date performance
- Filter by SKU, location, or product classification
- Export accuracy reports and visualizations

## 🎯 Key Metrics & Calculations

### Safety Stock Formula
```
Safety Stock = Z-score(Service Level) × Demand Standard Deviation
```

### Reorder Point Formula
```
Reorder Point = (Average Demand × Lead Time) + Safety Stock
```

### Inventory Turnover
```
Turnover Ratio = Total Demand / Average Inventory
Days on Hand = 365 / (Turnover Ratio × 52)
```

### ABC Classification
- **A Items**: Top 80% of demand value (high priority)
- **B Items**: Next 15% of demand value (medium priority)  
- **C Items**: Bottom 5% of demand value (low priority)

### Forecast Accuracy Metrics

#### Traditional Metrics

##### Bias (Mean Error)
```
Bias = Average of (Forecast - Actual)
```
- **Interpretation**: Positive values indicate over-forecasting, negative values indicate under-forecasting

##### MAE (Mean Absolute Error)
```
MAE = Average of |Forecast - Actual|
```
- **Interpretation**: Lower values indicate better accuracy (same unit as sales data)

##### MAPE (Mean Absolute Percentage Error)
```
MAPE = Average of |Forecast - Actual| / |Actual| × 100
```
- **Interpretation**: Expressed as percentage; <10% excellent, 10-20% good

##### RMSE (Root Mean Square Error)
```
RMSE = √(Average of (Forecast - Actual)²)
```
- **Interpretation**: Penalizes large errors more heavily; lower values are better

#### Industry-Standard Accuracy Metrics (0-100% Scale)

##### Forecast Accuracy
```
Forecast Accuracy = 100% - MAPE
```
- **Interpretation**: Direct accuracy percentage; ≥90% excellent, 80-90% good, <80% needs improvement
- **Industry Standard**: Most widely used accuracy metric in supply chain and forecasting

##### Weighted Accuracy
```
Weighted Accuracy = 100% - (Average of Capped Relative Errors × 100)
```
- **Interpretation**: Sophisticated accuracy measure that caps extreme outliers at 200%
- **Usage**: More robust than standard accuracy for datasets with occasional large errors

##### Tracking Signal
```
Tracking Signal = Bias / MAE
```
- **Interpretation**: Bias detection metric; typical acceptable range is -4 to +4
- **Industry Standard**: Used to identify systematic forecasting bias
- **Thresholds**: <2 in control, 2-4 monitor, ≥4 out of control

## 🔧 Technical Architecture

### Built With
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning algorithms
- **SciPy**: Statistical functions
- **Statsmodels**: Time series analysis
- **OpenPyXL**: Excel file processing for forecast data uploads

### Key Components
- `InventoryOptimizer`: Core optimization and analysis engine
- `load_and_process_data()`: Data ingestion and preprocessing
- `validate_data_structure()`: Data validation and error handling
- `calculate_forecast_accuracy_metrics()`: Forecast accuracy calculations
- `process_forecast_data()`: Forecast data processing and analysis
- Interactive Streamlit interface with session state management

## 💡 Tips & Best Practices

### Data Preparation
- Ensure weekly data consistency (same day of week for all entries)
- Clean data before upload (remove null values, standardize formats)
- Include at least 12 weeks of data for meaningful analysis
- Use consistent SKU and location naming conventions

### Analysis Guidelines
- Start with ABC analysis to prioritize efforts
- Use Isolation Forest for initial outlier detection
- Adjust service levels based on business criticality
- Consider seasonal patterns when setting parameters
- Regular monitoring and parameter adjustment recommended

### Performance Optimization
- For large datasets (>10K records), consider data aggregation
- Use filters to analyze specific SKUs or locations
- Export results for offline analysis and reporting

### Forecast Accuracy Best Practices
- Ensure forecast data includes at least 3 months of historical comparisons
- Use consistent SKU naming between actual and forecast data
- Regular monitoring of accuracy trends helps identify forecasting issues
- Focus on MAPE for percentage-based comparisons across different SKUs
- Use bias metrics to identify systematic over or under-forecasting patterns

## 🤝 Support & Contributing

For questions, issues, or feature requests, please refer to the application's help sections or contact your system administrator.

### Feature Roadmap
- [x] **Forecast accuracy evaluation** ✅ (Completed)
- [x] **Multi-format data upload support** ✅ (Completed)
- [ ] Multi-echelon optimization
- [ ] Cost optimization algorithms
- [ ] Integration with ERP systems
- [ ] Advanced ML models for demand prediction
- [ ] Automated forecast bias correction
- [ ] Real-time forecast accuracy monitoring

## 📄 License

This application is designed for internal business use. Please ensure compliance with your organization's data handling policies.

---

*Built with ❤️ for supply chain professionals* 
