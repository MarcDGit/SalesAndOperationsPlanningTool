# 🏭 Supply Chain Inventory Planning & Optimization Tool

A comprehensive web application for supply chain inventory planning with advanced analytics, outlier detection, and optimization capabilities.

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

## 📋 Data Requirements

### Required Columns
- `week`: Date in YYYY-MM-DD format (weekly time periods)
- `sku`: Stock Keeping Unit identifier
- `location`: Warehouse or location identifier  
- `demand`: Numerical demand values

### Optional Columns
- `inventory`: Current inventory levels (enables advanced turnover analysis)

### Sample Data Format
```csv
week,sku,location,demand,inventory
2024-01-01,SKU001,Warehouse_A,150,500
2024-01-08,SKU001,Warehouse_A,200,350
2024-01-15,SKU001,Warehouse_A,180,170
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

## 📖 User Guide

### Getting Started

1. **Navigate to Inventory Planning**: Select "🏭 Inventory Planning" from the sidebar
2. **Upload Your Data**: Go to "Data Upload & Overview" and upload your CSV/Excel file
3. **Explore Analytics**: Use the different analysis types in the sidebar navigation

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

## 🔧 Technical Architecture

### Built With
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning algorithms
- **SciPy**: Statistical functions
- **Statsmodels**: Time series analysis

### Key Components
- `InventoryOptimizer`: Core optimization and analysis engine
- `load_and_process_data()`: Data ingestion and preprocessing
- `validate_data_structure()`: Data validation and error handling
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

## 🤝 Support & Contributing

For questions, issues, or feature requests, please refer to the application's help sections or contact your system administrator.

### Feature Roadmap
- [ ] Forecasting integration
- [ ] Multi-echelon optimization
- [ ] Cost optimization algorithms
- [ ] Integration with ERP systems
- [ ] Advanced ML models for demand prediction

## 📄 License

This application is designed for internal business use. Please ensure compliance with your organization's data handling policies.

---

*Built with ❤️ for supply chain professionals* 
