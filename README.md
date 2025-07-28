# 🏭 Supply Chain Planning Suite

A comprehensive web application for end-to-end supply chain planning with advanced analytics, optimization capabilities, and real-time execution monitoring. The application features six integrated modules: **Inventory Planning**, **Demand Planning**, **Supply Planning**, **Financial Planning**, **S&OP (Sales & Operations Planning)**, and **S&OE (Sales & Operations Execution)**.

## 🚀 Features Overview

### 📤 Data Upload & Processing
- **Multi-format Support**: Upload data via CSV or Excel files across all modules
- **Flexible Schema**: Supports various data structures for different planning domains
- **Data Validation**: Automatic validation of required columns and data structure
- **Real-time Preview**: Instant data overview with statistics and visualizations

### 🔍 Advanced Analytics & Optimization
- **Multiple Detection Methods**: Outlier detection, trend analysis, and pattern recognition
- **Interactive Visualizations**: Comprehensive dashboards with Plotly-based charts
- **Machine Learning**: AI-powered forecasting and optimization algorithms
- **Industry Best Practices**: Implementation of proven supply chain methodologies

## 📋 Module Details

### 🏭 Inventory Planning
Advanced inventory optimization with outlier detection and parameter proposals.

**Key Features:**
- **Safety Stock Calculation**: Dynamic safety stock based on demand variability and service levels
- **Reorder Point Optimization**: Calculate optimal reorder points considering lead times
- **Inventory Turnover Analysis**: Comprehensive turnover metrics and days on hand calculations
- **ABC Analysis**: Pareto classification for strategic inventory focus
- **Outlier Detection**: Isolation Forest, Z-Score, and IQR methods

**Data Requirements:**
```csv
week,sku,location,demand,inventory
2024-01-01,SKU001,Warehouse_A,150,500
2024-01-08,SKU001,Warehouse_A,200,350
```

### 📊 Demand Planning
Comprehensive forecast accuracy evaluation and demand analytics.

**Key Features:**
- **Forecast Accuracy Metrics**: Bias, MAE, MAPE, RMSE, industry-standard accuracy percentages
- **Time-Based Analysis**: Monthly trends and year-to-date performance tracking
- **Multi-Dimensional Filtering**: Filter by SKU, location, classification, and date range
- **Performance Benchmarking**: Industry-standard forecast accuracy evaluation

**Data Requirements:**
```csv
sku,actual_sales,forecast,date,location,classification
SKU_001,120.5,115.2,2024-01-01,Store_A,Category_A
```

### 🚚 Supply Planning
Comprehensive supply chain planning with capacity optimization and procurement management.

**Key Features:**
- **Capacity Planning**: Calculate capacity requirements and utilization analysis
- **Procurement Optimization**: Supplier selection and cost optimization
- **Master Production Schedule (MPS)**: Production planning and scheduling
- **Supplier Analysis**: Performance evaluation and risk assessment
- **Supply Chain Metrics**: KPI monitoring and risk management

**Key Capabilities:**
- Capacity requirements planning with lead time considerations
- Multi-supplier procurement optimization based on cost, quality, and capacity
- Production scheduling with inventory balancing
- Supplier performance scoring and comparison
- Supply chain risk assessment and mitigation recommendations

### 💰 Financial Planning
Advanced financial planning with budget analysis and profitability optimization.

**Key Features:**
- **Budget Variance Analysis**: Plan vs actual performance tracking
- **Cash Flow Forecasting**: Predictive cash flow modeling with seasonality
- **Profitability Analysis**: Product-level margin analysis and optimization
- **Cost Structure Analysis**: Pareto analysis and cost optimization opportunities
- **Financial KPIs**: Comprehensive financial performance dashboards

**Key Capabilities:**
- Automated budget variance calculation and analysis
- Trend-based cash flow forecasting with seasonal adjustments
- Product profitability tracking and margin analysis
- Cost category optimization using 80/20 rule
- Financial performance dashboards with multiple metrics

### 📋 S&OP (Sales & Operations Planning)
Integrated planning with gap analysis, scenario planning, and consensus building.

**Key Features:**
- **Integrated Plan Review**: Demand, supply, and financial plan alignment
- **Gap Analysis**: Comprehensive variance analysis between plans
- **Scenario Planning**: Multiple scenario modeling with risk assessment
- **Consensus Building**: Stakeholder alignment and variability analysis
- **Executive Dashboard**: High-level KPIs and critical issue identification

**Key Capabilities:**
- Cross-functional plan integration and balance assessment
- Automated gap identification and action plan generation
- Multiple scenario analysis (optimistic, pessimistic, baseline)
- Stakeholder consensus measurement and improvement recommendations
- Executive-level reporting with critical issue flagging

### ⚡ S&OE (Sales & Operations Execution)
Real-time execution monitoring and performance management.

**Key Features:**
- **Real-time Dashboard**: Live execution performance monitoring
- **Execution Performance**: Plan vs actual tracking with variance analysis
- **KPI Monitoring**: Real-time KPI performance against targets
- **Issue Tracking**: Operational issue management and resolution tracking
- **Corrective Actions**: Automated action plan generation
- **Weekly Review**: Comprehensive weekly performance review

**Key Capabilities:**
- Real-time plan vs actual performance tracking
- KPI achievement monitoring with trend analysis
- Operational issue lifecycle management
- Automated corrective action recommendations
- Week-over-week performance comparison
- Executive-level execution dashboards

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
The application includes sample data files for testing all modules:
- `sample_inventory_data.csv`: Inventory planning data
- `sample_forecast_data.csv`: Demand planning data
- Each module generates its own sample data for demonstration

## 📖 User Guide

### Getting Started

#### Module Navigation:
1. **Select Module**: Choose from the sidebar navigation
2. **Upload Data**: Use sample data or upload your own files
3. **Explore Analytics**: Navigate through different analysis types
4. **Download Results**: Export analysis results and sample data

### Analysis Workflow by Module

#### Inventory Planning:
1. Data Upload & Overview → Outlier Detection → Inventory Optimization → Parameter Proposals → ABC Analysis

#### Demand Planning:
1. Data Upload → Accuracy Metrics Calculation → Trend Analysis → Filtering & Drill-down

#### Supply Planning:
1. Data Overview → Capacity Planning → Procurement Optimization → MPS → Supplier Analysis

#### Financial Planning:
1. Data Overview → Budget Variance → Cash Flow Forecasting → Profitability Analysis → Cost Analysis

#### S&OP:
1. Plan Review → Gap Analysis → Scenario Planning → Consensus Building → Executive Dashboard

#### S&OE:
1. Real-time Dashboard → Execution Performance → KPI Monitoring → Issue Tracking → Corrective Actions

## 🎯 Key Metrics & Calculations

### Inventory Metrics
```
Safety Stock = Z-score(Service Level) × Demand Standard Deviation
Reorder Point = (Average Demand × Lead Time) + Safety Stock
Turnover Ratio = Total Demand / Average Inventory
```

### Forecast Accuracy Metrics
```
Bias = Average of (Forecast - Actual)
MAE = Average of |Forecast - Actual|
MAPE = Average of |Forecast - Actual| / |Actual| × 100
Forecast Accuracy = 100% - MAPE
```

### Supply Chain Metrics
```
Capacity Utilization = Required Capacity / Available Capacity × 100
Supplier Score = Weighted Average of (Cost, Quality, Lead Time, Capacity)
```

### Financial Metrics
```
Gross Margin = (Revenue - COGS) / Revenue × 100
Budget Variance = Actual - Budget
Cash Flow Forecast = Historical Trend + Seasonal Adjustment
```

### S&OP Metrics
```
Plan Balance Score = % of periods with <5% volume gap
Volume Gap = Supply - Demand
Consensus Level = Based on coefficient of variation across stakeholders
```

### S&OE Metrics
```
Execution Score = 100 - |Variance %|
KPI Achievement = Actual / Target × 100
Issue Resolution Rate = Resolved Issues / Total Issues × 100
```

## 🔧 Technical Architecture

### Built With
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning algorithms
- **SciPy**: Statistical functions
- **Statsmodels**: Time series analysis
- **OpenPyXL**: Excel file processing

### Key Components
- **Modular Architecture**: Six independent but integrated modules
- **Class-based Design**: Separate planner classes for each domain
- **Session State Management**: Streamlit session state for data persistence
- **Error Handling**: Comprehensive error handling and validation
- **Sample Data Generation**: Built-in sample data for all modules

## 💡 Industry Best Practices

### Supply Chain Planning
- **Integrated Planning**: Cross-functional alignment across demand, supply, and financial plans
- **Exception Management**: Automated gap identification and action planning
- **Scenario Planning**: Multiple scenario analysis for risk management
- **Real-time Monitoring**: Continuous execution tracking and performance management

### Data Management
- **Data Quality**: Automated validation and cleansing
- **Standardization**: Consistent data formats across modules
- **Scalability**: Designed for large datasets and multiple SKUs/locations
- **Export Capabilities**: Results export for further analysis

### Performance Optimization
- **Efficient Algorithms**: Optimized calculations for large datasets
- **Parallel Processing**: Where applicable for improved performance
- **Memory Management**: Efficient data handling for large files
- **Caching**: Streamlit caching for improved user experience

## 📊 Sample Data Structures

### Inventory Planning Data
```csv
week,sku,location,demand,inventory
2024-01-01,SKU001,Warehouse_A,150,500
2024-01-08,SKU001,Warehouse_A,200,350
```

### Supply Planning Data
```csv
period,sku,location,demand,capacity,supplier,unit_cost
2024-01,SKU001,Plant_A,1000,1200,Supplier_A,25.50
```

### Financial Planning Data
```csv
period,category,budget,actual,type
2024-01,Sales Revenue,500000,485000,Budget
```

### S&OP Data
```csv
period,product_family,demand,supply,revenue,costs
2024-01,Family_A,1000,950,50000,30000
```

### S&OE Data
```csv
period,metric,plan,actual,target
2024-W01,sales,10000,9500,10000
```

## 🤝 Support & Contributing

### Feature Roadmap
- [x] **Inventory Planning with ABC Analysis** ✅
- [x] **Demand Planning with Forecast Accuracy** ✅
- [x] **Supply Planning with Capacity Optimization** ✅
- [x] **Financial Planning with Budget Analysis** ✅
- [x] **S&OP with Integrated Planning** ✅
- [x] **S&OE with Real-time Execution** ✅
- [ ] Advanced ML models for demand prediction
- [ ] Multi-echelon inventory optimization
- [ ] Integration with ERP systems
- [ ] Real-time data connectivity
- [ ] Advanced optimization algorithms
- [ ] Mobile-responsive dashboard

### Performance Guidelines
- For datasets >10K records, consider data aggregation
- Use filters for specific analysis focus
- Regular performance monitoring recommended
- Export results for offline analysis

### Best Practices
- **Data Preparation**: Ensure data consistency and completeness
- **Regular Monitoring**: Set up regular review cycles
- **Cross-functional Collaboration**: Use consensus features for alignment
- **Continuous Improvement**: Use gap analysis for process enhancement

## 📄 License

This application is designed for internal business use. Please ensure compliance with your organization's data handling policies.

---

*Built with ❤️ for supply chain professionals*

**Version**: 2.0 - Complete Planning Suite
**Last Updated**: December 2024 
