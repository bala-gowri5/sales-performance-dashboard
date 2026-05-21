# sales-performance-dashboard
A web-based sales analytics dashboard with data visualization and machine learning-based sales prediction.
# Sales Performance Dashboard

A web-based sales analytics dashboard that transforms raw sales data into meaningful business insights through interactive visualizations and machine learning-based sales prediction. The system enables users to analyze sales performance, monitor trends, and forecast future sales using a predictive model.

---

## Features

- Upload and analyze sales datasets (CSV)
- Interactive sales performance dashboard
- Region-wise sales analysis
- Category-wise sales distribution
- Total and average sales insights
- Data preprocessing and statistical analysis
- Machine Learning-based sales prediction
- Interactive visualizations using charts
- User-friendly web interface for business insights

---

## Tech Stack

### Frontend
- HTML5
- CSS3
- JavaScript
- Chart.js

### Backend
- Python
- Flask

### Data Analytics & Machine Learning
- Pandas
- NumPy
- Scikit-learn
- Pickle

---

## Project Structure

```bash
sales-performance-dashboard/
│── app.py
│── requirements.txt
│── Models/
│   └── sales_prediction_model.pkl
│── data/
│── templates/
│   ├── index.html
│   ├── dashboard.html
│   ├── predict.html
│   ├── insights.html
│   └── about.html
│── static/
│   ├── css/
│   ├── js/
│── screenshots/
│── README.md
```

---

## How It Works

1. Upload a sales dataset in CSV format  
2. The system preprocesses and analyzes the data  
3. Interactive charts visualize sales trends and distributions  
4. Sales insights are generated based on categories and regions  
5. The machine learning model predicts future sales values using user input  

---

## Dashboard Modules

### Sales Dashboard
Displays key business metrics such as:
- Total Sales
- Average Sales
- Region-wise Sales Analysis
- Category-wise Sales Distribution

### Prediction Module
Predicts future sales based on selected input parameters such as:
- Region
- Category
- Quantity

### Insights Module
Provides analytical insights to support data-driven business decisions.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/sales-performance-dashboard.git
cd sales-performance-dashboard
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

Open browser and visit:

```text
http://127.0.0.1:5000
```

---

## Screenshots

### Home Page
(Add Screenshot)

### Dashboard Analytics
(Add Screenshot)

### Sales Prediction Module
(Add Screenshot)

### Insights Page
(Add Screenshot)

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/load_data` | POST | Upload and process dataset |
| `/api/analyze` | POST | Generate sales analytics |
| `/api/predict_sales` | POST | Predict future sales |

---

## Future Improvements

- Real-time analytics support
- Advanced machine learning models for better prediction accuracy
- User authentication and role-based access
- Cloud deployment for scalability
- Enhanced business intelligence visualizations

---

## About the Project

The **Sales Performance Dashboard** was developed to simplify sales data analysis and support business decision-making through visualization and predictive analytics. By integrating data analytics techniques with machine learning, the system converts raw datasets into actionable insights, helping users understand performance trends and forecast future sales.

---

## Author

**BALAJI E**  
M.Sc Computer Science  
Sathyabama Institute of Science and Technology

GitHub: https://github.com/your-bala-gowri5

---

## License

This project is developed for educational and learning purposes.
