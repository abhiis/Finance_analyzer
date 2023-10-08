# **FinanceAnalyzer**

## **Overview**

FinanceAnalyzer is a Python library designed to offer deep and intuitive financial insights derived from your bank statements. Powered by Pandas for data wrangling, Plotly for interactive visualizations, and Dash for web-based interactive reports, FinanceAnalyzer aims to be a comprehensive tool for understanding your financial situation.

## **Features**

### **Rich Web Interface**

Host your own Dash-based web application to upload bank statements and visualize data. Experience fade-in effects and other animation touches for a smooth user interface.

### **Analysis and Metrics**

1. **Monthly Spending Overview**
2. **Income vs Expenditure**
3. **Identify Unusual Expenses**
4. **Average Monthly Balance**
5. **Cash Flow Analysis**
6. **Seasonal Trends in Expenses**
7. **Month-over-Month Growth in Expenses**
8. **Top 5 Expenses of the Month**
9. **... and many more**

### **Interactive Plot Generation**

Use Plotly to dynamically generate plots. Customize your insights by selecting different qualitative and quantitative metrics to visualize.

### **Advanced File Handling**

Support for encrypted PDFs and various other file formats for data input.

## **Setup**

### **Requirements**

Make sure to install all the necessary packages listed in **`requirements.txt`**:

```bash

pip install -r requirements.txt

```

### **Installation**

To install FinanceAnalyzer, you can use pip:

```bash

pip install FinanceAnalyzer

```

### **Getting Started**

After installation, you can run the Dash app by executing:

```bash

python app.py

```

This will host the app locally, and you can navigate to **`http://localhost:7777`** (or whatever port is specified) in your web browser to use FinanceAnalyzer.

## **Features**

### **Analysis & Metrics**

Here are some of the analyses and metrics that the **`FinanceAnalyzer`** provides:

- **Monthly Spending Overview**: Breaks down monthly spending to identify trends and high expenditure areas.
- **Income vs Expenditure**: Monthly comparison to understand your financial health.
- **Budget vs Actuals**: Checks your actual spendings against the budget.
- **Savings Rate**: Calculates the rate of saving on a monthly basis.
- **MoM Growth in Expenses**: Measures Month-on-Month growth rate in your expenses.
- **Cash Flow Analysis**: Analysis of your net monthly cash flow.
- **Top 5 Expenses**: Highlights the top 5 categories where you are spending the most.
- **Seasonal Trends**: Identifies any seasonal spending trends.
- **Financial Health Score**: A composite score based on various metrics to assess your overall financial health.

### **Plotting**

The plotting in the FinanceAnalyzer is performed by the **`_generate_plot()`** method. This method is versatile and allows for different kinds of plots like line graphs, bar charts, and pie charts. The method is called internally when you execute the **`run()`** method.

## **Requirements**

- Python 3.x
- Dash
- Pandas
- Plotly
- And other dependencies

## **Contributing**

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update the tests as appropriate.

## **License**

[MIT](https://choosealicense.com/licenses/mit/)
