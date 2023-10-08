import warnings as ws
import re
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import pandas as pd
from dash import Dash, html, dcc, callback, Output, Input, State, dash_table
import locale
# Set to a locale that uses comma as a thousands separator
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

ws.filterwarnings("ignore")


class AccountStatementAnalysis:
    def __init__(self, file_df, params=None):
        self.df = file_df.copy()
        self.window = 3
        self.budget_dict = {
            'Food & Groceries': 60000,  # Budgeted $5000 for Food & Groceries
            'EMI': 93600,  # Budgeted $10000 for EMI
            'Shopping': 200000,  # Budgeted $3000 for Shopping
            'Travel': 20000,  # Budgeted $2000 for Travel
            # Budgeted $1500 for Utilities like electricity, water, etc.
            'Subscriptions': 100000,
            # Budgeted $1500 for Utilities like electricity, water, etc.
            'Cloud Servers': 100000,
            'withUPI': 300000
        }
        # Implement your own categorization logic here
        expense_categories = {
            'Food & Groceries': ['swiggy', 'zomato', 'simpl', 'food', 'grocery', 'blinkit', 'instamart'],
            'EMI': ['emi', 'loan', 'MPOKKET'],
            'Shopping': ['myntra', 'amazon', 'flipkart'],
            'Subscriptions': ['rentickle', 'spotify', 'music', 'youtube', 'netflix', 'prime', 'hotstar', 'apple', 'google', 'notion'],
            'Cloud Servers': ['hetzner'],
            'withUPI': ['upi']
        }
        income_sources = {'Recurring Deposits': ['salary', 'TEG INDIA PRIVATE', 'GD RESEARCH'],
                          'Direct Transfers': ['upi'],
                          'Interest': ['interest'],
                          'Refunds': ['swiggy', 'zomato', 'simpl', 'food', 'grocery', 'blinkit', 'instamart', 'emi', 'rentickle', 'spotify', 'music', 'youtube', 'netflix', 'prime', 'hotstar', 'apple', 'google', 'notion'],
                          'Account Verify': ['RVSL DT']
                          }
        self.categories_dict = {
            'expense_categories': expense_categories, 'income_sources': income_sources}
        self._preprocess_data()

    def eda(self, returns):
        # Total Transactions
        total_transactions_div = html.H4(
            f'There were {len(self.df)} transactions seen in the uploaded statement.')

        # UPI Usage
        upi_usage_div = html.H4(
            f'UPI Payments accounted for {round(self.upi_usage, 1)}%')

        # Unusual Expense in Months
        unsual_expense_in_month_div = self.flag_unusual_expenses()

        # Recurring Payments
        recurring_payments_div = self.find_recurring_payments()

        # Spending Spikes
        spikes_div = self.identify_spikes()

        # Busiest Day of the Month
        busy_day_div = self.busiest_day_of_month()

        # Transaction Frequencies
        transac_freq_div = self.transaction_frequencies()

        # Expense Predictions
        predictions_div = self.expense_predictions()

        # Financial Health Score
        financial_score_div = self.financial_health_score()

        # Top 5 Expenses
        top_5_expenses_div = self.top_5_expenses()
        if returns:
            return html.Div([
                total_transactions_div,              # Starting with a general overview
                financial_score_div,                  # Next, discuss financial health score
                upi_usage_div,                        # UPI usage to understand payment methods
                recurring_payments_div,               # Recurring payments for expense tracking
                # unsual_expense_in_month_div,          # Any unusual expenses
                spikes_div,                           # Months with spikes in spending
                # Busy days could indicate special expenses or habits
                busy_day_div,
                # Transaction frequency to understand the usage pattern
                transac_freq_div,
                top_5_expenses_div,                   # Ending with top expenses
                predictions_div                       # Future expense predictions if any
            ])

    def _preprocess_data(self):
        self._format_dates()
        self._convert_amounts_to_float()
        self._initialize_categories()
        self.find_recurring_deposits()
        self.process_categorisation()
        self.expand_categories()
        self.upi_analysis()

        self.monthly_spending_overview()
        self.budget_vs_actuals()
        self.category_vs_budget()
        self.income_vs_expenditure()
        self.average_monthly_balance()
        self.cash_flow_analysis()
        self.seasonal_trends()
        self.savings_rate()
        self.mom_growth()
        self.withdrawal_trend_analysis()
        self.deposit_trend_analysis()
        self.daywise_spending()
        self.weekwise_spending()
        self.rolling_average_spending()
        self.eda(returns=False)
        self.run()

    def expand_categories(self):
        self.df['expense_categories'] = [
            ','.join(x) if x else x for x in self.df['expense_categories']]
        self.df['income_sources'] = [
            ','.join(x) if x else x for x in self.df['income_sources']]
        return self

    def _format_dates(self):
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%d/%m/%Y')
        self.df['Value Date'] = pd.to_datetime(
            self.df['Value Date'], format='%d/%m/%Y')
        self.df['Month'] = self.df['Date'].dt.to_period('M')
        self.df['Day'] = self.df['Date'].dt.day

    def _convert_amounts_to_float(self):
        for col in ['Deposit Amount', 'Withdrawal Amount', 'Closing Balance*']:
            self.df[col] = [locale.atof(x)
                            for x in self.df[col].astype(str).values]

    def _initialize_categories(self):
        for col in ['Categories', 'expense_categories', 'income_sources']:
            self.df[col] = self.df.apply(lambda x: [], axis=1)

    def generate_plot(self, title, xlabel, ylabel, kind='bar'):
        data = None  # Initialize data to None

        # Decide which data to use based on the title
        if title == 'Monthly Spending Overview':
            data = self.monthly_spending
        elif title == 'Monthly Income vs Expenditure':
            data = self.monthly_comparison
        elif title == 'Monthly Average Balance':
            data = self.average_balance
        elif title == 'Cash Flow Analysis':
            data = self.monthly_comparison
        elif title == 'Seasonal Trends in Withdrawal Amount':
            data = self.seasonal_trends
        elif title == 'Budget vs Actuals':
            data = self.budget_vs_actuals
        elif title == 'Monthly Savings Rate':
            data = self.monthly_comparison_savings_rate['Savings Rate']
        elif title == 'MoM Growth in Expenses':
            data = self.mom_growth
        elif title == 'Top 5 Expenses of the Month':
            data = self.top_expenses['Withdrawal Amount']
        elif title == 'Monthly Withdrawal Trend Analysis':
            data = self.monthly_withdrawals
        elif title == 'Monthly Deposit Trend Analysis':
            data = self.monthly_deposits
        elif title == 'Daywise Spending':
            data = self.daywise_expense
        elif title == 'Weekwise Spending':
            data = self.weekwise_expense
        elif title == 'Expense categories vs Budget':
            data = self.comparison_df
        elif title == f'Rolling Average Spending (Window: {self.window}) Days':
            data = self.rolling_average
        elif title == 'Income Categories':
            data = self.income_by_category
        elif title == 'Expense Categories':
            data = self.expense_by_category
        else:
            print("Unknown title. Cannot generate plot.")
            return

        if kind == 'bar':
            # Convert Period to string if needed
            if isinstance(data.index, pd.PeriodIndex):
                data.index = data.index.astype(str)
            fig = px.bar(data, title=title)
            fig.update_xaxes(title_text=xlabel)
            fig.update_yaxes(title_text=ylabel)
            fig.update_layout(
                hovermode='closest',
                title={
                    'y': 0.9,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                font=dict(
                    family="Courier New, monospace",
                    size=12,
                    color="#7f7f7f"
                )
            )
        elif kind == 'line':
            # Convert Period to string if needed
            if isinstance(data.index, pd.PeriodIndex):
                data.index = data.index.astype(str)
            fig = px.line(data, title=title)
            fig.update_xaxes(title_text=xlabel)
            fig.update_yaxes(title_text=ylabel)
            fig.update_layout(
                hovermode='closest',
                title={
                    'y': 0.9,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                font=dict(
                    family="Courier New, monospace",
                    size=12,
                    color="#7f7f7f"
                )
            )
        elif kind == 'pie':
            # Explicitly specify values
            fig = px.pie(data, names=data.index,
                         values=data.values, title=title)
            fig.update_traces(textinfo='percent+label')
            fig.update_layout(
                hovermode='closest',
                title={
                    'y': 0.9,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                font=dict(
                    family="Courier New, monospace",
                    size=12,
                    color="#7f7f7f"
                )
            )
        return dcc.Graph(figure=fig)
        # fig.show()

    def explore_narration(self):
        # Combine all the cleaned narrations into a single string
        combined_text = '\t'.join(self.df['Narration'])

        # Tokenize the combined text into individual words
        words = combined_text.split('\t')

        # Compute the frequency distribution of the words
        freq_dist = Counter(words)

        # Sort by frequency
        self.sorted_freq_dist = sorted(
            freq_dist.items(), key=lambda x: x[1], reverse=True)

    def monthly_spending_overview(self):
        self.monthly_spending = self.df.groupby(
            'Month')['Withdrawal Amount'].sum()

    def find_recurring_deposits(self):
        # Sorting by Date and resetting index
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df.sort_values(by='Date', inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        # Initialize an empty DataFrame to store recurring deposits
        recurring_deposits = pd.DataFrame()

        # Group by 'Deposit Amount'
        grouped = self.df.groupby('Deposit Amount')

        # Iterate through each group
        for name, group in grouped:
            group.sort_values(by='Date', inplace=True)
            # Calculate time difference in days
            group['Time Difference'] = group['Date'].diff().dt.days
            approx_monthly = group['Time Difference'].between(
                25, 35)  # Assuming a month is between 25 and 35 days

            # Check if there are more than 2 occurrences with an approximately monthly frequency
            if approx_monthly.sum() >= 2:
                recurring_deposits = pd.concat([recurring_deposits, group])
        self.recurring_deposit_narrations = recurring_deposits['Narration'].unique(
        )

    def process_categorisation(self):
        for identity, category in self.categories_dict.items():
            self.categorize_expense(category, identity)

    def categorize_expense(self, categories, identity):
        if identity == 'income_sources':
            transact_column = 'Deposit'
            # Find recurring transactions by text, and identify salary
            [categories['Recurring Deposits'].append(
                x) for x in self.recurring_deposit_narrations]
        elif identity == 'expense_categories':
            transact_column = "Withdrawal"
        else:
            return print("No valid identity found for category selection between income and expense.")
        df = self.df.loc[self.df[f'{transact_column} Amount'] != 0]
        df[identity] = df[identity].apply(lambda x: [] if pd.isnull(x) else x)
        for category, keywords in categories.items():
            for keyword in keywords:
                mask = df['Narration'].str.contains(
                    keyword, case=False, na=False)
                df.loc[mask, identity] = df.loc[mask, 'Categories'].apply(
                    lambda x: [category] if category not in x else x)

        if identity == 'income_sources':
            transact_column = 'Deposit'
            self.income_by_category = df.explode(identity).groupby(
                identity)[f'{transact_column} Amount'].sum()

        elif identity == 'expense_categories':
            transact_column = "Withdrawal"
            self.expense_by_category = df.explode(identity).groupby(
                identity)[f'{transact_column} Amount'].sum()
        self.df.update(df)

    def flag_unusual_expenses(self):
        self.monthly_expense = self.df.groupby(
            'Month')['Withdrawal Amount'].sum()
        z_scores = stats.zscore(self.monthly_expense)
        abs_z_scores = abs(z_scores)
        self.unusual_months = (abs_z_scores > 2)

        # List to hold the HTML components
        content_list = [html.H1("Unusually expensive month(s)")]

        for month, expense in self.monthly_expense[self.unusual_months].items():
            # Add the month and expense to the content list
            content_list.append(html.P(f"Month: {month}"))
            content_list.append(html.P(f"Expense Value: {expense}"))

            # If you also want to include the narration, assuming you have it filtered from your DataFrame
            narrations = self.df.loc[(self.df['Month'] == month) & (
                self.df['Withdrawal Amount'] == expense), 'Narration']
            for narration in narrations:
                content_list.append(html.P(f"Narration: {narration}"))

        # Wrap all components in a Div and return
        return html.Div(content_list)

    def income_vs_expenditure(self):
        self.monthly_comparison = self.df.groupby('Month').agg(
            {'Deposit Amount': 'sum', 'Withdrawal Amount': 'sum'})

    def find_recurring_payments(self):
        # Find narrations with recurring payments
        recurring_payments = self.df[self.df['Withdrawal Amount'].notnull(
        )]['Narration'].value_counts()
        self.recurring_payments = recurring_payments[recurring_payments > 1]

        # Filter the DataFrame for these recurring payments
        filtered_df = self.df[self.df['Narration'].isin(
            self.recurring_payments.index)]

        # Sum the amounts for each narration
        summed_amounts = filtered_df.groupby(
            'Narration')['Withdrawal Amount'].sum()

        # Generate the content
        content_list = [html.H1("These are your recurring payments")]
        content_list.append(html.H4("Identify your subscriptions here"))

        for narration, amount in summed_amounts.items():
            content_list.append(html.P(f"{narration} - {amount}"))

        return html.Div(content_list)

    def identify_spikes(self):
        self.monthly_spending = self.df.groupby(
            'Month')['Withdrawal Amount'].sum()
        self.spikes = self.monthly_spending[self.monthly_spending >
                                            self.monthly_spending.mean() + 2 * self.monthly_spending.std()]

        # List to hold the HTML components
        content_list = [html.H1("These are your spending spikes")]

        for month, expense in self.spikes.items():
            # Add the month and expense to the content list
            content_list.append(
                html.P(f"Spike in Spending for Month: {month}"))
            content_list.append(html.P(f"Spending Value: {expense}"))

            # If you also want to include the narration, assuming you have it filtered from your DataFrame
            narrations = self.df.loc[(self.df['Month'] == month) & (
                self.df['Withdrawal Amount'] == expense), 'Narration']
            for narration in narrations:
                content_list.append(html.P(f"Narration: {narration}"))

        # Wrap all components in a Div and return
        return html.Div(content_list)

    def average_monthly_balance(self):
        self.average_balance = self.df.groupby(
            'Month')['Closing Balance*'].mean()

    def busiest_day_of_month(self):

        self.busiest_day = self.df['Day'].value_counts().idxmax()
        return html.H1(f"Your busiest day seems to be {self.busiest_day} day of month.")

    def transaction_frequencies(self):
        self.withdrawal_count = self.df[self.df['Withdrawal Amount'] > 0].shape[0]
        self.deposit_count = self.df[self.df['Deposit Amount'] > 0].shape[0]

        # Create a DataFrame for the frequency data
        freq_df = pd.DataFrame({
            'Transaction Type': ['Withdrawal', 'Deposit'],
            'Count': [self.withdrawal_count, self.deposit_count]
        })

        # Create the figure using Plotly Express
        fig = px.pie(freq_df, names='Transaction Type', values='Count',
                     title='Transaction Frequencies')

        # Return a Dash dcc.Graph object
        return dcc.Graph(
            id='transaction-frequencies',
            figure=fig
        )

    def cash_flow_analysis(self):
        self.monthly_comparison = self.df.groupby('Month').agg(
            {'Deposit Amount': 'sum', 'Withdrawal Amount': 'sum'})
        self.monthly_comparison['Net Cash Flow'] = self.monthly_comparison['Deposit Amount'] - \
            self.monthly_comparison['Withdrawal Amount']

    def seasonal_trends(self):
        self.seasonal_trends = self.df.groupby(
            'Month')['Withdrawal Amount'].mean()

    def budget_vs_actuals(self):
        transact_column = "Withdrawal"
        df = self.df.loc[self.df[f'{transact_column} Amount'] != 0]
        actual_expense_by_category = df.explode('expense_categories').groupby(
            'expense_categories')[f'{transact_column} Amount'].sum()
        budget_vs_actuals = pd.DataFrame(
            list(self.budget_dict.items()), columns=['Category', 'Budget'])
        budget_vs_actuals.set_index('Category', inplace=True)
        budget_vs_actuals['Actuals'] = actual_expense_by_category
        budget_vs_actuals['Variance'] = budget_vs_actuals['Budget'] - \
            budget_vs_actuals['Actuals']
        self.budget_vs_actuals = budget_vs_actuals

    def savings_rate(self):
        monthly_comparison_savings_rate = self.df.groupby('Month').agg(
            {'Deposit Amount': 'sum', 'Withdrawal Amount': 'sum'})
        monthly_comparison_savings_rate['Savings Rate'] = (
            (monthly_comparison_savings_rate['Deposit Amount'] - monthly_comparison_savings_rate['Withdrawal Amount']) / monthly_comparison_savings_rate['Deposit Amount']) * 100
        self.monthly_comparison_savings_rate = monthly_comparison_savings_rate

    def expense_predictions(self):
        self.monthly_expense = self.df.groupby(
            'Month')['Withdrawal Amount'].sum()
        # Simple forecast: next month equals average of last three months
        self.next_month_prediction = self.monthly_expense[-3:].mean()
        return html.H1(f"Predicted expense for next month: {round(self.next_month_prediction, 2)}")
        # You can add more complex predictive models here

    def mom_growth(self, period='Month', value='Withdrawal'):
        self.mom_growth = self.df.groupby(period)[f'{value} Amount'].sum()
        self.mom_growth = self.mom_growth.pct_change() * 100

    def financial_health_score(self):
        budget_vs_actuals_df = self.budget_vs_actuals
        savings_rate_df = self.monthly_comparison_savings_rate['Savings Rate']
        self.score = (savings_rate_df.mean() -
                      budget_vs_actuals_df['Variance'].mean()) / 2
        return html.H1(f"Your Finacial health score stands at: {round(self.score, 2)}")

    def top_5_expenses(self):
        self.top_expenses = self.df.nlargest(5, 'Withdrawal Amount')
        top_expenses_table = dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i} for i in self.top_expenses.columns],
            data=self.top_expenses.to_dict('records'),
        )

        # Extract narrations and amounts for the top 5 expenses
        top_5_narrations = self.top_expenses['Narration'].values
        top_5_amounts = self.top_expenses['Withdrawal Amount'].values

        narration_list = [html.P("Descriptions for Top 5 Expenses")]

        # Combine narrations with amounts
        for narration, amount in zip(top_5_narrations, top_5_amounts):
            narration_list.append(html.P(f"{narration} - {amount}"))

        return html.Div([
            html.H1("Top 5 Expenses"),
            html.Div(narration_list)
        ])

    def withdrawal_trend_analysis(self):
        self.df['YearMonth'] = self.df['Date'].dt.to_period('M')
        self.monthly_withdrawals = self.df.groupby(
            'YearMonth')['Withdrawal Amount'].sum()

    def deposit_trend_analysis(self):
        self.df['YearMonth'] = self.df['Date'].dt.to_period('M')
        self.monthly_deposits = self.df.groupby(
            'YearMonth')['Deposit Amount'].sum()

    def daywise_spending(self):
        self.daywise_expense = self.df.groupby(
            'Day')['Withdrawal Amount'].sum()

    def weekwise_spending(self):
        self.df['Week'] = self.df['Date'].dt.isocalendar().week
        self.weekwise_expense = self.df.groupby(
            'Week')['Withdrawal Amount'].sum()

    def category_vs_budget(self):
        transact_column = "Withdrawal"
        category_expense = self.df.explode('expense_categories').groupby(
            'expense_categories')[f'{transact_column} Amount'].sum()
        budget_series = pd.Series(self.budget_dict)
        self.comparison_df = pd.concat(
            [category_expense, budget_series], axis=1, keys=['Actual', 'Budget'])

    def rolling_average_spending(self, window=3):
        self.df.sort_values('Date', inplace=True)
        self.rolling_average = self.df['Withdrawal Amount'].rolling(
            window=window).mean()

    def run(self):
        # Create empty dictionaries
        qual_plots = {}
        quant_plots = {}
        print("Generating plots for qualitative insights...")
        # qual_plots['Insights'] = self.eda()
        qual_plots['Monthly Spending Overview'] = self.generate_plot(
            'Monthly Spending Overview', 'Month', 'Amount Spent', 'line')
        qual_plots['Seasonal Trends in Withdrawal Amount'] = self.generate_plot(
            'Seasonal Trends in Withdrawal Amount', 'Month', 'Average Withdrawal Amount', 'line')
        qual_plots['Top 5 Expenses of the Month'] = self.generate_plot(
            'Top 5 Expenses of the Month', 'Narration', 'Amount', 'bar')
        qual_plots['Daywise Spending'] = self.generate_plot(
            'Daywise Spending', 'Day', 'Amount Spent', 'line')
        qual_plots['Weekwise Spending'] = self.generate_plot(
            'Weekwise Spending', 'Week', 'Amount Spent', 'line')
        qual_plots['MoM Growth in Expenses'] = self.generate_plot(
            'MoM Growth in Expenses', 'Month', 'Growth (%)', 'line')

        # Additional qualitative plots if any can be added in the same manner

        print("Generating plots for quantitative insights...")
        quant_plots['Monthly Income vs Expenditure'] = self.generate_plot(
            'Monthly Income vs Expenditure', 'Month', 'Amount', 'bar')
        quant_plots['Monthly Average Balance'] = self.generate_plot(
            'Monthly Average Balance', 'Month', 'Amount', 'bar')
        quant_plots['Cash Flow Analysis'] = self.generate_plot(
            'Cash Flow Analysis', 'Month', 'Net Cash Flow', 'bar')
        quant_plots['Budget vs Actuals'] = self.generate_plot(
            'Budget vs Actuals', 'Category', 'Amount', 'bar')
        quant_plots['Monthly Savings Rate'] = self.generate_plot(
            'Monthly Savings Rate', 'Month', 'Savings Rate (%)', 'line')
        quant_plots['Monthly Withdrawal Trend Analysis'] = self.generate_plot(
            'Monthly Withdrawal Trend Analysis', 'Month', 'Withdrawal Amount', 'line')
        quant_plots['Monthly Deposit Trend Analysis'] = self.generate_plot(
            'Monthly Deposit Trend Analysis', 'Month', 'Deposit Amount', 'line')
        quant_plots['Expense categories vs Budget'] = self.generate_plot(
            'Expense categories vs Budget', 'Category', 'Amount', 'bar')
        quant_plots[f'Rolling Average Spending (Window: {3}) Days'] = self.generate_plot(
            f'Rolling Average Spending (Window: {3}) Days', 'Date', 'Amount', 'line')
        quant_plots['Income Categories'] = self.generate_plot(
            'Income Categories', 'Category', 'Income Amount', 'pie')
        quant_plots['Expense Categories'] = self.generate_plot(
            'Expense Categories', 'Category', 'Expense Amount', 'pie')

        # Additional quantitative plots if any can be added in the same manner

        quant_plots['UPI Expense Drilldown'] = self.group_p2p_upi_withdrawals()
        quant_plots['UPI Deposit Drilldown'] = self.group_p2p_upi_deposits()
        # Assign dictionaries to the respective class attributes
        self.qual_plots = qual_plots
        self.quant_plots = quant_plots

    def group_p2p_upi_withdrawals(self):
        # Group by 'Person Associated' and sum the 'Amount'
        total_amounts = self.UPIsent_to_individuals.groupby(
            'Person Associated')['Withdrawal Amount'].sum()
        # Filter based on value_counts > 1
        filtered_total_amounts = total_amounts[self.UPIsent_to_individuals['Person Associated'].value_counts(
        ) > 1]
        # Tag rest transactions as "misc"
        # This will print the filtered DataFrame
        # print(filtered_total_amounts)
        # Assuming filtered_total_amounts is your Series object
        filtered_total_amounts_df = pd.DataFrame({
            'Person': filtered_total_amounts.index,
            'Total Amount': filtered_total_amounts.values
        })
        # Sort the DataFrame by 'Total Amount' in descending order
        sorted_df = filtered_total_amounts_df.sort_values(
            by='Total Amount', ascending=False)

        # Create a bubble chart
        fig = px.scatter(
            sorted_df,
            x='Person',
            y='Total Amount',
            size='Total Amount',  # Size of bubbles
            title='Total UPI Transactions sent to Individuals',
            labels={'Person': 'Person Associated',
                    'Total Amount': 'Total Amount Sent'},
            size_max=60  # You can adjust this to control the maximum bubble size
        )

        return dcc.Graph(figure=fig)

    def group_p2p_upi_deposits(self):
        # Group by 'Person Associated' and sum the 'Amount'
        total_amounts = self.withUPI_incomes.groupby('Person Associated')[
            'Deposit Amount'].sum()

        # Filter based on value_counts > 1
        filtered_total_amounts = total_amounts[self.withUPI_incomes['Person Associated'].value_counts(
        ) > 1]

        # Tag rest transactions as "misc"

        # This will print the filtered DataFrame
        # print(filtered_total_amounts)
        # Assuming filtered_total_amounts is your Series object
        filtered_total_amounts_df = pd.DataFrame({
            'Person': filtered_total_amounts.index,
            'Total Amount': filtered_total_amounts.values
        })

        # Sort the DataFrame by 'Total Amount' in descending order
        sorted_df = filtered_total_amounts_df.sort_values(
            by='Total Amount', ascending=False)

        # Create a bubble chart
        fig = px.scatter(
            sorted_df,
            x='Person',
            y='Total Amount',
            size='Total Amount',  # Size of bubbles
            title='Total UPI Transactions recieved from Individuals',
            labels={'Person': 'Person Associated',
                    'Total Amount': 'Total Amount Received'},
            size_max=60  # You can adjust this to control the maximum bubble size
        )

        return dcc.Graph(figure=fig)

    def upi_analysis(self):
        statement = self.df
        # Now qual_outputs and quant_outputs will have the results keyed by method names

        withUPI_transactions = statement.query(
            'expense_categories == "withUPI" | income_sources == "Direct Transfers"')

        # Use of UPI
        self.upi_usage = len(withUPI_transactions) / len(statement) * 100

        withUPI_transactions['Cleaned_Narration'] = withUPI_transactions['Narration'].str.lower(
        )
        withUPI_transactions['Cleaned_Narration'] = withUPI_transactions['Cleaned_Narration'].apply(
            lambda x: re.sub('[^a-z\s]', '\t', x))  # Remove numbers and special characters

        # I still see some UPI transactions tagged to online business
        # Let's tag them too under categories again

        # Implement your own categorization logic here
        expense_categories = {
            'Food & Groceries': ['swiggy', 'zomato', 'simpl', 'food', 'grocery', 'blinkit', 'instamart', 'general', 'store', 'pizza', 'DOMINOS', 'macdonalds', 'MART',
                                 'CAFE', 'DHABA', 'BAKERS', 'POHA', '24SEVEN', 'RED CHILLI DRAGON', 'YOTIBET', 'YAARAN DA ADDA', 'LEA IZAKAYA'],
            'EMI': ['emi', 'loan', 'MPOKKET'],
            'Shopping': ['myntra', 'amazon', 'flipkart', 'EKART', 'DECATHLON'],
            'Subscriptions': ['rentickle', 'spotify', 'music', 'youtube', 'netflix', 'prime', 'hotstar', 'apple', 'google', 'notion', 'RELIANCE BP MOBILITY', 'RENTOMOJO'],
            'Cloud Servers': ['hetzner'],
            'Fuel': ['Petrol', 'pump', 'filling', 'Indian Oil', 'Oil', 'fill', 'FUEL'],
            'Bill Payments': ['BILLDESK', 'EURONETGPAY'],
            'Travel': ['IRCTC', 'RAPIDO', 'BUS', 'OLAMoney', 'ibibo', 'booking', 'agoda', 'hosteller'],
            'Health': ['CLINIC', 'Dental', 'Hospital', 'MEDICAL', 'Medicin', 'chemist', 'opticals'],
            'Utilities': ['ELECTRICALS', 'SOFTWARE'],
            'Miscelleneous': ['BHARATPEMERCHANT', 'ENTERPRISES', 'ADD MONEY TO WALLET', 'SKOOTR', 'SBIMOPS'],
            'Insurance': ['CHOLAMANDALAM']




        }
        income_sources = {'Recurring Deposits': ['salary', 'TEG INDIA PRIVATE', 'GD RESEARCH'],
                          'Direct Transfers': ['upi'],
                          'Interest': ['interest'],
                          'Refunds': ['swiggy', 'zomato', 'simpl', 'food', 'grocery', 'blinkit', 'instamart', 'emi', 'rentickle', 'spotify', 'music', 'youtube', 'netflix', 'prime', 'hotstar', 'apple', 'google', 'notion', 'flipkart'],
                          'Account Verify': ['RVSL DT']
                          }
        self.categories_dict = {
            'expense_categories': expense_categories, 'income_sources': income_sources}

        withUPI_expenses = withUPI_transactions.loc[withUPI_transactions['Withdrawal Amount'] > 0]
        for category, keywords in expense_categories.items():
            for keyword in keywords:
                mask = withUPI_expenses['Cleaned_Narration'].str.contains(
                    keyword, case=False, na=False)
                withUPI_expenses.loc[mask, 'Categories'] = withUPI_expenses.loc[mask, 'Categories'].apply(
                    lambda x: category if category not in x else x)
        # print(withUPI_expenses.shape)

        withUPI_incomes = withUPI_transactions.loc[withUPI_transactions['Deposit Amount'] > 0]
        for category, keywords in income_sources.items():
            for keyword in keywords:
                mask = withUPI_incomes['Cleaned_Narration'].str.contains(
                    keyword, case=False, na=False)
                withUPI_incomes.loc[mask, 'Categories'] = withUPI_incomes.loc[mask, 'Categories'].apply(
                    lambda x: category if category not in x else x)
        # print(withUPI_incomes.shape)
        withUPI_incomes['Person Associated'] = withUPI_incomes['Narration'].map(
            lambda x: x.split("-")[1])
        self.withUPI_incomes = withUPI_incomes

        UPIsent_to_individuals = withUPI_expenses[withUPI_expenses['Categories'].apply(
            lambda x: isinstance(x, list))]
        UPIsent_to_individuals['Person Associated'] = UPIsent_to_individuals['Narration'].map(
            lambda x: x.split("-")[1])

        self.social_circle = UPIsent_to_individuals['Person Associated'].value_counts(
        )
        self.filtered_social_circle = self.social_circle[self.social_circle > 1]
        # print("Names with more than one transaction:", len(filtered_social_circle))

        misc_trans = self.social_circle[self.social_circle == 1]
        self.misc_individuals = misc_trans.index.tolist()
        # Modify 'Person Associated' in the original DataFrame

        # Modify 'expense_categories' in the original DataFrame where 'Person Associated' matches misc_individuals
        UPIsent_to_individuals.loc[UPIsent_to_individuals['Person Associated'].isin(
            self.misc_individuals), 'Categories'] = 'Miscelleneous'
        UPIsent_to_individuals.loc[~UPIsent_to_individuals['Person Associated'].isin(
            self.misc_individuals), 'Categories'] = 'Direct Transfers'
        self.UPIsent_to_individuals = UPIsent_to_individuals

        withUPI_expenses['Person Associated'] = ''
        withUPI_expenses.update(UPIsent_to_individuals)
        self.withUPI_expenses = withUPI_expenses
        withUPI_transactions['Person Associated'] = ''
        withUPI_transactions.update(withUPI_expenses)
        withUPI_transactions.update(withUPI_incomes)
        self.withUPI_transactions = withUPI_transactions
        statement['Person Associated'] = ''
        statement.update(withUPI_transactions)
        # Custom function to update 'Categories' based on 'expense_categories' and 'income_sources'

        def update_categories(row):
            # Check if 'Categories' is a list
            if isinstance(row['Categories'], list):
                # If it's an empty list
                if not row['Categories']:
                    # If 'expense_categories' has a value (not empty)
                    if row['expense_categories']:
                        return row['expense_categories']
                    # If 'income_sources' has a value (not empty)
                    elif row['income_sources']:
                        return row['income_sources']
                    # If both are empty
                    else:
                        return row['Categories']
                # If 'Categories' is already a non-empty list
                else:
                    return row['Categories']
            # If 'Categories' is not a list
            else:
                return row['Categories']

        # Apply the custom function
        statement['Categories'] = statement.apply(update_categories, axis=1)
        self.df = statement
        self.process_categorisation()
