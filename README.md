
# 🪙 Cryptocurrency Exchange Admin Dashboard

A **Streamlit-based admin panel** for managing and analyzing data from a cryptocurrency exchange platform. This dashboard provides a user-friendly interface to explore and administer users, wallets, transactions, KYC, orders, reports, and more using a SQL Server backend.

## 🚀 Features

- 📋 View and manage **Users**, **Digital Wallets**, **Cryptocurrencies**, **Orders**, **Transactions**, and **Support Tickets**
- 📁 Administer **KYC Documents**, view **User Holdings**, and track **Reports**
- 📆 Fetch **Monthly Transactions** using stored procedures
- 🧾 Execute orders and resolve tickets through **stored procedures**
- 📊 Visual dashboards with:
  - Daily transaction trends
  - Wallet balance distributions
  - Buy/Sell volumes per cryptocurrency
  - Top 5 crypto holdings in USD
- 💼 View individual **User Portfolio Value** via scalar functions
- 🛡️ Check **User KYC Status** via scalar functions
- 📝 View **User Change Logs** and **Transaction Logs**

## 🧠 Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend**: [SQL Server](https://www.microsoft.com/en-us/sql-server)
- **Visuals**: [Plotly](https://plotly.com/python/)
- **Data Handling**: [Pandas](https://pandas.pydata.org/)
- **Connection**: `pyodbc` for ODBC connectivity to SQL Server

## 🗂️ Project Structure

```
.
├── crypto_app.py              # Main Streamlit application
├── requirements.txt           # Python dependencies
├── crypto_exchange_ddl.sql    # Database schema (DDL)
├── Insert_script.sql          # Sample data insertion
├── PSM_Script.sql             # Stored procedures, functions, triggers
```

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/crypto-admin-dashboard.git
cd crypto-admin-dashboard
```

### 2. Set Up the Database

Run the following SQL scripts in order on your SQL Server instance:

1. `crypto_exchange_ddl.sql` — creates all necessary tables and views.
2. `Insert_script.sql` — inserts initial data.
3. `PSM_Script.sql` — defines procedures, scalar functions, triggers.

### 3. Install Dependencies

Ensure Python ≥ 3.8 is installed.

```bash
pip install -r requirements.txt
```

### 4. Configure Database Credentials

Edit the connection details inside `crypto_app.py`:

```python
'DRIVER={ODBC Driver 17 for SQL Server};'
'SERVER=localhost,1433;'
'DATABASE=CryptoExchangeDB;'
'UID=SA;'
'PWD=password'
```

> 💡 You can also move these credentials to a `.env` file for better security.

### 5. Run the App

```bash
streamlit run crypto_app.py
```

## 📊 Dashboard Preview

The sidebar allows you to navigate across modules like:

- **Transactions**
- **Orders**
- **KYC Status**
- **Support Ticket Metrics**
- **Portfolio Value**
- **Analytics Dashboard** with multiple visualizations

## 📌 TODOs & Enhancements

- ✅ Add user authentication layer for admin
- 🔐 Add field-level encryption for sensitive data (e.g., wallet info)
- 🧪 Add unit tests for DB calls and error handling
- 📤 Dockerize the Streamlit app for deployment

## 📃 License

This project is for academic/demo purposes. Please replace the password and deployment credentials before use in production.
