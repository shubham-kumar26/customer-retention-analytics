import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns

engine = create_engine('postgresql://postgres:1234@localhost:5432/olist_churn')

# Query 1: Customer value data
query1 = """
SELECT 
    c.customer_unique_id,
    COUNT(DISTINCT o.order_id) AS total_orders,
    SUM(op.payment_value) AS total_spent
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_payments op ON o.order_id = op.order_id
GROUP BY c.customer_unique_id
"""
df_customers = pd.read_sql(query1, engine)

print(df_customers.shape)
print(df_customers.head())

# Chart 1: Distribution of number of orders per customer
plt.figure(figsize=(10,5))
order_counts = df_customers['total_orders'].value_counts().sort_index()
order_counts.plot(kind='bar', color='steelblue')
plt.title('Number of Customers by Order Count')
plt.xlabel('Total Orders')
plt.ylabel('Number of Customers')
plt.yscale('log')  # log scale since one-time buyers dominate so heavily
plt.tight_layout()
plt.savefig('orders_distribution.png')

print("Chart saved as orders_distribution.png")


# Chart 2: Total spend distribution (capped for readability)
plt.figure(figsize=(10,5))
sns.histplot(df_customers[df_customers['total_spent'] < 1000]['total_spent'], bins=50, color='indianred')
plt.title('Customer Total Spend Distribution (capped at R$1000)')
plt.xlabel('Total Spent (R$)')
plt.ylabel('Number of Customers')
plt.tight_layout()
plt.savefig('spend_distribution.png')

print("Chart saved as spend_distribution.png")
