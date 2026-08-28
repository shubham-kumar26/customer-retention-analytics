import requests
import pandas as pd

response = requests.get('https://dummyjson.com/products/categories')
categories = response.json()
print(categories)


import requests
import pandas as pd

categories_map = {
    'sports_leisure': 'sports-accessories',
    'furniture_decor': 'furniture',
    'bed_bath_table': 'home-decoration'
}

results = []
for olist_category, api_category in categories_map.items():
    response = requests.get(f'https://dummyjson.com/products/category/{api_category}')
    data = response.json()
    prices = [p['price'] for p in data['products']]
    avg_price = sum(prices) / len(prices)
    results.append({
        'olist_category': olist_category,
        'api_category': api_category,
        'avg_market_price_usd': round(avg_price, 2),
        'num_products': len(prices)
    })

df_prices = pd.DataFrame(results)
print(df_prices)
df_prices.to_csv('competitor_prices.csv', index=False)
