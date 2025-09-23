# Python Data Science Interview Questions & Solutions

## Table of Contents
1. [Basic Pandas Operations](#basic-pandas-operations)
2. [Data Cleaning & Preprocessing](#data-cleaning--preprocessing)
3. [Advanced Pandas](#advanced-pandas)
4. [NumPy Operations](#numpy-operations)
5. [Data Analysis & Statistics](#data-analysis--statistics)
6. [Performance Optimization](#performance-optimization)

---

## Basic Pandas Operations

### Q1: DataFrame Creation and Basic Operations
**Problem:** Create a DataFrame from a dictionary and perform basic operations.

**Category:** DataFrame fundamentals
**Difficulty:** Easy

```python
import pandas as pd
import numpy as np

# Create DataFrame
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'age': [25, 30, 35, 28],
    'salary': [50000, 60000, 70000, 55000],
    'department': ['IT', 'HR', 'IT', 'Finance']
}

df = pd.DataFrame(data)

# Basic operations
print("Shape:", df.shape)
print("Info:")
print(df.info())
print("First 2 rows:")
print(df.head(2))
print("Unique departments:", df['department'].unique())
```

**Time Complexity:** O(1) for shape, O(n) for info() and head()
**Space Complexity:** O(n) where n is number of rows

---

### Q2: Data Selection and Filtering
**Problem:** Select specific columns and filter rows based on conditions.

**Category:** Data selection/filtering
**Difficulty:** Easy-Medium

```python
# Solution
def filter_dataframe(df):
    # Select specific columns
    subset = df[['name', 'age', 'salary']]
    
    # Filter rows where age > 28 and salary > 55000
    filtered = df[(df['age'] > 28) & (df['salary'] > 55000)]
    
    # Using query method (alternative)
    filtered_query = df.query('age > 28 and salary > 55000')
    
    return subset, filtered, filtered_query

subset, filtered, filtered_query = filter_dataframe(df)
print("Filtered result:")
print(filtered)
```

**Time Complexity:** O(n) for filtering operations
**Space Complexity:** O(k) where k is number of rows meeting criteria

---

## Data Cleaning & Preprocessing

### Q3: Handle Missing Data
**Problem:** Clean a dataset with various types of missing values.

**Category:** Data preprocessing
**Difficulty:** Medium

```python
# Create sample data with missing values
data_missing = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, 4, np.nan],
    'C': ['x', 'y', None, 'z', 'w'],
    'D': [1.1, np.nan, 3.3, 4.4, 5.5]
}

df_missing = pd.DataFrame(data_missing)

def clean_missing_data(df):
    # Strategy 1: Drop rows with any missing values
    cleaned_drop = df.dropna()
    
    # Strategy 2: Fill missing values
    # Forward fill for column A
    df['A'] = df['A'].fillna(method='ffill')
    
    # Mean for numeric column B
    df['B'] = df['B'].fillna(df['B'].mean())
    
    # Mode for categorical column C
    df['C'] = df['C'].fillna(df['C'].mode()[0])
    
    # Median for column D
    df['D'] = df['D'].fillna(df['D'].median())
    
    return df

cleaned_df = clean_missing_data(df_missing.copy())
print("Cleaned DataFrame:")
print(cleaned_df)
```

**Time Complexity:** O(n*m) where n=rows, m=columns
**Space Complexity:** O(n*m)

---

### Q4: Data Type Conversion and Validation
**Problem:** Convert data types and validate data integrity.

**Category:** Data validation/conversion
**Difficulty:** Medium

```python
def convert_and_validate(df):
    # Create sample data with wrong types
    data = {
        'id': ['1', '2', '3', '4'],
        'price': ['10.5', '20.3', 'invalid', '15.7'],
        'date': ['2023-01-01', '2023-01-02', '2023-13-01', '2023-01-04'],
        'category': ['A', 'B', 'A', 'C']
    }
    
    df = pd.DataFrame(data)
    
    # Convert id to numeric
    df['id'] = pd.to_numeric(df['id'])
    
    # Convert price to numeric, handling errors
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    # Convert date, handling invalid dates
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Convert category to categorical
    df['category'] = df['category'].astype('category')
    
    # Validate and report issues
    issues = {
        'invalid_prices': df['price'].isna().sum(),
        'invalid_dates': df['date'].isna().sum()
    }
    
    return df, issues

converted_df, issues = convert_and_validate(df)
print("Converted DataFrame:")
print(converted_df)
print("Data quality issues:", issues)
```

**Time Complexity:** O(n) for each conversion
**Space Complexity:** O(n)

---

## Advanced Pandas

### Q5: GroupBy Operations and Aggregations
**Problem:** Perform complex groupby operations with multiple aggregation functions.

**Category:** Data aggregation
**Difficulty:** Medium-Hard

```python
# Create sample sales data
np.random.seed(42)
sales_data = {
    'region': np.random.choice(['North', 'South', 'East', 'West'], 1000),
    'product': np.random.choice(['A', 'B', 'C'], 1000),
    'sales': np.random.normal(1000, 200, 1000),
    'profit': np.random.normal(100, 50, 1000),
    'date': pd.date_range('2023-01-01', periods=1000, freq='D')
}

sales_df = pd.DataFrame(sales_data)

def advanced_groupby_analysis(df):
    # Multiple aggregations
    agg_result = df.groupby(['region', 'product']).agg({
        'sales': ['mean', 'std', 'sum', 'count'],
        'profit': ['mean', 'sum'],
    }).round(2)
    
    # Flatten column names
    agg_result.columns = ['_'.join(col).strip() for col in agg_result.columns]
    
    # Custom aggregation function
    def profit_margin(x):
        return (x['profit'].sum() / x['sales'].sum()) * 100
    
    margin_by_region = df.groupby('region').apply(profit_margin).round(2)
    
    # Transform operations
    df['sales_rank_by_region'] = df.groupby('region')['sales'].rank(ascending=False)
    df['sales_pct_of_region'] = df.groupby('region')['sales'].transform(
        lambda x: (x / x.sum()) * 100
    )
    
    return agg_result, margin_by_region, df

agg_result, margin_by_region, enhanced_df = advanced_groupby_analysis(sales_df)
print("Aggregation results:")
print(agg_result.head())
print("\nProfit margins by region:")
print(margin_by_region)
```

**Time Complexity:** O(n log n) for groupby operations
**Space Complexity:** O(k) where k is number of unique groups

---

### Q6: Pivot Tables and Reshaping Data
**Problem:** Create pivot tables and reshape data between wide and long formats.

**Category:** Data reshaping
**Difficulty:** Medium

```python
def pivot_and_reshape(df):
    # Create pivot table
    pivot = df.pivot_table(
        index='region',
        columns='product',
        values='sales',
        aggfunc='mean',
        fill_value=0
    ).round(2)
    
    # Melt (wide to long)
    melted = df[['region', 'product', 'sales', 'profit']].melt(
        id_vars=['region', 'product'],
        value_vars=['sales', 'profit'],
        var_name='metric',
        value_name='value'
    )
    
    # Cross-tabulation
    crosstab = pd.crosstab(
        df['region'],
        df['product'],
        values=df['sales'],
        aggfunc='count',
        fill_value=0
    )
    
    return pivot, melted, crosstab

pivot, melted, crosstab = pivot_and_reshape(sales_df)
print("Pivot table:")
print(pivot)
print("\nCross-tabulation:")
print(crosstab)
```

**Time Complexity:** O(n) for pivot operations
**Space Complexity:** O(n)

---

## NumPy Operations

### Q7: Array Operations and Broadcasting
**Problem:** Perform efficient array operations using NumPy broadcasting.

**Category:** Array manipulation
**Difficulty:** Medium

```python
def numpy_operations():
    # Create arrays
    arr1 = np.random.randint(1, 10, (5, 4))
    arr2 = np.random.randint(1, 5, (4,))
    
    # Broadcasting operations
    result1 = arr1 + arr2  # Broadcasting (5,4) + (4,)
    result2 = arr1 * arr2  # Element-wise multiplication
    
    # Matrix operations
    matrix_a = np.random.randn(3, 4)
    matrix_b = np.random.randn(4, 2)
    matrix_mult = np.dot(matrix_a, matrix_b)
    
    # Statistical operations
    stats = {
        'mean_by_column': np.mean(arr1, axis=0),
        'std_by_row': np.std(arr1, axis=1),
        'percentile_95': np.percentile(arr1, 95),
        'correlation': np.corrcoef(arr1.T)  # Column correlations
    }
    
    # Boolean indexing
    mask = arr1 > 5
    filtered_values = arr1[mask]
    
    return {
        'original': arr1,
        'broadcast_add': result1,
        'matrix_mult': matrix_mult,
        'stats': stats,
        'filtered': filtered_values
    }

numpy_results = numpy_operations()
print("Original array shape:", numpy_results['original'].shape)
print("Matrix multiplication result shape:", numpy_results['matrix_mult'].shape)
```

**Time Complexity:** O(n) for element-wise operations, O(n³) for matrix multiplication
**Space Complexity:** O(n) for result arrays

---

### Q8: Vectorized Operations vs Loops
**Problem:** Compare performance of vectorized operations vs Python loops.

**Category:** Performance optimization
**Difficulty:** Medium

```python
import time

def performance_comparison():
    # Large array for testing
    size = 1000000
    arr = np.random.randn(size)
    
    # Method 1: Python loop (slow)
    start_time = time.time()
    result_loop = []
    for x in arr:
        result_loop.append(x**2 if x > 0 else 0)
    result_loop = np.array(result_loop)
    loop_time = time.time() - start_time
    
    # Method 2: Vectorized with np.where (fast)
    start_time = time.time()
    result_vectorized = np.where(arr > 0, arr**2, 0)
    vectorized_time = time.time() - start_time
    
    # Method 3: Boolean indexing (also fast)
    start_time = time.time()
    result_bool = np.zeros_like(arr)
    mask = arr > 0
    result_bool[mask] = arr[mask]**2
    bool_time = time.time() - start_time
    
    return {
        'loop_time': loop_time,
        'vectorized_time': vectorized_time,
        'boolean_time': bool_time,
        'speedup_vectorized': loop_time / vectorized_time,
        'speedup_boolean': loop_time / bool_time
    }

perf_results = performance_comparison()
print("Performance comparison:")
for key, value in perf_results.items():
    print(f"{key}: {value:.4f}")
```

**Time Complexity:** O(n) for vectorized, O(n) for loops but with higher constant factor
**Space Complexity:** O(n)

---

## Data Analysis & Statistics

### Q9: Time Series Analysis
**Problem:** Analyze time series data with resampling and rolling calculations.

**Category:** Time series analysis
**Difficulty:** Hard

```python
def time_series_analysis():
    # Create time series data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    ts_data = pd.DataFrame({
        'date': dates,
        'value': np.cumsum(np.random.randn(len(dates))) + 100,
        'volume': np.random.poisson(1000, len(dates))
    })
    ts_data.set_index('date', inplace=True)
    
    # Rolling calculations
    ts_data['ma_7'] = ts_data['value'].rolling(window=7).mean()
    ts_data['ma_30'] = ts_data['value'].rolling(window=30).mean()
    ts_data['volatility'] = ts_data['value'].rolling(window=30).std()
    
    # Resampling
    monthly = ts_data.resample('M').agg({
        'value': 'last',
        'volume': 'sum',
        'ma_7': 'last'
    })
    
    # Seasonal decomposition simulation
    ts_data['trend'] = ts_data['value'].rolling(window=365, center=True).mean()
    ts_data['detrended'] = ts_data['value'] - ts_data['trend']
    
    # Calculate returns
    ts_data['returns'] = ts_data['value'].pct_change()
    ts_data['log_returns'] = np.log(ts_data['value'] / ts_data['value'].shift(1))
    
    return ts_data, monthly

ts_data, monthly = time_series_analysis()
print("Time series statistics:")
print(ts_data[['value', 'ma_7', 'ma_30', 'returns']].describe())
```

**Time Complexity:** O(n*w) where w is window size for rolling operations
**Space Complexity:** O(n)

---

### Q10: Statistical Analysis and Hypothesis Testing
**Problem:** Perform statistical tests and correlation analysis.

**Category:** Statistical analysis
**Difficulty:** Hard

```python
from scipy import stats

def statistical_analysis():
    # Generate sample data
    np.random.seed(42)
    group_a = np.random.normal(100, 15, 1000)
    group_b = np.random.normal(105, 15, 1000)
    
    # Descriptive statistics
    desc_stats = {
        'group_a': {
            'mean': np.mean(group_a),
            'std': np.std(group_a),
            'median': np.median(group_a),
            'skew': stats.skew(group_a),
            'kurtosis': stats.kurtosis(group_a)
        },
        'group_b': {
            'mean': np.mean(group_b),
            'std': np.std(group_b),
            'median': np.median(group_b),
            'skew': stats.skew(group_b),
            'kurtosis': stats.kurtosis(group_b)
        }
    }
    
    # Hypothesis tests
    # T-test
    t_stat, t_pvalue = stats.ttest_ind(group_a, group_b)
    
    # Mann-Whitney U test (non-parametric)
    u_stat, u_pvalue = stats.mannwhitneyu(group_a, group_b)
    
    # Normality test
    shapiro_a = stats.shapiro(group_a[:1000])  # Limited sample for Shapiro
    shapiro_b = stats.shapiro(group_b[:1000])
    
    # Correlation analysis
    correlation_matrix = np.corrcoef([group_a, group_b])
    
    results = {
        'descriptive_stats': desc_stats,
        'tests': {
            't_test': {'statistic': t_stat, 'p_value': t_pvalue},
            'mann_whitney': {'statistic': u_stat, 'p_value': u_pvalue},
            'normality_a': {'statistic': shapiro_a[0], 'p_value': shapiro_a[1]},
            'normality_b': {'statistic': shapiro_b[0], 'p_value': shapiro_b[1]}
        },
        'correlation': correlation_matrix
    }
    
    return results

stats_results = statistical_analysis()
print("Statistical test results:")
print(f"T-test p-value: {stats_results['tests']['t_test']['p_value']:.4f}")
print(f"Mann-Whitney p-value: {stats_results['tests']['mann_whitney']['p_value']:.4f}")
```

**Time Complexity:** O(n log n) for most statistical tests
**Space Complexity:** O(n)

---

## Performance Optimization

### Q11: Memory-Efficient Data Processing
**Problem:** Process large datasets efficiently with chunking and memory optimization.

**Category:** Memory optimization
**Difficulty:** Hard

```python
def memory_efficient_processing(file_size_mb=100):
    # Simulate large dataset processing
    def process_chunk(chunk):
        # Example processing: calculate summary statistics
        return {
            'mean': chunk['value'].mean(),
            'std': chunk['value'].std(),
            'count': len(chunk),
            'outliers': len(chunk[chunk['value'] > chunk['value'].quantile(0.95)])
        }
    
    # Method 1: Chunked processing
    def chunked_processing():
        chunk_size = 10000
        results = []
        
        # Simulate reading in chunks
        for i in range(0, file_size_mb * 1000, chunk_size):
            # Create chunk data
            chunk_data = pd.DataFrame({
                'id': range(i, min(i + chunk_size, file_size_mb * 1000)),
                'value': np.random.randn(min(chunk_size, file_size_mb * 1000 - i))
            })
            
            # Process chunk
            chunk_result = process_chunk(chunk_data)
            results.append(chunk_result)
            
            # Memory management
            del chunk_data
        
        return results
    
    # Method 2: Using categorical data to save memory
    def optimize_dtypes():
        df = pd.DataFrame({
            'category': np.random.choice(['A', 'B', 'C'], 100000),
            'value': np.random.randn(100000).astype('float32'),  # Use float32 instead of float64
            'id': np.arange(100000, dtype='int32')  # Use int32 instead of int64
        })
        
        # Convert to categorical
        df['category'] = df['category'].astype('category')
        
        memory_usage = df.memory_usage(deep=True)
        return df, memory_usage
    
    chunk_results = chunked_processing()
    optimized_df, memory_info = optimize_dtypes()
    
    return chunk_results, optimized_df, memory_info

chunk_results, optimized_df, memory_info = memory_efficient_processing()
print("Memory usage by column:")
print(memory_info)
print("\nFirst chunk results:", chunk_results[0])
```

**Time Complexity:** O(n) but processed in chunks to manage memory
**Space Complexity:** O(chunk_size) instead of O(n)

---

### Q12: Advanced Data Manipulation Challenge
**Problem:** Combine multiple techniques to solve a complex data problem.

**Category:** Comprehensive data manipulation
**Difficulty:** Very Hard

```python
def advanced_data_challenge():
    """
    Problem: You have sales data with inconsistent formats, missing values,
    and need to create a comprehensive analysis with multiple derived metrics.
    """
    
    # Create complex, messy dataset
    np.random.seed(42)
    n_records = 10000
    
    raw_data = {
        'transaction_id': [f'T{i:06d}' for i in range(n_records)],
        'customer_id': np.random.choice([f'C{i:04d}' for i in range(1000)], n_records),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], n_records),
        'sales_amount': np.random.lognormal(4, 1, n_records),
        'discount': np.random.uniform(0, 0.3, n_records),
        'sales_date': pd.date_range('2020-01-01', periods=n_records, freq='H'),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_records),
        'sales_rep': np.random.choice([f'Rep{i}' for i in range(50)], n_records)
    }
    
    df = pd.DataFrame(raw_data)
    
    # Introduce missing values and inconsistencies
    df.loc[np.random.choice(df.index, 500), 'sales_amount'] = np.nan
    df.loc[np.random.choice(df.index, 200), 'customer_id'] = None
    
    # Solution pipeline
    def comprehensive_analysis(df):
        # 1. Data cleaning
        df = df.dropna(subset=['customer_id'])  # Remove rows without customer
        df['sales_amount'] = df['sales_amount'].fillna(df['sales_amount'].median())
        
        # 2. Feature engineering
        df['net_sales'] = df['sales_amount'] * (1 - df['discount'])
        df['sales_month'] = df['sales_date'].dt.to_period('M')
        df['sales_quarter'] = df['sales_date'].dt.to_period('Q')
        
        # 3. Customer segmentation using RFM analysis
        current_date = df['sales_date'].max()
        
        rfm = df.groupby('customer_id').agg({
            'sales_date': lambda x: (current_date - x.max()).days,  # Recency
            'transaction_id': 'count',  # Frequency
            'net_sales': 'sum'  # Monetary
        }).rename(columns={
            'sales_date': 'recency',
            'transaction_id': 'frequency',
            'net_sales': 'monetary'
        })
        
        # Create quantile-based scores
        rfm['r_score'] = pd.qcut(rfm['recency'], 5, labels=[5,4,3,2,1])
        rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
        rfm['m_score'] = pd.qcut(rfm['monetary'], 5, labels=[1,2,3,4,5])
        
        rfm['rfm_score'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)
        
        # 4. Sales performance analysis
        performance = df.groupby(['region', 'product_category', 'sales_month']).agg({
            'net_sales': ['sum', 'mean', 'count'],
            'discount': 'mean'
        }).round(2)
        
        # Flatten column names
        performance.columns = ['_'.join(col).strip() for col in performance.columns]
        
        # 5. Time series decomposition
        monthly_sales = df.groupby('sales_month')['net_sales'].sum()
        monthly_sales.index = monthly_sales.index.to_timestamp()
        
        # Calculate growth rates
        monthly_sales_df = monthly_sales.to_frame()
        monthly_sales_df['growth_rate'] = monthly_sales.pct_change()
        monthly_sales_df['rolling_avg'] = monthly_sales.rolling(window=3).mean()
        
        # 6. Top performers identification
        top_customers = rfm.nlargest(10, 'monetary')[['frequency', 'monetary', 'rfm_score']]
        top_products = df.groupby('product_category')['net_sales'].sum().sort_values(ascending=False)
        top_reps = df.groupby('sales_rep')['net_sales'].sum().nlargest(10)
        
        return {
            'cleaned_data_shape': df.shape,
            'rfm_analysis': rfm.head(),
            'performance_summary': performance.head(),
            'monthly_trends': monthly_sales_df,
            'top_customers': top_customers,
            'top_products': top_products,
            'top_sales_reps': top_reps
        }
    
    results = comprehensive_analysis(df)
    return results

challenge_results = advanced_data_challenge()
print("Challenge Results Summary:")
print(f"Cleaned data shape: {challenge_results['cleaned_data_shape']}")
print("\nTop product categories by sales:")
print(challenge_results['top_products'])
print("\nTop 5 customers by RFM analysis:")
print(challenge_results['top_customers'])
```

**Time Complexity:** O(n log n) due to sorting and ranking operations
**Space Complexity:** O(n + k) where k is number of unique groups

---

## Key Strategies and Tips

### Problem-Solving Approaches:

1. **Data Exploration First**
   - Always examine shape, dtypes, and missing values
   - Use `.info()`, `.describe()`, `.head()`, `.tail()`

2. **Vectorization Over Loops**
   - Use NumPy/Pandas vectorized operations
   - Apply functions with `.apply()` when necessary
   - Leverage broadcasting for array operations

3. **Memory Management**
   - Use appropriate dtypes (int32 vs int64, categorical)
   - Process data in chunks for large datasets
   - Use `.memory_usage(deep=True)` to monitor

4. **Performance Optimization**
   - Use `.query()` for complex filtering
   - Prefer `.loc[]` and `.iloc[]` for indexing
   - Use `.groupby()` efficiently with multiple aggregations

5. **Data Quality**
   - Handle missing values appropriately
   - Validate data types and ranges
   - Check for duplicates and outliers

### Common Complexity Patterns:

- **O(1)**: Direct indexing, shape operations
- **O(n)**: Linear scans, element-wise operations
- **O(n log n)**: Sorting, groupby operations
- **O(n²)**: Nested operations, cross-joins
- **O(n*m)**: Operations on n×m DataFrames

---

## Advanced Real-World Scenarios

### Q13: E-commerce Recommendation System Analysis
**Problem:** You're working for an e-commerce platform. Build a customer similarity system and analyze purchase patterns to identify cross-selling opportunities.

**Category:** Recommendation systems, customer analytics
**Difficulty:** Very Hard

```python
def ecommerce_recommendation_analysis():
    """
    Business Context: Online marketplace wants to improve cross-selling
    by identifying similar customers and product affinity patterns.
    """
    
    # Generate realistic e-commerce data
    np.random.seed(42)
    
    # Product catalog
    products = [f'PROD_{i:03d}' for i in range(1, 201)]  # 200 products
    categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports', 'Beauty']
    product_categories = {prod: np.random.choice(categories) for prod in products}
    product_prices = {prod: np.random.lognormal(3, 0.8) for prod in products}
    
    # Customer transactions (50k transactions, 5k customers)
    n_transactions = 50000
    n_customers = 5000
    
    transactions = []
    for i in range(n_transactions):
        customer_id = f'CUST_{np.random.randint(1, n_customers+1):05d}'
        # Simulate customer preferences (some customers like certain categories)
        preferred_category = np.random.choice(categories, p=[0.3, 0.2, 0.15, 0.15, 0.1, 0.1])
        
        # Bias product selection toward preferred category
        category_products = [p for p, c in product_categories.items() if c == preferred_category]
        if np.random.random() < 0.6 and category_products:  # 60% chance of preferred category
            product = np.random.choice(category_products)
        else:
            product = np.random.choice(products)
        
        transactions.append({
            'transaction_id': f'TXN_{i:06d}',
            'customer_id': customer_id,
            'product_id': product,
            'category': product_categories[product],
            'price': product_prices[product],
            'quantity': np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1]),
            'transaction_date': pd.Timestamp('2023-01-01') + pd.Timedelta(days=np.random.randint(0, 365)),
            'rating': np.random.choice([3, 4, 5], p=[0.2, 0.3, 0.5])  # Mostly positive ratings
        })
    
    df = pd.DataFrame(transactions)
    
    def build_recommendation_system(df):
        # 1. Create customer-product matrix
        customer_product = df.pivot_table(
            index='customer_id', 
            columns='product_id', 
            values='quantity', 
            fill_value=0
        )
        
        # 2. Calculate customer similarity using cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Only use customers with at least 5 purchases for similarity
        active_customers = df.groupby('customer_id').size()
        active_customers = active_customers[active_customers >= 5].index
        
        active_matrix = customer_product.loc[active_customers]
        similarity_matrix = cosine_similarity(active_matrix)
        similarity_df = pd.DataFrame(
            similarity_matrix, 
            index=active_customers, 
            columns=active_customers
        )
        
        # 3. Market basket analysis - find frequently bought together items
        def market_basket_analysis(df, min_support=0.01):
            # Create transaction baskets
            baskets = df.groupby('transaction_id')['product_id'].apply(list).reset_index()
            
            # Calculate item frequencies
            all_items = df['product_id'].value_counts()
            total_transactions = len(baskets)
            
            # Find frequent itemsets (simplified - just pairs)
            from itertools import combinations
            
            pair_counts = {}
            for basket in baskets['product_id']:
                if len(basket) >= 2:
                    for pair in combinations(sorted(basket), 2):
                        pair_counts[pair] = pair_counts.get(pair, 0) + 1
            
            # Calculate support and confidence
            rules = []
            for (item_a, item_b), count in pair_counts.items():
                if count >= min_support * total_transactions:
                    support = count / total_transactions
                    confidence_a_to_b = count / all_items[item_a]
                    confidence_b_to_a = count / all_items[item_b]
                    
                    rules.append({
                        'item_a': item_a,
                        'item_b': item_b,
                        'support': support,
                        'confidence_a_to_b': confidence_a_to_b,
                        'confidence_b_to_a': confidence_b_to_a,
                        'lift': support / ((all_items[item_a]/total_transactions) * (all_items[item_b]/total_transactions))
                    })
            
            return pd.DataFrame(rules).sort_values('lift', ascending=False)
        
        # 4. Customer segmentation using RFM + behavioral features
        def advanced_customer_segmentation(df):
            current_date = df['transaction_date'].max()
            
            customer_features = df.groupby('customer_id').agg({
                'transaction_date': [
                    lambda x: (current_date - x.max()).days,  # Recency
                    'count'  # Frequency
                ],
                'price': ['sum', 'mean'],  # Monetary + avg order value
                'quantity': 'sum',
                'category': lambda x: x.nunique(),  # Category diversity
                'rating': 'mean'
            }).round(2)
            
            # Flatten columns
            customer_features.columns = ['_'.join(col).strip() for col in customer_features.columns]
            customer_features.columns = ['recency', 'frequency', 'monetary', 'avg_order_value', 
                                       'total_items', 'category_diversity', 'avg_rating']
            
            # Create RFM scores
            customer_features['r_score'] = pd.qcut(customer_features['recency'], 5, labels=[5,4,3,2,1])
            customer_features['f_score'] = pd.qcut(customer_features['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
            customer_features['m_score'] = pd.qcut(customer_features['monetary'], 5, labels=[1,2,3,4,5])
            
            # Advanced segmentation
            def segment_customers(row):
                if row['r_score'] >= 4 and row['f_score'] >= 4 and row['m_score'] >= 4:
                    return 'Champions'
                elif row['r_score'] >= 3 and row['f_score'] >= 3:
                    return 'Loyal Customers'
                elif row['r_score'] >= 4 and row['f_score'] <= 2:
                    return 'New Customers'
                elif row['r_score'] <= 2 and row['f_score'] >= 3:
                    return 'At Risk'
                else:
                    return 'Others'
            
            customer_features['segment'] = customer_features.apply(segment_customers, axis=1)
            
            return customer_features
        
        # 5. Product performance analysis
        product_performance = df.groupby(['product_id', 'category']).agg({
            'quantity': 'sum',
            'price': 'mean',
            'rating': ['mean', 'count'],
            'customer_id': 'nunique'
        }).round(2)
        
        product_performance.columns = ['total_sold', 'avg_price', 'avg_rating', 
                                     'rating_count', 'unique_customers']
        product_performance = product_performance.reset_index()
        
        # Calculate revenue and popularity scores
        product_performance['revenue'] = product_performance['total_sold'] * product_performance['avg_price']
        product_performance['popularity_score'] = (
            product_performance['unique_customers'] * 0.4 + 
            product_performance['avg_rating'] * 0.3 + 
            (product_performance['total_sold'] / product_performance['total_sold'].max()) * 0.3
        )
        
        # Execute analyses
        market_basket_rules = market_basket_analysis(df)
        customer_segments = advanced_customer_segmentation(df)
        
        return {
            'similarity_matrix': similarity_df,
            'market_basket_rules': market_basket_rules.head(20),
            'customer_segments': customer_segments,
            'product_performance': product_performance.sort_values('popularity_score', ascending=False),
            'segment_summary': customer_segments['segment'].value_counts(),
            'top_cross_sell_opportunities': market_basket_rules.head(10)[['item_a', 'item_b', 'lift', 'confidence_a_to_b']]
        }
    
    results = build_recommendation_system(df)
    
    # Business insights
    print("=== E-COMMERCE RECOMMENDATION ANALYSIS ===")
    print(f"Dataset: {len(df)} transactions, {df['customer_id'].nunique()} customers, {df['product_id'].nunique()} products")
    print("\nCustomer Segmentation:")
    print(results['segment_summary'])
    print("\nTop Cross-Selling Opportunities (by lift):")
    print(results['top_cross_sell_opportunities'])
    print(f"\nTop performing product: {results['product_performance'].iloc[0]['product_id']} (score: {results['product_performance'].iloc[0]['popularity_score']:.3f})")
    
    return results

ecommerce_results = ecommerce_recommendation_analysis()
```

**Time Complexity:** O(n²) for similarity matrix, O(n*m²) for market basket analysis
**Space Complexity:** O(n*m) where n=customers, m=products

---

### Q14: Financial Portfolio Risk Management
**Problem:** You're a quantitative analyst at an investment firm. Analyze a portfolio of stocks to calculate risk metrics, optimize allocation, and detect anomalous trading patterns.

**Category:** Financial analytics, risk management, anomaly detection
**Difficulty:** Very Hard

```python
def portfolio_risk_analysis():
    """
    Business Context: Investment firm needs comprehensive portfolio analysis
    including risk metrics, correlation analysis, and anomaly detection.
    """
    
    np.random.seed(42)
    
    # Generate realistic stock data
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'PG']
    start_date = pd.Timestamp('2020-01-01')
    end_date = pd.Timestamp('2023-12-31')
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Simulate correlated stock returns
    from scipy.stats import multivariate_normal
    
    # Create correlation structure (tech stocks more correlated)
    correlation_matrix = np.eye(len(tickers))
    tech_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA']
    tech_indices = [i for i, ticker in enumerate(tickers) if ticker in tech_stocks]
    
    for i in tech_indices:
        for j in tech_indices:
            if i != j:
                correlation_matrix[i, j] = 0.6
    
    # Generate daily returns
    daily_returns = multivariate_normal.rvs(
        mean=np.array([0.0008] * len(tickers)),  # ~20% annual return
        cov=correlation_matrix * 0.0004,  # ~20% annual volatility
        size=len(dates)
    )
    
    returns_df = pd.DataFrame(daily_returns, index=dates, columns=tickers)
    
    # Convert to price data
    prices_df = (1 + returns_df).cumprod() * 100  # Start at $100
    
    # Add some realistic features
    volumes = {}
    for ticker in tickers:
        base_volume = np.random.randint(1000000, 10000000)  # Base daily volume
        volume_variation = np.random.normal(1, 0.3, len(dates))
        volumes[ticker] = (base_volume * volume_variation).clip(min=100000)
    
    volume_df = pd.DataFrame(volumes, index=dates)
    
    # Portfolio weights (example allocation)
    portfolio_weights = pd.Series({
        'AAPL': 0.15, 'GOOGL': 0.12, 'MSFT': 0.13, 'AMZN': 0.10, 'TSLA': 0.08,
        'META': 0.07, 'NVDA': 0.09, 'JPM': 0.10, 'JNJ': 0.08, 'PG': 0.08
    })
    
    def comprehensive_portfolio_analysis(prices_df, returns_df, volume_df, weights):
        
        # 1. Risk Metrics Calculation
        def calculate_risk_metrics(returns, weights):
            # Portfolio returns
            portfolio_returns = (returns * weights).sum(axis=1)
            
            # Basic metrics
            annual_return = portfolio_returns.mean() * 252
            annual_vol = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / annual_vol  # Assuming risk-free rate = 0
            
            # Value at Risk (VaR)
            var_95 = np.percentile(portfolio_returns, 5)
            var_99 = np.percentile(portfolio_returns, 1)
            
            # Conditional VaR (Expected Shortfall)
            cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
            cvar_99 = portfolio_returns[portfolio_returns <= var_99].mean()
            
            # Maximum Drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Beta calculation (vs market proxy - average of all stocks)
            market_proxy = returns.mean(axis=1)
            portfolio_beta = np.cov(portfolio_returns, market_proxy)[0, 1] / np.var(market_proxy)
            
            return {
                'annual_return': annual_return,
                'annual_volatility': annual_vol,
                'sharpe_ratio': sharpe_ratio,
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'max_drawdown': max_drawdown,
                'portfolio_beta': portfolio_beta,
                'portfolio_returns': portfolio_returns
            }
        
        # 2. Correlation and Factor Analysis
        def correlation_analysis(returns):
            # Rolling correlation analysis
            window = 60  # 60-day rolling correlation
            
            correlations = {}
            for i, stock1 in enumerate(returns.columns):
                for j, stock2 in enumerate(returns.columns[i+1:], i+1):
                    rolling_corr = returns[stock1].rolling(window).corr(returns[stock2])
                    correlations[f'{stock1}_{stock2}'] = rolling_corr
            
            corr_df = pd.DataFrame(correlations)
            
            # Current correlation matrix
            current_corr = returns.corr()
            
            # Eigenvalue decomposition for factor analysis
            eigenvals, eigenvecs = np.linalg.eigh(current_corr)
            
            # Explained variance by factors
            explained_variance = eigenvals[::-1] / eigenvals.sum()
            
            return {
                'correlation_matrix': current_corr,
                'rolling_correlations': corr_df,
                'eigenvalues': eigenvals[::-1],
                'explained_variance': explained_variance
            }
        
        # 3. Anomaly Detection
        def detect_anomalies(returns, prices, volumes):
            anomalies = []
            
            for ticker in returns.columns:
                # Z-score based anomaly detection for returns
                z_scores = np.abs(stats.zscore(returns[ticker]))
                return_anomalies = returns.index[z_scores > 3]
                
                # Volume spike detection
                volume_z = np.abs(stats.zscore(volumes[ticker]))
                volume_anomalies = volumes.index[volume_z > 3]
                
                # Price gap detection (day-to-day price changes > 10%)
                price_changes = prices[ticker].pct_change()
                gap_anomalies = prices.index[np.abs(price_changes) > 0.10]
                
                anomalies.append({
                    'ticker': ticker,
                    'return_anomalies': len(return_anomalies),
                    'volume_anomalies': len(volume_anomalies),
                    'price_gaps': len(gap_anomalies),
                    'latest_return_anomaly': return_anomalies[-1] if len(return_anomalies) > 0 else None,
                    'latest_volume_anomaly': volume_anomalies[-1] if len(volume_anomalies) > 0 else None
                })
            
            return pd.DataFrame(anomalies)
        
        # 4. Optimization Analysis
        def portfolio_optimization(returns, weights):
            # Mean-variance optimization (simplified)
            mean_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            
            # Current portfolio metrics
            current_return = np.dot(weights, mean_returns)
            current_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            current_sharpe = current_return / current_vol
            
            # Equally weighted portfolio comparison
            n_assets = len(returns.columns)
            equal_weights = pd.Series(1/n_assets, index=returns.columns)
            equal_return = np.dot(equal_weights, mean_returns)
            equal_vol = np.sqrt(np.dot(equal_weights.T, np.dot(cov_matrix, equal_weights)))
            equal_sharpe = equal_return / equal_vol
            
            # Risk parity weights (inverse volatility)
            individual_vols = np.sqrt(np.diag(cov_matrix))
            risk_parity_weights = (1/individual_vols) / (1/individual_vols).sum()
            risk_parity_weights = pd.Series(risk_parity_weights, index=returns.columns)
            
            rp_return = np.dot(risk_parity_weights, mean_returns)
            rp_vol = np.sqrt(np.dot(risk_parity_weights.T, np.dot(cov_matrix, risk_parity_weights)))
            rp_sharpe = rp_return / rp_vol
            
            return {
                'current': {'return': current_return, 'volatility': current_vol, 'sharpe': current_sharpe},
                'equal_weight': {'return': equal_return, 'volatility': equal_vol, 'sharpe': equal_sharpe},
                'risk_parity': {'return': rp_return, 'volatility': rp_vol, 'sharpe': rp_sharpe, 'weights': risk_parity_weights}
            }
        
        # Execute all analyses
        risk_metrics = calculate_risk_metrics(returns_df, weights)
        correlation_results = correlation_analysis(returns_df)
        anomaly_results = detect_anomalies(returns_df, prices_df, volume_df)
        optimization_results = portfolio_optimization(returns_df, weights)
        
        # Stress testing - simulate market crash scenario
        def stress_test(returns, weights, scenarios):
            results = {}
            for scenario_name, shock in scenarios.items():
                shocked_returns = returns.copy()
                if scenario_name == 'market_crash':
                    # Apply -30% shock to all stocks
                    shocked_returns.iloc[-1] = -0.30
                elif scenario_name == 'tech_crash':
                    # Apply -50% shock to tech stocks only
                    tech_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA']
                    shocked_returns.loc[shocked_returns.index[-1], tech_stocks] = -0.50
                
                portfolio_loss = (shocked_returns.iloc[-1] * weights).sum()
                results[scenario_name] = portfolio_loss
            
            return results
        
        stress_scenarios = {
            'market_crash': -0.30,
            'tech_crash': -0.50
        }
        
        stress_results = stress_test(returns_df, weights, stress_scenarios)
        
        return {
            'risk_metrics': risk_metrics,
            'correlation_analysis': correlation_results,
            'anomalies': anomaly_results,
            'optimization': optimization_results,
            'stress_test': stress_results
        }
    
    results = comprehensive_portfolio_analysis(prices_df, returns_df, volume_df, portfolio_weights)
    
    # Business insights
    print("=== PORTFOLIO RISK ANALYSIS ===")
    print(f"Portfolio Performance:")
    print(f"  Annual Return: {results['risk_metrics']['annual_return']:.2%}")
    print(f"  Annual Volatility: {results['risk_metrics']['annual_volatility']:.2%}")
    print(f"  Sharpe Ratio: {results['risk_metrics']['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {results['risk_metrics']['max_drawdown']:.2%}")
    print(f"  95% VaR: {results['risk_metrics']['var_95']:.3f}")
    
    print(f"\nPortfolio Optimization Comparison:")
    for strategy, metrics in results['optimization'].items():
        if strategy != 'risk_parity' or 'weights' not in metrics:
            print(f"  {strategy.replace('_', ' ').title()}: Return={metrics['return']:.2%}, Vol={metrics['volatility']:.2%}, Sharpe={metrics['sharpe']:.3f}")
    
    print(f"\nStress Test Results:")
    for scenario, loss in results['stress_test'].items():
        print(f"  {scenario.replace('_', ' ').title()}: {loss:.2%} loss")
    
    print(f"\nAnomalies Detected:")
    total_anomalies = results['anomalies'][['return_anomalies', 'volume_anomalies', 'price_gaps']].sum().sum()
    print(f"  Total anomalous events: {total_anomalies}")
    
    return results

portfolio_results = portfolio_risk_analysis()
```

**Time Complexity:** O(n²*t) for correlation calculations, O(n³) for optimization
**Space Complexity:** O(n*t) where n=assets, t=time periods

---

### Q15: Healthcare Patient Outcome Prediction
**Problem:** You're a data scientist at a hospital system. Analyze patient data to predict readmission risk, identify high-cost patients, and optimize resource allocation.

**Category:** Healthcare analytics, predictive modeling, survival analysis
**Difficulty:** Very Hard

```python
def healthcare_analytics():
    """
    Business Context: Hospital system wants to predict patient outcomes,
    reduce readmissions, and optimize resource allocation.
    """
    
    np.random.seed(42)
    
    # Generate realistic patient data
    n_patients = 10000
    
    # Patient demographics
    ages = np.random.normal(65, 15, n_patients).clip(18, 95).astype(int)
    genders = np.random.choice(['M', 'F'], n_patients, p=[0.48, 0.52])
    
    # Medical conditions (simplified)
    conditions = ['Diabetes', 'Hypertension', 'Heart Disease', 'COPD', 'Cancer', 'Kidney Disease']
    
    patients = []
    for i in range(n_patients):
        patient_id = f'P{i:06d}'
        age = ages[i]
        gender = genders[i]
        
        # Age-related condition probabilities
        condition_probs = {
            'Diabetes': 0.15 + (age - 40) * 0.004,
            'Hypertension': 0.20 + (age - 30) * 0.006,
            'Heart Disease': 0.10 + (age - 50) * 0.005,
            'COPD': 0.08 + (age - 45) * 0.003,
            'Cancer': 0.05 + (age - 40) * 0.002,
            'Kidney Disease': 0.06 + (age - 50) * 0.003
        }
        
        # Patient conditions
        patient_conditions = []
        for condition, prob in condition_probs.items():
            if np.random.random() < max(0, min(prob, 0.5)):
                patient_conditions.append(condition)
        
        # Generate hospital visits
        n_visits = np.random.poisson(2) + 1  # At least 1 visit
        
        for visit in range(n_visits):
            # Length of stay influenced by age and conditions
            base_los = 3
            age_factor = (age - 40) * 0.05
            condition_factor = len(patient_conditions) * 0.5
            length_of_stay = max(1, int(np.random.exponential(base_los + age_factor + condition_factor)))
            
            # Admission date
            admission_date = pd.Timestamp('2023-01-01') + pd.Timedelta(days=np.random.randint(0, 365))
            discharge_date = admission_date + pd.Timedelta(days=length_of_stay)
            
            # Department
            if patient_conditions:
                if 'Heart Disease' in patient_conditions:
                    department = np.random.choice(['Cardiology', 'ICU', 'Emergency'], p=[0.6, 0.2, 0.2])
                elif 'Cancer' in patient_conditions:
                    department = np.random.choice(['Oncology', 'Surgery', 'ICU'], p=[0.5, 0.3, 0.2])
                else:
                    department = np.random.choice(['Internal Medicine', 'Emergency', 'Surgery'], p=[0.5, 0.3, 0.2])
            else:
                department = np.random.choice(['Emergency', 'Surgery', 'Internal Medicine'], p=[0.4, 0.3, 0.3])
            
            # Costs influenced by LOS, department, and conditions
            base_cost = 1000
            los_cost = length_of_stay * 500
            dept_multiplier = {'ICU': 3, 'Surgery': 2.5, 'Cardiology': 2, 'Oncology': 2.2, 
                             'Emergency': 1.2, 'Internal Medicine': 1}
            condition_cost = len(patient_conditions) * 200
            
            total_cost = (base_cost + los_cost + condition_cost) * dept_multiplier.get(department, 1)
            total_cost = max(500, int(np.random.normal(total_cost, total_cost * 0.2)))
            
            # Readmission risk factors
            readmission_risk = 0.1  # Base 10% risk
            if age > 70: readmission_risk += 0.05
            if len(patient_conditions) >= 3: readmission_risk += 0.08
            if department == 'ICU': readmission_risk += 0.12
            if length_of_stay > 7: readmission_risk += 0.06
            
            readmitted = np.random.random() < readmission_risk
            readmission_days = np.random.randint(1, 31) if readmitted else None
            
            patients.append({
                'patient_id': patient_id,
                'visit_id': f'{patient_id}_V{visit:02d}',
                'age': age,
                'gender': gender,
                'conditions': ','.join(patient_conditions) if patient_conditions else 'None',
                'n_conditions': len(patient_conditions),
                'admission_date': admission_date,
                'discharge_date': discharge_date,
                'length_of_stay': length_of_stay,
                'department': department,
                'total_cost': total_cost,
                'readmitted': readmitted,
                'readmission_days': readmission_days
            })
    
    df = pd.DataFrame(patients)
    
    def comprehensive_healthcare_analysis(df):
        
        # 1. Readmission Risk Analysis
        def readmission_analysis(df):
            # Overall readmission rate
            readmission_rate = df['readmitted'].mean()
            
            # Risk factors analysis
            risk_factors = {}
            
            # Age groups
            df['age_group'] = pd.cut(df['age'], bins=[0, 50, 65, 80, 100], 
                                   labels=['<50', '50-65', '65-80', '80+'])
            age_risk = df.groupby('age_group')['readmitted'].agg(['mean', 'count'])
            
            # Conditions
            condition_risk = df.groupby('n_conditions')['readmitted'].agg(['mean', 'count'])
            
            # Department
            dept_risk = df.groupby('department')['readmitted'].agg(['mean', 'count']).sort_values('mean', ascending=False)
            
            # Length of stay
            df['los_group'] = pd.cut(df['length_of_stay'], bins=[0, 3, 7, 14, float('inf')], 
                                   labels=['1-3', '4-7', '8-14', '15+'])
            los_risk = df.groupby('los_group')['readmitted'].agg(['mean', 'count'])
            
            # Predictive model features
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report, roc_auc_score
            
            # Prepare features
            features = pd.get_dummies(df[['age', 'gender', 'n_conditions', 'length_of_stay', 'department']])
            target = df['readmitted']
            
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
            
            # Train model
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': features.columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Model performance
            y_pred = rf_model.predict(X_test)
            y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
            
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            return {
                'overall_rate': readmission_rate,
                'age_risk': age_risk,
                'condition_risk': condition_risk,
                'department_risk': dept_risk,
                'los_risk': los_risk,
                'feature_importance': feature_importance.head(10),
                'model_auc': auc_score,
                'classification_report': classification_report(y_test, y_pred)
            }
        
        # 2. Cost Analysis and High-Cost Patient Identification
        def cost_analysis(df):
            # Overall cost statistics
            total_cost = df['total_cost'].sum()
            avg_cost_per_visit = df['total_cost'].mean()
            avg_cost_per_patient = df.groupby('patient_id')['total_cost'].sum().mean()
            
            # High-cost patients (top 10% by total cost)
            patient_costs = df.groupby('patient_id').agg({
                'total_cost': 'sum',
                'visit_id': 'count',
                'length_of_stay': 'sum',
                'n_conditions': 'max',
                'readmitted': 'any',
                'age': 'first'
            }).rename(columns={'visit_id': 'n_visits'})
            
            high_cost_threshold = patient_costs['total_cost'].quantile(0.9)
            high_cost_patients = patient_costs[patient_costs['total_cost'] >= high_cost_threshold]
            
            # Cost drivers analysis
            cost_by_dept = df.groupby('department')['total_cost'].agg(['mean', 'sum', 'count'])
            cost_by_conditions = df.groupby('n_conditions')['total_cost'].agg(['mean', 'sum', 'count'])
            cost_by_los = df.groupby('los_group')['total_cost'].agg(['mean', 'sum', 'count'])
            
            # Cost prediction model
            cost_features = pd.get_dummies(df[['age', 'gender', 'n_conditions', 'length_of_stay', 'department']])
            cost_target = df['total_cost']
            
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_squared_error, r2_score
            
            X_train, X_test, y_train, y_test = train_test_split(cost_features, cost_target, test_size=0.2, random_state=42)
            
            rf_cost_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_cost_model.fit(X_train, y_train)
            
            y_pred = rf_cost_model.predict(X_test)
            cost_r2 = r2_score(y_test, y_pred)
            cost_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            cost_feature_importance = pd.DataFrame({
                'feature': cost_features.columns,
                'importance': rf_cost_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return {
                'total_healthcare_cost': total_cost,
                'avg_cost_per_visit': avg_cost_per_visit,
                'avg_cost_per_patient': avg_cost_per_patient,
                'high_cost_patients': high_cost_patients.sort_values('total_cost', ascending=False),
                'cost_by_department': cost_by_dept.sort_values('mean', ascending=False),
                'cost_by_conditions': cost_by_conditions,
                'cost_prediction_r2': cost_r2,
                'cost_prediction_rmse': cost_rmse,
                'cost_drivers': cost_feature_importance.head(10)
            }
        
        # 3. Resource Allocation and Capacity Planning
        def resource_allocation_analysis(df):
            # Daily admission patterns
            df['admission_day'] = df['admission_date'].dt.day_name()
            df['admission_month'] = df['admission_date'].dt.month
            
            daily_admissions = df.groupby('admission_day').size()
            monthly_admissions = df.groupby('admission_month').size()
            
            # Department utilization
            dept_utilization = df.groupby(['department', df['admission_date'].dt.date]).size().reset_index()
            dept_utilization.columns = ['department', 'date', 'admissions']
            
            dept_capacity = dept_utilization.groupby('department')['admissions'].agg(['mean', 'max', 'std'])
            
            # Length of stay patterns
            los_by_dept = df.groupby('department')['length_of_stay'].agg(['mean', 'median', 'std'])
            
            # Bed occupancy simulation
            def simulate_bed_occupancy(df):
                # Simplified bed occupancy calculation
                df_sorted = df.sort_values('admission_date')
                
                occupancy_data = []
                for _, row in df_sorted.iterrows():
                    for day in range(row['length_of_stay']):
                        occupancy_date = row['admission_date'] + pd.Timedelta(days=day)
                        occupancy_data.append({
                            'date': occupancy_date,
                            'department': row['department'],
                            'patient_id': row['patient_id']
                        })
                
                occupancy_df = pd.DataFrame(occupancy_data)
                daily_occupancy = occupancy_df.groupby(['date', 'department']).size().reset_index(name='beds_occupied')
                
                # Peak occupancy by department
                peak_occupancy = daily_occupancy.groupby('department')['beds_occupied'].max()
                avg_occupancy = daily_occupancy.groupby('department')['beds_occupied'].mean()
                
                return {
                    'daily_occupancy': daily_occupancy,
                    'peak_occupancy': peak_occupancy,
                    'average_occupancy': avg_occupancy
                }
            
            occupancy_results = simulate_bed_occupancy(df)
            
            return {
                'daily_admission_pattern': daily_admissions,
                'monthly_admission_pattern': monthly_admissions,
                'department_capacity_needs': dept_capacity,
                'length_of_stay_patterns': los_by_dept,
                'occupancy_analysis': occupancy_results
            }
        
        # 4. Patient Journey and Outcome Analysis
        def patient_journey_analysis(df):
            # Multi-visit patients
            patient_summary = df.groupby('patient_id').agg({
                'visit_id': 'count',
                'total_cost': ['sum', 'mean'],
                'length_of_stay': 'sum',
                'readmitted': 'any',
                'department': lambda x: x.nunique(),
                'admission_date': ['min', 'max']
            })
            
            # Flatten column names
            patient_summary.columns = ['_'.join(col).strip() for col in patient_summary.columns]
            patient_summary.columns = ['n_visits', 'total_cost', 'avg_cost_per_visit', 
                                     'total_los', 'ever_readmitted', 'n_departments',
                                     'first_visit', 'last_visit']
            
            # Patient complexity scoring
            patient_summary['complexity_score'] = (
                patient_summary['n_visits'] * 0.3 +
                (patient_summary['total_cost'] / patient_summary['total_cost'].max()) * 0.4 +
                patient_summary['n_departments'] * 0.2 +
                patient_summary['ever_readmitted'].astype(int) * 0.1
            )
            
            # Care continuity analysis
            patient_summary['care_span_days'] = (patient_summary['last_visit'] - patient_summary['first_visit']).dt.days
            
            # Segment patients by complexity
            patient_summary['complexity_tier'] = pd.qcut(
                patient_summary['complexity_score'], 
                q=4, 
                labels=['Low', 'Medium', 'High', 'Very High']
            )
            
            complexity_analysis = patient_summary.groupby('complexity_tier').agg({
                'total_cost': ['mean', 'sum'],
                'n_visits': 'mean',
                'total_los': 'mean',
                'ever_readmitted': 'mean'
            })
            
            return {
                'patient_summary': patient_summary,
                'complexity_distribution': patient_summary['complexity_tier'].value_counts(),
                'complexity_analysis': complexity_analysis,
                'high_complexity_patients': patient_summary[patient_summary['complexity_tier'] == 'Very High'].sort_values('complexity_score', ascending=False)
            }
        
        # Execute all analyses
        readmission_results = readmission_analysis(df)
        cost_results = cost_analysis(df)
        resource_results = resource_allocation_analysis(df)
        journey_results = patient_journey_analysis(df)
        
        return {
            'readmission_analysis': readmission_results,
            'cost_analysis': cost_results,
            'resource_allocation': resource_results,
            'patient_journey': journey_results
        }
    
    results = comprehensive_healthcare_analysis(df)
    
    # Business insights
    print("=== HEALTHCARE ANALYTICS DASHBOARD ===")
    print(f"Dataset: {len(df)} visits, {df['patient_id'].nunique()} unique patients")
    
    print(f"\nREADMISSION ANALYSIS:")
    print(f"  Overall readmission rate: {results['readmission_analysis']['overall_rate']:.1%}")
    print(f"  Prediction model AUC: {results['readmission_analysis']['model_auc']:.3f}")
    print(f"  Highest risk department: {results['readmission_analysis']['department_risk'].index[0]} ({results['readmission_analysis']['department_risk'].iloc[0]['mean']:.1%})")
    
    print(f"\nCOST ANALYSIS:")
    print(f"  Total healthcare costs: ${results['cost_analysis']['total_healthcare_cost']:,.0f}")
    print(f"  Average cost per visit: ${results['cost_analysis']['avg_cost_per_visit']:,.0f}")
    print(f"  Cost prediction R²: {results['cost_analysis']['cost_prediction_r2']:.3f}")
    print(f"  High-cost patients (top 10%): {len(results['cost_analysis']['high_cost_patients'])} patients")
    
    print(f"\nRESOURCE ALLOCATION:")
    peak_dept = results['resource_allocation']['occupancy_analysis']['peak_occupancy'].idxmax()
    peak_beds = results['resource_allocation']['occupancy_analysis']['peak_occupancy'].max()
    print(f"  Peak capacity need: {peak_dept} department ({peak_beds} beds)")
    print(f"  Average LOS by department:")
    for dept, los in results['resource_allocation']['length_of_stay_patterns']['mean'].head(3).items():
        print(f"    {dept}: {los:.1f} days")
    
    print(f"\nPATIENT COMPLEXITY:")
    complexity_dist = results['patient_journey']['complexity_distribution']
    print(f"  Very high complexity patients: {complexity_dist['Very High']} ({complexity_dist['Very High']/complexity_dist.sum():.1%})")
    print(f"  These patients represent significant care management opportunities")
    
    return results

healthcare_results = healthcare_analytics()
```

**Time Complexity:** O(n²) for occupancy simulation, O(n log n) for ML models
**Space Complexity:** O(n*k) where k is number of features

---

### Q16: Social Media Content Optimization
**Problem:** You're working for a social media platform. Analyze user engagement patterns, detect trending content, and optimize content recommendation algorithms.

**Category:** Social media analytics, NLP, recommendation systems
**Difficulty:** Very Hard

```python
def social_media_analytics():
    """
    Business Context: Social media platform needs to understand user behavior,
    optimize content recommendations, and identify trending topics.
    """
    
    np.random.seed(42)
    
    # Generate realistic social media data
    n_users = 5000
    n_posts = 50000
    
    # User demographics
    user_types = ['Casual', 'Influencer', 'Business', 'Creator']
    users = []
    
    for i in range(n_users):
        user_type = np.random.choice(user_types, p=[0.7, 0.1, 0.1, 0.1])
        
        # Follower count based on user type
        if user_type == 'Influencer':
            followers = np.random.lognormal(10, 1)  # 10k-1M followers
        elif user_type == 'Creator':
            followers = np.random.lognormal(8, 1)   # 1k-100k followers
        elif user_type == 'Business':
            followers = np.random.lognormal(7, 1)   # 500-50k followers
        else:  # Casual
            followers = np.random.lognormal(5, 1)   # 100-10k followers
        
        users.append({
            'user_id': f'U{i:06d}',
            'user_type': user_type,
            'followers': int(followers),
            'account_age_days': np.random.randint(30, 2000),
            'verified': np.random.random() < (0.8 if user_type == 'Influencer' else 0.1)
        })
    
    user_df = pd.DataFrame(users)
    
    # Content categories and topics
    categories = ['Technology', 'Entertainment', 'Sports', 'News', 'Lifestyle', 'Education']
    trending_topics = ['AI', 'Climate', 'Election', 'Sports', 'Celebrity', 'Tech']
    
    # Generate posts
    posts = []
    for i in range(n_posts):
        user = user_df.sample(1).iloc[0]
        category = np.random.choice(categories)
        
        # Post characteristics based on user type
        if user['user_type'] == 'Influencer':
            base_engagement = 1000
            content_quality = np.random.normal(0.8, 0.1)
        elif user['user_type'] == 'Creator':
            base_engagement = 500
            content_quality = np.random.normal(0.7, 0.15)
        else:
            base_engagement = 50
            content_quality = np.random.normal(0.5, 0.2)
        
        # Time of posting affects engagement
        post_time = pd.Timestamp('2023-01-01') + pd.Timedelta(days=np.random.randint(0, 365))
        hour = np.random.choice(24, p=[0.02, 0.01, 0.01, 0.01, 0.02, 0.03, 0.04, 0.06, 
                                     0.08, 0.09, 0.10, 0.11, 0.10, 0.08, 0.07, 0.06, 
                                     0.05, 0.04, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02])
        
        post_datetime = post_time.replace(hour=hour)
        
        # Engagement metrics
        follower_factor = min(user['followers'] / 1000, 100)  # Cap influence
        time_factor = 1.2 if 9 <= hour <= 11 or 19 <= hour <= 21 else 0.8  # Peak hours
        
        expected_engagement = base_engagement * content_quality * time_factor * (follower_factor ** 0.3)
        
        likes = max(0, int(np.random.poisson(expected_engagement)))
        shares = max(0, int(np.random.poisson(likes * 0.1)))
        comments = max(0, int(np.random.poisson(likes * 0.05)))
        
        # Viral content (rare but high engagement)
        if np.random.random() < 0.001:  # 0.1% chance of viral
            viral_multiplier = np.random.uniform(10, 100)
            likes = int(likes * viral_multiplier)
            shares = int(shares * viral_multiplier)
            comments = int(comments * viral_multiplier)
        
        posts.append({
            'post_id': f'P{i:08d}',
            'user_id': user['user_id'],
            'user_type': user['user_type'],
            'user_followers': user['followers'],
            'category': category,
            'post_datetime': post_datetime,
            'post_hour': hour,
            'likes': likes,
            'shares': shares,
            'comments': comments,
            'total_engagement': likes + shares + comments,
            'content_quality_score': max(0, min(1, content_quality))
        })
    
    posts_df = pd.DataFrame(posts)
    
    def comprehensive_social_media_analysis(posts_df, user_df):
        
        # 1. Engagement Pattern Analysis
        def engagement_analysis(posts_df):
            # Overall engagement statistics
            engagement_stats = posts_df[['likes', 'shares', 'comments', 'total_engagement']].describe()
            
            # Engagement by time of day
            hourly_engagement = posts_df.groupby('post_hour').agg({
                'total_engagement': ['mean', 'sum'],
                'post_id': 'count'
            }).round(2)
            hourly_engagement.columns = ['avg_engagement', 'total_engagement', 'post_count']
            
            # Engagement by day of week
            posts_df['day_of_week'] = posts_df['post_datetime'].dt.day_name()
            daily_engagement = posts_df.groupby('day_of_week').agg({
                'total_engagement': ['mean', 'sum'],
                'post_id': 'count'
            }).round(2)
            daily_engagement.columns = ['avg_engagement', 'total_engagement', 'post_count']
            
            # Engagement by user type
            user_type_engagement = posts_df.groupby('user_type').agg({
                'total_engagement': ['mean', 'sum'],
                'likes': 'mean',
                'shares': 'mean',
                'comments': 'mean'
            }).round(2)
            
            # Engagement rate calculation
            posts_df['engagement_rate'] = posts_df['total_engagement'] / (posts_df['user_followers'] + 1)
            
            # Top performing content
            top_posts = posts_df.nlargest(20, 'total_engagement')[
                ['post_id', 'user_type', 'category', 'total_engagement', 'engagement_rate']
            ]
            
            return {
                'engagement_stats': engagement_stats,
                'hourly_patterns': hourly_engagement,
                'daily_patterns': daily_engagement,
                'user_type_performance': user_type_engagement,
                'top_posts': top_posts
            }
        
        # 2. Content Performance and Trending Analysis
        def trending_analysis(posts_df):
            # Category performance
            category_performance = posts_df.groupby('category').agg({
                'total_engagement': ['mean', 'sum', 'count'],
                'shares': 'sum',  # Viral potential
                'engagement_rate': 'mean'
            }).round(3)
            
            # Time-based trending detection
            posts_df['post_date'] = posts_df['post_datetime'].dt.date
            
            # Daily engagement trends
            daily_trends = posts_df.groupby(['post_date', 'category']).agg({
                'total_engagement': 'sum',
                'post_id': 'count'
            }).reset_index()
            
            # Calculate engagement velocity (recent vs historical)
            recent_period = posts_df['post_datetime'].max() - pd.Timedelta(days=7)
            recent_posts = posts_df[posts_df['post_datetime'] >= recent_period]
            historical_posts = posts_df[posts_df['post_datetime'] < recent_period]
            
            recent_avg = recent_posts.groupby('category')['total_engagement'].mean()
            historical_avg = historical_posts.groupby('category')['total_engagement'].mean()
            
            trend_velocity = ((recent_avg - historical_avg) / historical_avg * 100).fillna(0)
            trend_velocity = trend_velocity.sort_values(ascending=False)
            
            # Viral content detection
            engagement_threshold = posts_df['total_engagement'].quantile(0.99)
            viral_posts = posts_df[posts_df['total_engagement'] >= engagement_threshold]
            
            viral_characteristics = viral_posts.groupby(['user_type', 'category']).size().reset_index(name='viral_count')
            
            return {
                'category_performance': category_performance,
                'trend_velocity': trend_velocity,
                'viral_posts': viral_posts[['post_id', 'user_type', 'category', 'total_engagement', 'post_datetime']],
                'viral_characteristics': viral_characteristics
            }
        
        # 3. User Behavior Segmentation
        def user_segmentation(posts_df, user_df):
            # User activity patterns
            user_activity = posts_df.groupby('user_id').agg({
                'post_id': 'count',
                'total_engagement': ['mean', 'sum'],
                'likes': 'sum',
                'shares': 'sum',
                'comments': 'sum',
                'post_datetime': ['min', 'max']
            }).round(2)
            
            # Flatten column names
            user_activity.columns = ['post_count', 'avg_engagement', 'total_engagement', 
                                   'total_likes', 'total_shares', 'total_comments',
                                   'first_post', 'last_post']
            
            # Merge with user demographics
            user_activity = user_activity.merge(user_df, on='user_id', how='left')
            
            # Calculate activity metrics
            user_activity['days_active'] = (user_activity['last_post'] - user_activity['first_post']).dt.days + 1
            user_activity['posts_per_day'] = user_activity['post_count'] / user_activity['days_active']
            user_activity['engagement_per_follower'] = user_activity['total_engagement'] / (user_activity['followers'] + 1)
            
            # User segmentation based on activity and engagement
            def segment_users(row):
                if row['post_count'] >= 10 and row['avg_engagement'] >= posts_df['total_engagement'].quantile(0.8):
                    return 'Power User'
                elif row['post_count'] >= 5 and row['avg_engagement'] >= posts_df['total_engagement'].quantile(0.6):
                    return 'Active User'
                elif row['post_count'] >= 2:
                    return 'Regular User'
                else:
                    return 'Lurker'
            
            user_activity['segment'] = user_activity.apply(segment_users, axis=1)
            
            segment_analysis = user_activity.groupby('segment').agg({
                'post_count': 'mean',
                'avg_engagement': 'mean',
                'followers': 'mean',
                'posts_per_day': 'mean'
            }).round(2)
            
            return {
                'user_activity': user_activity,
                'segment_distribution': user_activity['segment'].value_counts(),
                'segment_analysis': segment_analysis
            }
        
        # 4. Content Recommendation Engine
        def recommendation_system(posts_df):
            # Content-based features
            content_features = posts_df.groupby('category').agg({
                'total_engagement': 'mean',
                'likes': 'mean',
                'shares': 'mean',
                'comments': 'mean'
            }).round(2)
            
            # User preference analysis
            user_preferences = posts_df.groupby(['user_id', 'category']).agg({
                'total_engagement': 'sum',
                'post_id': 'count'
            }).reset_index()
            
            # Create user-category preference matrix
            user_category_matrix = user_preferences.pivot_table(
                index='user_id',
                columns='category',
                values='total_engagement',
                fill_value=0
            )
            
            # Collaborative filtering - find similar users
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Only use users with sufficient activity
            active_users = posts_df.groupby('user_id').size()
            active_users = active_users[active_users >= 3].index
            
            active_matrix = user_category_matrix.loc[active_users]
            user_similarity = cosine_similarity(active_matrix)
            user_similarity_df = pd.DataFrame(
                user_similarity,
                index=active_users,
                columns=active_users
            )
            
            # Content recommendation based on similar users
            def get_recommendations(user_id, n_recommendations=5):
                if user_id not in user_similarity_df.index:
                    return "User not found or insufficient activity"
                
                # Find similar users
                similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:6]  # Top 5 similar
                
                # Get categories they engage with
                recommendations = {}
                for similar_user in similar_users.index:
                    user_cats = user_category_matrix.loc[similar_user]
                    for category, engagement in user_cats.items():
                        if engagement > 0:
                            recommendations[category] = recommendations.get(category, 0) + engagement * similar_users[similar_user]
                
                # Sort by recommendation score
                sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
                return sorted_recs[:n_recommendations]
            
            # Performance metrics by content strategy
            strategy_performance = posts_df.groupby(['user_type', 'category']).agg({
                'total_engagement': 'mean',
                'engagement_rate': 'mean'
            }).reset_index()
            
            optimal_strategies = strategy_performance.loc[strategy_performance.groupby('user_type')['total_engagement'].idxmax()]
            
            return {
                'content_features': content_features,
                'user_similarity_matrix': user_similarity_df,
                'recommendation_function': get_recommendations,
                'optimal_strategies': optimal_strategies
            }
        
        # Execute all analyses
        engagement_results = engagement_analysis(posts_df)
        trending_results = trending_analysis(posts_df)
        user_seg_results = user_segmentation(posts_df, user_df)
        recommendation_results = recommendation_system(posts_df)
        
        return {
            'engagement_analysis': engagement_results,
            'trending_analysis': trending_results,
            'user_segmentation': user_seg_results,
            'recommendation_system': recommendation_results
        }
    
    results = comprehensive_social_media_analysis(posts_df, user_df)
    
    # Business insights
    print("=== SOCIAL MEDIA ANALYTICS DASHBOARD ===")
    print(f"Dataset: {len(posts_df)} posts from {len(user_df)} users")
    
    print(f"\nENGAGEMENT INSIGHTS:")
    best_hour = results['engagement_analysis']['hourly_patterns']['avg_engagement'].idxmax()
    best_engagement = results['engagement_analysis']['hourly_patterns'].loc[best_hour, 'avg_engagement']
    print(f"  Best posting time: {best_hour}:00 (avg {best_engagement:.0f} engagement)")
    
    best_user_type = results['engagement_analysis']['user_type_performance']['total_engagement']['mean'].idxmax()
    print(f"  Best performing user type: {best_user_type}")
    
    print(f"\nTRENDING ANALYSIS:")
    top_trending = results['trending_analysis']['trend_velocity'].head(1)
    print(f"  Fastest growing category: {top_trending.index[0]} (+{top_trending.iloc[0]:.1f}%)")
    print(f"  Viral posts detected: {len(results['trending_analysis']['viral_posts'])}")
    
    print(f"\nUSER SEGMENTATION:")
    segment_dist = results['user_segmentation']['segment_distribution']
    print(f"  Power Users: {segment_dist.get('Power User', 0)} ({segment_dist.get('Power User', 0)/len(user_df)*100:.1f}%)")
    print(f"  Active Users: {segment_dist.get('Active User', 0)} ({segment_dist.get('Active User', 0)/len(user_df)*100:.1f}%)")
    
    print(f"\nCONTENT STRATEGY:")
    top_strategy = results['recommendation_system']['optimal_strategies'].iloc[0]
    print(f"  Optimal strategy for {top_strategy['user_type']}: {top_strategy['category']} content")
    print(f"  Expected engagement: {top_strategy['total_engagement']:.0f}")
    
    return results

social_media_results = social_media_analytics()
```

**Time Complexity:** O(n²) for similarity calculations, O(n log n) for trending analysis
**Space Complexity:** O(n*k) for user-content matrix

---

### Interview Success Tips:

1. **Start with the simplest solution** that works
2. **Optimize only when necessary** and explain trade-offs
3. **Handle edge cases** (empty data, single row, etc.)
4. **Explain your approach** before coding
5. **Test your solution** with sample data
6. **Discuss alternatives** and their pros/cons

### Advanced Problem-Solving Strategies:

**For Complex Business Problems:**
1. **Break down into components** - Each problem has multiple analytical layers
2. **Think about stakeholder needs** - What decisions will this analysis drive?
3. **Consider data quality** - Always validate and clean before analyzing
4. **Scale considerations** - How would this work with 10x more data?
5. **Actionable insights** - Every analysis should lead to business recommendations

**Performance Optimization Patterns:**
- Use vectorized operations over loops
- Leverage pandas groupby efficiency
- Consider memory usage with large datasets
- Use appropriate data types (categorical, int32 vs int64)
- Implement chunked processing for very large datasets

---

### Q17: Supply Chain Optimization Analytics
**Problem:** You're working for a global manufacturing company. Analyze supply chain data to optimize inventory levels, predict demand, and identify bottlenecks in the distribution network.

**Category:** Supply chain analytics, demand forecasting, network optimization
**Difficulty:** Very Hard

```python
def supply_chain_analytics():
    """
    Business Context: Manufacturing company needs to optimize their global
    supply chain, reduce costs, and improve delivery performance.
    """
    
    np.random.seed(42)
    
    # Supply chain network structure
    suppliers = [f'SUP_{i:03d}' for i in range(1, 51)]  # 50 suppliers
    warehouses = [f'WH_{i:02d}' for i in range(1, 21)]  # 20 warehouses
    customers = [f'CUST_{i:04d}' for i in range(1, 1001)]  # 1000 customers
    products = [f'PROD_{i:03d}' for i in range(1, 201)]  # 200 products
    
    # Geographic regions
    regions = ['North_America', 'Europe', 'Asia_Pacific', 'Latin_America']
    
    # Assign geographic locations
    supplier_locations = {sup: np.random.choice(regions) for sup in suppliers}
    warehouse_locations = {wh: np.random.choice(regions, p=[0.4, 0.3, 0.2, 0.1]) for wh in warehouses}
    customer_locations = {cust: np.random.choice(regions, p=[0.3, 0.25, 0.3, 0.15]) for cust in customers}
    
    # Product characteristics
    product_categories = ['Electronics', 'Automotive', 'Industrial', 'Consumer']
    product_info = {}
    for prod in products:
        product_info[prod] = {
            'category': np.random.choice(product_categories),
            'unit_cost': np.random.lognormal(3, 0.5),  # $10-200
            'weight_kg': np.random.lognormal(1, 0.8),  # 0.5-20kg
            'shelf_life_days': np.random.choice([30, 90, 180, 365, 9999], p=[0.1, 0.2, 0.3, 0.3, 0.1])
        }
    
    # Generate historical demand data (2 years)
    start_date = pd.Timestamp('2022-01-01')
    end_date = pd.Timestamp('2023-12-31')
    date_range = pd.date_range(start_date, end_date, freq='D')
    
    demand_data = []
    for date in date_range:
        # Seasonal patterns
        month = date.month
        day_of_year = date.dayofyear
        
        # Holiday effects
        is_holiday = month in [11, 12] or (month == 1 and date.day < 15)  # Simplified holiday season
        holiday_multiplier = 1.5 if is_holiday else 1.0
        
        # Seasonal multiplier
        seasonal_multiplier = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
        
        # Generate demand for random subset of products/customers
        n_orders = np.random.poisson(500)  # Average 500 orders per day
        
        for _ in range(n_orders):
            customer = np.random.choice(customers)
            product = np.random.choice(products)
            
            # Base demand influenced by product category and seasonality
            if product_info[product]['category'] == 'Consumer':
                base_demand = np.random.exponential(10)
            else:
                base_demand = np.random.exponential(5)
            
            quantity = max(1, int(base_demand * seasonal_multiplier * holiday_multiplier))
            
            demand_data.append({
                'date': date,
                'customer_id': customer,
                'product_id': product,
                'quantity': quantity,
                'unit_price': product_info[product]['unit_cost'] * np.random.uniform(1.2, 2.0),  # Markup
                'customer_region': customer_locations[customer]
            })
    
    demand_df = pd.DataFrame(demand_data)
    
    # Generate supply data
    supply_data = []
    for date in pd.date_range('2022-01-01', '2023-12-31', freq='W'):  # Weekly deliveries
        for supplier in suppliers:
            # Each supplier provides random subset of products
            supplier_products = np.random.choice(products, size=np.random.randint(20, 50), replace=False)
            
            for product in supplier_products:
                if np.random.random() < 0.7:  # 70% chance of delivery
                    quantity = np.random.exponential(100)
                    lead_time = np.random.randint(7, 42)  # 1-6 weeks lead time
                    
                    supply_data.append({
                        'date': date,
                        'supplier_id': supplier,
                        'product_id': product,
                        'quantity': int(quantity),
                        'unit_cost': product_info[product]['unit_cost'] * np.random.uniform(0.8, 1.2),
                        'lead_time_days': lead_time,
                        'supplier_region': supplier_locations[supplier]
                    })
    
    supply_df = pd.DataFrame(supply_data)
    
    # Generate inventory data
    inventory_data = []
    for warehouse in warehouses:
        for product in products:
            if np.random.random() < 0.6:  # 60% chance product is stocked
                current_stock = np.random.randint(0, 1000)
                safety_stock = np.random.randint(50, 200)
                max_capacity = np.random.randint(500, 2000)
                
                inventory_data.append({
                    'warehouse_id': warehouse,
                    'product_id': product,
                    'current_stock': current_stock,
                    'safety_stock': safety_stock,
                    'max_capacity': max_capacity,
                    'warehouse_region': warehouse_locations[warehouse],
                    'holding_cost_per_unit': product_info[product]['unit_cost'] * 0.02  # 2% monthly
                })
    
    inventory_df = pd.DataFrame(inventory_data)
    
    def comprehensive_supply_chain_analysis(demand_df, supply_df, inventory_df, product_info):
        
        # 1. Demand Forecasting and Pattern Analysis
        def demand_forecasting(demand_df):
            # Aggregate daily demand by product
            daily_demand = demand_df.groupby(['date', 'product_id']).agg({
                'quantity': 'sum'
            }).reset_index()
            
            # Calculate demand statistics
            demand_stats = demand_df.groupby('product_id').agg({
                'quantity': ['mean', 'std', 'sum'],
                'date': ['min', 'max', 'nunique']
            }).round(2)
            
            demand_stats.columns = ['avg_daily_demand', 'demand_std', 'total_demand', 
                                  'first_order', 'last_order', 'days_with_demand']
            
            # Demand variability coefficient
            demand_stats['cv'] = demand_stats['demand_std'] / demand_stats['avg_daily_demand']
            
            # Seasonal decomposition (simplified)
            monthly_demand = demand_df.groupby([demand_df['date'].dt.to_period('M'), 'product_id'])['quantity'].sum().reset_index()
            monthly_demand['date'] = monthly_demand['date'].dt.to_timestamp()
            
            # Calculate seasonality index for top products
            top_products = demand_stats.nlargest(20, 'total_demand').index
            seasonality = {}
            
            for product in top_products:
                product_monthly = monthly_demand[monthly_demand['product_id'] == product]
                if len(product_monthly) >= 12:
                    avg_monthly = product_monthly['quantity'].mean()
                    monthly_index = product_monthly.set_index('date')['quantity'] / avg_monthly
                    seasonality[product] = monthly_index.to_dict()
            
            # Simple forecasting using moving average and trend
            def forecast_demand(product_id, days_ahead=30):
                product_data = daily_demand[daily_demand['product_id'] == product_id].sort_values('date')
                if len(product_data) < 30:
                    return None
                
                # Moving average
                product_data['ma_7'] = product_data['quantity'].rolling(7).mean()
                product_data['ma_30'] = product_data['quantity'].rolling(30).mean()
                
                # Simple trend
                recent_trend = product_data['ma_7'].tail(7).mean() - product_data['ma_30'].tail(1).iloc[0]
                
                # Forecast
                last_ma = product_data['ma_7'].tail(1).iloc[0]
                forecast = max(0, last_ma + recent_trend * days_ahead / 7)
                
                return {
                    'forecast_quantity': forecast,
                    'confidence': 'medium' if product_data['cv'].iloc[0] < 1 else 'low'
                }
            
            return {
                'demand_statistics': demand_stats,
                'seasonality_patterns': seasonality,
                'forecast_function': forecast_demand,
                'top_products_by_demand': demand_stats.nlargest(10, 'total_demand')
            }
        
        # 2. Inventory Optimization
        def inventory_optimization(inventory_df, demand_df, supply_df):
            # Calculate inventory turnover and performance metrics
            total_demand_by_product = demand_df.groupby('product_id')['quantity'].sum()
            avg_inventory = inventory_df.groupby('product_id')['current_stock'].mean()
            
            inventory_performance = pd.DataFrame({
                'avg_inventory': avg_inventory,
                'annual_demand': total_demand_by_product,
            }).fillna(0)
            
            inventory_performance['turnover_ratio'] = inventory_performance['annual_demand'] / (inventory_performance['avg_inventory'] + 1)
            inventory_performance['days_of_supply'] = (inventory_performance['avg_inventory'] * 365) / (inventory_performance['annual_demand'] + 1)
            
            # Identify slow-moving and fast-moving inventory
            inventory_performance['movement_category'] = pd.cut(
                inventory_performance['turnover_ratio'],
                bins=[0, 2, 6, float('inf')],
                labels=['Slow', 'Medium', 'Fast']
            )
            
            # ABC Analysis (by value)
            inventory_df['inventory_value'] = inventory_df['current_stock'] * inventory_df.apply(
                lambda row: product_info[row['product_id']]['unit_cost'], axis=1
            )
            
            warehouse_inventory_value = inventory_df.groupby(['warehouse_id', 'product_id'])['inventory_value'].sum().reset_index()
            total_value_by_product = warehouse_inventory_value.groupby('product_id')['inventory_value'].sum().sort_values(ascending=False)
            
            # ABC classification
            cumsum_pct = (total_value_by_product.cumsum() / total_value_by_product.sum()) * 100
            
            abc_classification = pd.DataFrame({
                'product_id': total_value_by_product.index,
                'inventory_value': total_value_by_product.values,
                'cumsum_pct': cumsum_pct.values
            })
            
            abc_classification['ABC_category'] = abc_classification['cumsum_pct'].apply(
                lambda x: 'A' if x <= 80 else 'B' if x <= 95 else 'C'
            )
            
            # Safety stock recommendations
            demand_stats = demand_df.groupby('product_id').agg({
                'quantity': ['mean', 'std']
            })
            demand_stats.columns = ['avg_demand', 'std_demand']
            
            # Lead time analysis from supply data
            avg_lead_time = supply_df.groupby('product_id')['lead_time_days'].mean()
            
            safety_stock_calc = pd.DataFrame({
                'avg_daily_demand': demand_stats['avg_demand'],
                'demand_std': demand_stats['std_demand'],
                'avg_lead_time': avg_lead_time
            }).fillna(0)
            
            # Safety stock = Z_score * sqrt(lead_time) * demand_std
            service_level_z = 1.96  # 97.5% service level
            safety_stock_calc['recommended_safety_stock'] = (
                service_level_z * 
                np.sqrt(safety_stock_calc['avg_lead_time']) * 
                safety_stock_calc['demand_std']
            ).fillna(0).astype(int)
            
            return {
                'inventory_performance': inventory_performance,
                'abc_analysis': abc_classification,
                'safety_stock_recommendations': safety_stock_calc,
                'movement_analysis': inventory_performance['movement_category'].value_counts()
            }
        
        # 3. Supply Chain Network Analysis
        def network_analysis(demand_df, supply_df, inventory_df):
            # Regional supply-demand balance
            demand_by_region = demand_df.groupby('customer_region')['quantity'].sum()
            supply_by_region = supply_df.groupby('supplier_region')['quantity'].sum()
            
            regional_balance = pd.DataFrame({
                'demand': demand_by_region,
                'supply': supply_by_region
            }).fillna(0)
            regional_balance['net_position'] = regional_balance['supply'] - regional_balance['demand']
            regional_balance['self_sufficiency'] = regional_balance['supply'] / regional_balance['demand']
            
            # Supplier performance analysis
            supplier_performance = supply_df.groupby('supplier_id').agg({
                'quantity': 'sum',
                'unit_cost': 'mean',
                'lead_time_days': 'mean',
                'product_id': 'nunique',
                'date': 'nunique'
            }).round(2)
            
            supplier_performance.columns = ['total_quantity', 'avg_unit_cost', 'avg_lead_time', 
                                          'product_variety', 'delivery_frequency']
            
            # Supplier reliability score
            supplier_performance['reliability_score'] = (
                (1 / (supplier_performance['avg_lead_time'] / 30)) * 0.4 +  # Lead time factor
                (supplier_performance['delivery_frequency'] / 52) * 0.3 +   # Consistency
                (supplier_performance['product_variety'] / 50) * 0.3        # Flexibility
            ).round(3)
            
            # Warehouse efficiency analysis
            warehouse_performance = inventory_df.groupby(['warehouse_id', 'warehouse_region']).agg({
                'current_stock': 'sum',
                'holding_cost_per_unit': 'sum',
                'product_id': 'nunique'
            }).reset_index()
            
            warehouse_performance['utilization'] = warehouse_performance['current_stock'] / (
                inventory_df.groupby('warehouse_id')['max_capacity'].sum()
            )
            
            # Transportation cost estimation (simplified)
            def estimate_transport_costs():
                # Cross-region shipping cost matrix ($/unit)
                transport_costs = {
                    ('North_America', 'North_America'): 5,
                    ('North_America', 'Europe'): 25,
                    ('North_America', 'Asia_Pacific'): 30,
                    ('North_America', 'Latin_America'): 15,
                    ('Europe', 'Europe'): 8,
                    ('Europe', 'Asia_Pacific'): 35,
                    ('Europe', 'Latin_America'): 40,
                    ('Asia_Pacific', 'Asia_Pacific'): 6,
                    ('Asia_Pacific', 'Latin_America'): 45,
                    ('Latin_America', 'Latin_America'): 10,
                }
                
                # Make symmetric
                symmetric_costs = transport_costs.copy()
                for (origin, dest), cost in transport_costs.items():
                    if (dest, origin) not in symmetric_costs:
                        symmetric_costs[(dest, origin)] = cost
                
                return symmetric_costs
            
            transport_cost_matrix = estimate_transport_costs()
            
            return {
                'regional_balance': regional_balance.sort_values('net_position', ascending=False),
                'supplier_performance': supplier_performance.sort_values('reliability_score', ascending=False),
                'warehouse_efficiency': warehouse_performance.sort_values('utilization', ascending=False),
                'transport_cost_matrix': transport_cost_matrix
            }
        
        # 4. Risk Analysis and Mitigation
        def risk_analysis(supply_df, inventory_df, demand_df):
            # Supplier concentration risk
            total_supply = supply_df.groupby('supplier_id')['quantity'].sum().sort_values(ascending=False)
            total_supply_volume = total_supply.sum()
            supplier_concentration = (total_supply.cumsum() / total_supply_volume * 100).head(10)
            
            # Single point of failure analysis
            critical_suppliers = total_supply.head(5).index  # Top 5 suppliers
            
            supplier_dependency = []
            for supplier in critical_suppliers:
                supplier_products = supply_df[supply_df['supplier_id'] == supplier]['product_id'].unique()
                alternative_suppliers = {}
                
                for product in supplier_products:
                    alternatives = supply_df[
                        (supply_df['product_id'] == product) & 
                        (supply_df['supplier_id'] != supplier)
                    ]['supplier_id'].nunique()
                    alternative_suppliers[product] = alternatives
                
                supplier_dependency.append({
                    'supplier_id': supplier,
                    'products_supplied': len(supplier_products),
                    'avg_alternatives_per_product': np.mean(list(alternative_suppliers.values())),
                    'products_with_no_alternatives': sum(1 for alt in alternative_suppliers.values() if alt == 0)
                })
            
            dependency_df = pd.DataFrame(supplier_dependency)
            
            # Demand volatility risk
            demand_volatility = demand_df.groupby('product_id')['quantity'].agg(['mean', 'std']).reset_index()
            demand_volatility['cv'] = demand_volatility['std'] / demand_volatility['mean']
            demand_volatility['risk_level'] = pd.cut(
                demand_volatility['cv'],
                bins=[0, 0.5, 1.0, float('inf')],
                labels=['Low', 'Medium', 'High']
            )
            
            high_risk_products = demand_volatility[demand_volatility['risk_level'] == 'High']['product_id'].tolist()
            
            # Inventory risk (stockout probability)
            inventory_risk = inventory_df.copy()
            inventory_risk['stockout_risk'] = inventory_risk['current_stock'] / (inventory_risk['safety_stock'] + 1)
            
            critical_inventory = inventory_risk[inventory_risk['stockout_risk'] < 1.5].sort_values('stockout_risk')
            
            return {
                'supplier_concentration': supplier_concentration,
                'supplier_dependency': dependency_df,
                'demand_volatility_risk': demand_volatility.sort_values('cv', ascending=False),
                'high_risk_products': high_risk_products,
                'critical_inventory_levels': critical_inventory.head(20)
            }
        
        # Execute all analyses
        demand_results = demand_forecasting(demand_df)
        inventory_results = inventory_optimization(inventory_df, demand_df, supply_df)
        network_results = network_analysis(demand_df, supply_df, inventory_df)
        risk_results = risk_analysis(supply_df, inventory_df, demand_df)
        
        # Executive summary and recommendations
        def generate_recommendations():
            recommendations = []
            
            # Inventory recommendations
            slow_moving = inventory_results['movement_analysis'].get('Slow', 0)
            if slow_moving > 0:
                recommendations.append(f"Optimize {slow_moving} slow-moving products - consider demand stimulation or liquidation")
            
            # Supplier recommendations
            top_supplier_share = risk_results['supplier_concentration'].iloc[0]
            if top_supplier_share > 40:
                recommendations.append(f"Critical supplier risk: Top supplier represents {top_supplier_share:.1f}% of supply - diversify sources")
            
            # Regional balance
            unbalanced_regions = network_results['regional_balance'][network_results['regional_balance']['self_sufficiency'] < 0.8]
            if len(unbalanced_regions) > 0:
                recommendations.append(f"Supply shortfall in {len(unbalanced_regions)} regions - consider local sourcing")
            
            # Inventory risks
            critical_stock = len(risk_results['critical_inventory_levels'])
            if critical_stock > 10:
                recommendations.append(f"Stockout risk for {critical_stock} product-warehouse combinations - increase safety stock")
            
            return recommendations
        
        recommendations = generate_recommendations()
        
        return {
            'demand_analysis': demand_results,
            'inventory_optimization': inventory_results,
            'network_analysis': network_results,
            'risk_analysis': risk_results,
            'recommendations': recommendations
        }
    
    results = comprehensive_supply_chain_analysis(demand_df, supply_df, inventory_df, product_info)
    
    # Business insights
    print("=== SUPPLY CHAIN ANALYTICS DASHBOARD ===")
    print(f"Dataset: {len(demand_df)} demand records, {len(supply_df)} supply records, {len(inventory_df)} inventory positions")
    
    print(f"\nDEMAND INSIGHTS:")
    top_product = results['demand_analysis']['top_products_by_demand'].index[0]
    top_demand = results['demand_analysis']['top_products_by_demand'].iloc[0]['total_demand']
    print(f"  Highest demand product: {top_product} ({top_demand:,.0f} units)")
    
    print(f"\nINVENTORY OPTIMIZATION:")
    movement_dist = results['inventory_optimization']['movement_analysis']
    print(f"  Inventory movement: {movement_dist.get('Fast', 0)} fast, {movement_dist.get('Medium', 0)} medium, {movement_dist.get('Slow', 0)} slow")
    
    abc_dist = results['inventory_optimization']['abc_analysis']['ABC_category'].value_counts()
    print(f"  ABC Analysis: {abc_dist.get('A', 0)} A-items, {abc_dist.get('B', 0)} B-items, {abc_dist.get('C', 0)} C-items")
    
    print(f"\nSUPPLIER PERFORMANCE:")
    top_supplier = results['network_analysis']['supplier_performance'].index[0]
    top_score = results['network_analysis']['supplier_performance'].iloc[0]['reliability_score']
    print(f"  Best supplier: {top_supplier} (reliability score: {top_score:.3f})")
    
    print(f"\nRISK ASSESSMENT:")
    high_risk_products = len(results['risk_analysis']['high_risk_products'])
    print(f"  High volatility products: {high_risk_products}")
    
    critical_inventory = len(results['risk_analysis']['critical_inventory_levels'])
    print(f"  Critical inventory positions: {critical_inventory}")
    
    print(f"\nKEY RECOMMENDATIONS:")
    for i, rec in enumerate(results['recommendations'][:3], 1):
        print(f"  {i}. {rec}")
    
    return results

supply_chain_results = supply_chain_analytics()
```

**Time Complexity:** O(n*m) for network analysis, O(n log n) for optimization algorithms
**Space Complexity:** O(n*k) where n=products, k=locations

---

### Q18: Advanced A/B Testing and Experimentation Platform
**Problem:** You're building an experimentation platform for a tech company. Design and analyze A/B tests with proper statistical methods, handle multiple testing, and provide business recommendations.

**Category:** Experimental design, statistical inference, causal analysis
**Difficulty:** Very Hard

```python
def ab_testing_platform():
    """
    Business Context: Tech company needs a robust A/B testing platform
    to evaluate product changes, with proper statistical rigor.
    """
    
    np.random.seed(42)
    
    # Simulate multiple concurrent A/B tests
    experiments = {
        'checkout_flow': {
            'description': 'New checkout process vs current',
            'metric': 'conversion_rate',
            'effect_size': 0.02,  # 2 percentage point increase
            'baseline_rate': 0.15
        },
        'recommendation_algorithm': {
            'description': 'ML-based recommendations vs collaborative filtering',
            'metric': 'click_through_rate',
            'effect_size': 0.015,  # 1.5 percentage point increase
            'baseline_rate': 0.08
        },
        'pricing_strategy': {
            'description': 'Dynamic pricing vs fixed pricing',
            'metric': 'revenue_per_user',
            'effect_size': 5.0,  # $5 increase
            'baseline_mean': 45.0,
            'baseline_std': 15.0
        },
        'ui_redesign': {
            'description': 'New UI design vs current design',
            'metric': 'time_on_page',
            'effect_size': -30,  # 30 seconds decrease (improvement)
            'baseline_mean': 180,
            'baseline_std': 60
        }
    }
    
    # Generate user data
    n_users = 100000
    
    users = []
    for i in range(n_users):
        # User demographics affect baseline metrics
        age_group = np.random.choice(['18-25', '26-35', '36-45', '46-55', '55+'], 
                                   p=[0.2, 0.3, 0.25, 0.15, 0.1])
        
        device_type = np.random.choice(['mobile', 'desktop', 'tablet'], p=[0.6, 0.35, 0.05])
        
        # User engagement level affects metrics
        engagement_level = np.random.choice(['low', 'medium', 'high'], p=[0.4, 0.4, 0.2])
        
        # Geographic region
        region = np.random.choice(['US', 'EU', 'Asia', 'Other'], p=[0.4, 0.25, 0.25, 0.1])
        
        users.append({
            'user_id': f'USER_{i:06d}',
            'age_group': age_group,
            'device_type': device_type,
            'engagement_level': engagement_level,
            'region': region,
            'signup_date': pd.Timestamp('2023-01-01') + pd.Timedelta(days=np.random.randint(0, 365))
        })
    
    users_df = pd.DataFrame(users)
    
    # Generate experiment data
    def generate_experiment_data(experiment_name, experiment_config, users_df):
        # Random assignment to control/treatment (50/50 split)
        experiment_users = users_df.sample(frac=0.3)  # 30% of users in each experiment
        experiment_users['variant'] = np.random.choice(['control', 'treatment'], len(experiment_users))
        
        results = []
        
        for _, user in experiment_users.iterrows():
            # User characteristics affect baseline performance
            user_multiplier = 1.0
            
            if user['engagement_level'] == 'high':
                user_multiplier *= 1.3
            elif user['engagement_level'] == 'low':
                user_multiplier *= 0.7
            
            if user['device_type'] == 'mobile':
                user_multiplier *= 0.9  # Slightly lower performance on mobile
            
            # Generate metric based on experiment type
            if experiment_config['metric'] in ['conversion_rate', 'click_through_rate']:
                # Binary outcome
                baseline_prob = experiment_config['baseline_rate'] * user_multiplier
                
                if user['variant'] == 'treatment':
                    prob = min(0.95, baseline_prob + experiment_config['effect_size'])
                else:
                    prob = baseline_prob
                
                outcome = np.random.binomial(1, prob)
                metric_value = outcome
                
            else:
                # Continuous outcome
                if user['variant'] == 'treatment':
                    mean = experiment_config['baseline_mean'] + experiment_config['effect_size']
                else:
                    mean = experiment_config['baseline_mean']
                
                # Apply user multiplier to mean
                mean *= user_multiplier
                std = experiment_config['baseline_std']
                
                metric_value = np.random.normal(mean, std)
                if experiment_config['metric'] == 'revenue_per_user':
                    metric_value = max(0, metric_value)  # Can't have negative revenue
            
            results.append({
                'experiment': experiment_name,
                'user_id': user['user_id'],
                'variant': user['variant'],
                'age_group': user['age_group'],
                'device_type': user['device_type'],
                'engagement_level': user['engagement_level'],
                'region': user['region'],
                'metric_value': metric_value,
                'metric_type': experiment_config['metric']
            })
        
        return pd.DataFrame(results)
    
    # Generate data for all experiments
    all_experiment_data = []
    for exp_name, exp_config in experiments.items():
        exp_data = generate_experiment_data(exp_name, exp_config, users_df)
        all_experiment_data.append(exp_data)
    
    experiment_results = pd.concat(all_experiment_data, ignore_index=True)
    
    def comprehensive_ab_testing_analysis(experiment_results, experiments):
        
        # 1. Statistical Analysis Framework
        def statistical_testing(exp_data, experiment_config):
            from scipy import stats
            
            control_data = exp_data[exp_data['variant'] == 'control']['metric_value']
            treatment_data = exp_data[exp_data['variant'] == 'treatment']['metric_value']
            
            # Basic statistics
            control_stats = {
                'count': len(control_data),
                'mean': control_data.mean(),
                'std': control_data.std(),
                'sem': control_data.sem()  # Standard error of mean
            }
            
            treatment_stats = {
                'count': len(treatment_data),
                'mean': treatment_data.mean(),
                'std': treatment_data.std(),
                'sem': treatment_data.sem()
            }
            
            # Choose appropriate test
            if experiment_config['metric'] in ['conversion_rate', 'click_through_rate']:
                # Proportions test
                control_successes = int(control_data.sum())
                treatment_successes = int(treatment_data.sum())
                
                # Two-proportion z-test
                p1 = control_successes / len(control_data)
                p2 = treatment_successes / len(treatment_data)
                
                # Pooled proportion
                p_pooled = (control_successes + treatment_successes) / (len(control_data) + len(treatment_data))
                
                # Standard error
                se = np.sqrt(p_pooled * (1 - p_pooled) * (1/len(control_data) + 1/len(treatment_data)))
                
                # Z-score
                z_score = (p2 - p1) / se
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                
                # Confidence interval for difference
                se_diff = np.sqrt(p1*(1-p1)/len(control_data) + p2*(1-p2)/len(treatment_data))
                ci_lower = (p2 - p1) - 1.96 * se_diff
                ci_upper = (p2 - p1) + 1.96 * se_diff
                
                effect_size = p2 - p1
                relative_lift = (p2 - p1) / p1 if p1 > 0 else 0
                
            else:
                # T-test for continuous metrics
                t_stat, p_value = stats.ttest_ind(treatment_data, control_data, equal_var=False)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(control_data)-1)*control_data.var() + 
                                    (len(treatment_data)-1)*treatment_data.var()) / 
                                   (len(control_data) + len(treatment_data) - 2))
                cohens_d = (treatment_stats['mean'] - control_stats['mean']) / pooled_std
                
                # Confidence interval for difference
                se_diff = np.sqrt(control_stats['sem']**2 + treatment_stats['sem']**2)
                ci_lower = (treatment_stats['mean'] - control_stats['mean']) - 1.96 * se_diff
                ci_upper = (treatment_stats['mean'] - control_stats['mean']) + 1.96 * se_diff
                
                effect_size = treatment_stats['mean'] - control_stats['mean']
                relative_lift = effect_size / control_stats['mean'] if control_stats['mean'] > 0 else 0
                z_score = None
            
            # Statistical significance
            is_significant = p_value < 0.05
            
            # Statistical power calculation (post-hoc)
            from statsmodels.stats import power
            if experiment_config['metric'] in ['conversion_rate', 'click_through_rate']:
                # Power for proportions
                effect_size_cohen = abs(p2 - p1) / np.sqrt(p_pooled * (1 - p_pooled))
                statistical_power = power.ttest_power(effect_size_cohen, len(control_data), alpha=0.05)
            else:
                # Power for t-test
                statistical_power = power.ttest_power(abs(cohens_d), len(control_data), alpha=0.05)
            
            return {
                'control_stats': control_stats,
                'treatment_stats': treatment_stats,
                'effect_size': effect_size,
                'relative_lift': relative_lift,
                'p_value': p_value,
                'is_significant': is_significant,
                'confidence_interval': (ci_lower, ci_upper),
                'statistical_power': statistical_power,
                'z_score': z_score,
                'test_type': 'proportions_test' if experiment_config['metric'] in ['conversion_rate', 'click_through_rate'] else 't_test'
            }
        
        # 2. Segmentation Analysis
        def segmentation_analysis(exp_data):
            segments = ['age_group', 'device_type', 'engagement_level', 'region']
            segment_results = {}
            
            for segment in segments:
                segment_analysis = []
                
                for segment_value in exp_data[segment].unique():
                    segment_data = exp_data[exp_data[segment] == segment_value]
                    
                    if len(segment_data) < 100:  # Skip small segments
                        continue
                    
                    control_segment = segment_data[segment_data['variant'] == 'control']['metric_value']
                    treatment_segment = segment_data[segment_data['variant'] == 'treatment']['metric_value']
                    
                    if len(control_segment) > 10 and len(treatment_segment) > 10:
                        # Simple t-test for segments
                        if exp_data['metric_type'].iloc[0] in ['conversion_rate', 'click_through_rate']:
                            control_rate = control_segment.mean()
                            treatment_rate = treatment_segment.mean()
                            lift = treatment_rate - control_rate
                        else:
                            control_mean = control_segment.mean()
                            treatment_mean = treatment_segment.mean()
                            lift = treatment_mean - control_mean
                        
                        # Quick significance test
                        _, p_val = stats.ttest_ind(treatment_segment, control_segment)
                        
                        segment_analysis.append({
                            'segment_value': segment_value,
                            'control_mean': control_segment.mean(),
                            'treatment_mean': treatment_segment.mean(),
                            'lift': lift,
                            'p_value': p_val,
                            'sample_size': len(segment_data)
                        })
                
                segment_results[segment] = pd.DataFrame(segment_analysis)
            
            return segment_results
        
        # 3. Multiple Testing Correction
        def multiple_testing_correction(all_results):
            # Collect all p-values
            p_values = []
            experiment_names = []
            
            for exp_name, results in all_results.items():
                p_values.append(results['statistical_test']['p_value'])
                experiment_names.append(exp_name)
            
            # Bonferroni correction
            bonferroni_alpha = 0.05 / len(p_values)
            bonferroni_significant = [p < bonferroni_alpha for p in p_values]
            
            # Benjamini-Hochberg (FDR) correction
            from statsmodels.stats.multitest import multipletests
            fdr_results = multipletests(p_values, alpha=0.05, method='fdr_bh')
            
            correction_results = pd.DataFrame({
                'experiment': experiment_names,
                'p_value': p_values,
                'bonferroni_significant': bonferroni_significant,
                'fdr_significant': fdr_results[0],
                'fdr_adjusted_p': fdr_results[1]
            })
            
            return correction_results
        
        # 4. Business Impact Assessment
        def business_impact_assessment(exp_data, experiment_config, statistical_results):
            # Calculate business metrics
            control_data = exp_data[exp_data['variant'] == 'control']
            treatment_data = exp_data[exp_data['variant'] == 'treatment']
            
            # Sample size for full rollout estimation
            total_users = 1000000  # Assume 1M total users
            
            if experiment_config['metric'] in ['conversion_rate', 'click_through_rate']:
                # Revenue impact for conversion metrics
                avg_order_value = 75  # Assume $75 AOV
                
                current_conversions = statistical_results['control_stats']['mean'] * total_users
                projected_conversions = statistical_results['treatment_stats']['mean'] * total_users
                
                additional_conversions = projected_conversions - current_conversions
                revenue_impact = additional_conversions * avg_order_value
                
                # Confidence interval for revenue impact
                ci_lower_rev = statistical_results['confidence_interval'][0] * total_users * avg_order_value
                ci_upper_rev = statistical_results['confidence_interval'][1] * total_users * avg_order_value
                
            elif experiment_config['metric'] == 'revenue_per_user':
                # Direct revenue impact
                revenue_impact = statistical_results['effect_size'] * total_users
                ci_lower_rev = statistical_results['confidence_interval'][0] * total_users
                ci_upper_rev = statistical_results['confidence_interval'][1] * total_users
                
            else:
                # For other metrics, estimate indirect revenue impact
                revenue_impact = None  # Would need business logic to convert
                ci_lower_rev = None
                ci_upper_rev = None
            
            # Implementation cost estimation
            implementation_costs = {
                'checkout_flow': 50000,
                'recommendation_algorithm': 200000,
                'pricing_strategy': 100000,
                'ui_redesign': 75000
            }
            
            exp_name = exp_data['experiment'].iloc[0]
            implementation_cost = implementation_costs.get(exp_name, 50000)
            
            # ROI calculation
            if revenue_impact:
                annual_revenue_impact = revenue_impact * 12  # Assume monthly impact
                roi = (annual_revenue_impact - implementation_cost) / implementation_cost
            else:
                annual_revenue_impact = None
                roi = None
            
            return {
                'projected_annual_revenue_impact': annual_revenue_impact,
                'revenue_confidence_interval': (ci_lower_rev * 12 if ci_lower_rev else None, 
                                              ci_upper_rev * 12 if ci_upper_rev else None),
                'implementation_cost': implementation_cost,
                'roi': roi,
                'payback_period_months': implementation_cost / (revenue_impact if revenue_impact and revenue_impact > 0 else 1)
            }
        
        # Execute analysis for each experiment
        experiment_analyses = {}
        
        for exp_name in experiment_results['experiment'].unique():
            exp_data = experiment_results[experiment_results['experiment'] == exp_name]
            exp_config = experiments[exp_name]
            
            # Statistical testing
            statistical_test = statistical_testing(exp_data, exp_config)
            
            # Segmentation analysis
            segment_results = segmentation_analysis(exp_data)
            
            # Business impact
            business_impact = business_impact_assessment(exp_data, exp_config, statistical_test)
            
            experiment_analyses[exp_name] = {
                'statistical_test': statistical_test,
                'segmentation': segment_results,
                'business_impact': business_impact,
                'sample_size': len(exp_data)
            }
        
        # Multiple testing correction
        correction_results = multiple_testing_correction(experiment_analyses)
        
        # 5. Experiment Recommendations
        def generate_recommendations(experiment_analyses, correction_results):
            recommendations = []
            
            for exp_name, analysis in experiment_analyses.items():
                stat_result = analysis['statistical_test']
                business_result = analysis['business_impact']
                
                # Check multiple testing correction
                corrected_significant = correction_results[correction_results['experiment'] == exp_name]['fdr_significant'].iloc[0]
                
                recommendation = {
                    'experiment': exp_name,
                    'recommendation': '',
                    'confidence': 'high',
                    'reasoning': []
                }
                
                if corrected_significant and stat_result['statistical_power'] > 0.8:
                    if stat_result['effect_size'] > 0:
                        recommendation['recommendation'] = 'SHIP'
                        recommendation['reasoning'].append(f"Statistically significant positive effect ({stat_result['relative_lift']:.1%} lift)")
                    else:
                        recommendation['recommendation'] = 'DO NOT SHIP'
                        recommendation['reasoning'].append(f"Statistically significant negative effect ({stat_result['relative_lift']:.1%} decline)")
                
                elif not corrected_significant:
                    if stat_result['statistical_power'] < 0.8:
                        recommendation['recommendation'] = 'EXTEND TEST'
                        recommendation['confidence'] = 'medium'
                        recommendation['reasoning'].append(f"Insufficient power ({stat_result['statistical_power']:.2f} < 0.8)")
                    else:
                        recommendation['recommendation'] = 'NO EFFECT'
                        recommendation['reasoning'].append("Well-powered test shows no significant effect")
                
                # Business impact considerations
                if business_result['roi'] and business_result['roi'] > 2:
                    recommendation['reasoning'].append(f"High ROI potential ({business_result['roi']:.1f}x)")
                elif business_result['roi'] and business_result['roi'] < 0.5:
                    recommendation['reasoning'].append(f"Low ROI ({business_result['roi']:.1f}x)")
                    if recommendation['recommendation'] == 'SHIP':
                        recommendation['recommendation'] = 'EVALUATE COSTS'
                        recommendation['confidence'] = 'medium'
                
                recommendations.append(recommendation)
            
            return pd.DataFrame(recommendations)
        
        recommendations = generate_recommendations(experiment_analyses, correction_results)
        
        return {
            'experiment_analyses': experiment_analyses,
            'multiple_testing_correction': correction_results,
            'recommendations': recommendations
        }
    
    results = comprehensive_ab_testing_analysis(experiment_results, experiments)
    
    # Business insights and reporting
    print("=== A/B TESTING PLATFORM RESULTS ===")
    print(f"Analyzed {len(experiments)} concurrent experiments with {len(experiment_results)} total observations")
    
    print(f"\nEXPERIMENT RESULTS SUMMARY:")
    for exp_name, analysis in results['experiment_analyses'].items():
        stat_result = analysis['statistical_test']
        business_result = analysis['business_impact']
        
        print(f"\n{exp_name.upper().replace('_', ' ')}:")
        print(f"  Effect Size: {stat_result['relative_lift']:+.2%} ({stat_result['effect_size']:+.4f})")
        print(f"  P-value: {stat_result['p_value']:.4f}")
        print(f"  Statistical Power: {stat_result['statistical_power']:.3f}")
        print(f"  Sample Size: {analysis['sample_size']:,}")
        
        if business_result['projected_annual_revenue_impact']:
            print(f"  Projected Annual Revenue Impact: ${business_result['projected_annual_revenue_impact']:,.0f}")
            print(f"  ROI: {business_result['roi']:.1f}x")
    
    print(f"\nMULTIPLE TESTING CORRECTION:")
    significant_after_correction = results['multiple_testing_correction']['fdr_significant'].sum()
    print(f"  Significant experiments after FDR correction: {significant_after_correction}/{len(experiments)}")
    
    print(f"\nRECOMMENDATIONS:")
    for _, rec in results['recommendations'].iterrows():
        print(f"  {rec['experiment'].replace('_', ' ').title()}: {rec['recommendation']} ({rec['confidence']} confidence)")
        for reason in rec['reasoning']:
            print(f"    - {reason}")
    
    # Advanced insights
    print(f"\nADVANCED INSIGHTS:")
    
    # Identify best performing segments
    for exp_name, analysis in results['experiment_analyses'].items():
        best_segments = []
        for segment_type, segment_data in analysis['segmentation'].items():
            if len(segment_data) > 0:
                best_segment = segment_data.loc[segment_data['lift'].idxmax()]
                if best_segment['p_value'] < 0.05:
                    best_segments.append(f"{segment_type}={best_segment['segment_value']} (+{best_segment['lift']:.3f})")
        
        if best_segments:
            print(f"  {exp_name}: Best performing segments: {', '.join(best_segments[:2])}")
    
    return results

ab_testing_results = ab_testing_platform()
```

**Time Complexity:** O(n*k) for segmentation analysis, O(n) for statistical tests
**Space Complexity:** O(n*s) where s is number of segments

---

### Key Advanced Problem-Solving Patterns:

**1. Multi-Layered Analysis Approach:**
- Start with exploratory data analysis
- Build statistical models and tests
- Apply business context and constraints
- Generate actionable recommendations

**2. Real-World Complexity Handling:**
- Multiple concurrent experiments
- Segmentation and heterogeneous effects
- Multiple testing corrections
- Power analysis and sample size planning

**3. Business Integration:**
- Connect statistical results to business metrics
- Cost-benefit analysis
- Risk assessment and mitigation
- Implementation feasibility

**4. Advanced Statistical Techniques:**
- Proper experimental design
- Causal inference methods
- Bayesian approaches where appropriate
- Effect size estimation and confidence intervals

### Interview Success Strategies for Advanced Problems:

**1. Structure Your Approach:**
- Clearly outline your analysis plan
- Explain the business context first
- Break complex problems into manageable pieces
- Show how different analyses connect

**2. Demonstrate Statistical Rigor:**
- Choose appropriate statistical tests
- Address assumptions and limitations
- Handle multiple comparisons properly
- Discuss power and sample size considerations

**3. Business Acumen:**
- Connect technical results to business value
- Consider implementation constraints
- Provide clear recommendations
- Quantify impact where possible

**4. Code Quality:**
- Write modular, reusable functions
- Use clear variable names and comments
- Handle edge cases and missing data
- Optimize for both readability and performance

These advanced problems simulate the complexity you'll encounter in senior data science roles, where you need to combine technical expertise with business judgment to drive real-world decisions.

