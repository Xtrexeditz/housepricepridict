import os
import numpy as np
import pandas as pd

def generate_housing_data(num_samples=5000, seed=42):
    np.random.seed(seed)
    
    print(f"Generating {num_samples} synthetic housing & agricultural land (field) records...")
    
    # 1. Property Type (Apartment, Independent House, Villa, Penthouse, Agricultural Land)
    property_types = np.random.choice(['Apartment', 'Independent House', 'Villa', 'Penthouse', 'Agricultural Land'], 
                                      size=num_samples, p=[0.35, 0.25, 0.15, 0.10, 0.15])
    
    # 2. Square Footage (based on property type)
    # Agricultural lands are much larger (measured in Acres: 1 Acre = 43,560 sq ft)
    square_feet = []
    for pt in property_types:
        if pt == 'Apartment':
            square_feet.append(np.random.randint(600, 1800))
        elif pt == 'Penthouse':
            square_feet.append(np.random.randint(1500, 3500))
        elif pt == 'Independent House':
            square_feet.append(np.random.randint(1000, 3500))
        elif pt == 'Villa':
            square_feet.append(np.random.randint(1800, 5000))
        else: # Agricultural Land (Field) - between 0.2 Acres (8712 sqft) and 5.0 Acres (217800 sqft)
            acres = np.random.uniform(0.2, 5.0)
            square_feet.append(int(acres * 43560))
    square_feet = np.array(square_feet)
    
    # 3. Bedrooms (based on property type, 0 for Agricultural Land)
    bedrooms = []
    for pt in property_types:
        if pt == 'Apartment':
            bedrooms.append(np.random.randint(1, 4))
        elif pt == 'Penthouse':
            bedrooms.append(np.random.randint(3, 5))
        elif pt == 'Independent House':
            bedrooms.append(np.random.randint(2, 5))
        elif pt == 'Villa':
            bedrooms.append(np.random.randint(3, 6))
        else: # Agricultural Land
            bedrooms.append(0)
    bedrooms = np.array(bedrooms)
    
    # 4. Bathrooms (based on property type, 0.0 for Agricultural Land)
    bathrooms = []
    for pt in property_types:
        if pt == 'Agricultural Land':
            bathrooms.append(0.0)
        else:
            bathrooms.append(np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], 
                                             p=[0.15, 0.20, 0.25, 0.20, 0.10, 0.06, 0.04]))
    bathrooms = np.array(bathrooms)
    
    # 5. Neighborhoods (Downtown, Suburban, Rural)
    neighborhoods = np.random.choice(['Downtown', 'Suburban', 'Rural'], size=num_samples, p=[0.3, 0.5, 0.2])
    
    # 6. Year Built (1950 to 2025, 1950 baseline for Farmland)
    year_built = []
    for pt in property_types:
        if pt == 'Agricultural Land':
            year_built.append(1950)
        else:
            year_built.append(np.random.randint(1950, 2026))
    year_built = np.array(year_built)
    
    # 7. Floors (1 to 3, 0 for Agricultural Land)
    floors = []
    for pt in property_types:
        if pt == 'Agricultural Land':
            floors.append(0)
        else:
            floors.append(np.random.randint(1, 4))
    floors = np.array(floors)
    
    # 8. Garage (0 = No, 1 = Yes, 0 for Agricultural Land)
    garage = []
    for pt in property_types:
        if pt == 'Agricultural Land':
            garage.append(0)
        else:
            garage.append(np.random.choice([0, 1], p=[0.3, 0.7]))
    garage = np.array(garage)
    
    # Create DataFrame
    df = pd.DataFrame({
        'SquareFeet': square_feet,
        'PropertyType': property_types,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Neighborhood': neighborhoods,
        'YearBuilt': year_built,
        'Floors': floors,
        'Garage': garage
    })
    
    # Calculate price in Indian Rupees (INR)
    base_price = 1500000
    bed_coeff = 350000
    bath_coeff = 250000
    floor_coeff = 200000
    garage_coeff = 300000
    
    prices = []
    for i in range(num_samples):
        pt = df.iloc[i]['PropertyType']
        sqft = df.iloc[i]['SquareFeet']
        beds = df.iloc[i]['Bedrooms']
        baths = df.iloc[i]['Bathrooms']
        nb = df.iloc[i]['Neighborhood']
        yb = df.iloc[i]['YearBuilt']
        fl = df.iloc[i]['Floors']
        gr = df.iloc[i]['Garage']
        
        # Property type specific rates
        if pt == 'Apartment':
            pt_base = 0
            sqft_rate = 4800
            age_val = (yb - 1950) * 12000
            nb_adds = {'Downtown': 2000000, 'Suburban': 800000, 'Rural': -1200000}
        elif pt == 'Independent House':
            pt_base = 1500000
            sqft_rate = 5500
            age_val = (yb - 1950) * 12000
            nb_adds = {'Downtown': 2000000, 'Suburban': 800000, 'Rural': -1200000}
        elif pt == 'Penthouse':
            pt_base = 3000000
            sqft_rate = 6000
            age_val = (yb - 1950) * 12000
            nb_adds = {'Downtown': 2000000, 'Suburban': 800000, 'Rural': -1200000}
        elif pt == 'Villa':
            pt_base = 4000000
            sqft_rate = 6500
            age_val = (yb - 1950) * 12000
            nb_adds = {'Downtown': 2000000, 'Suburban': 800000, 'Rural': -1200000}
        else: # Agricultural Land (Field)
            pt_base = -700000  # Vacant plot baseline
            sqft_rate = 110     # ₹110 per sq ft (approx ₹48 Lakhs per Acre)
            age_val = 0        # Farmland doesn't depreciate with age
            nb_adds = {'Downtown': 3500000, 'Suburban': 1200000, 'Rural': -600000}
            
        nb_val = nb_adds[nb]
        
        # Sum pricing signals
        row_price = (
            base_price +
            pt_base +
            sqft * sqft_rate +
            beds * bed_coeff +
            baths * bath_coeff +
            fl * floor_coeff +
            gr * garage_coeff +
            nb_val +
            age_val
        )
        
        prices.append(row_price)
        
    df['Price'] = np.array(prices)
    
    # Add random noise
    noise = np.random.normal(0, 300000, size=num_samples)
    df['Price'] = (df['Price'] + noise).round(-3)
    
    # Ensure minimum price of ₹500,000 (5 Lakhs) for farmland plots
    df['Price'] = df['Price'].clip(lower=500000)
    
    # Save directory
    os.makedirs('data', exist_ok=True)
    output_path = 'data/housing_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Data successfully generated and saved to {output_path}")
    print(df.head())

if __name__ == '__main__':
    generate_housing_data()
