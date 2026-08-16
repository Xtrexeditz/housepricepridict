import os
import argparse
import pandas as pd
import joblib

def load_pipeline():
    pipeline_path = 'models/house_price_pipeline.joblib'
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Model pipeline not found at {pipeline_path}. Please train the model first by running train.py.")
    return joblib.load(pipeline_path)

def predict_single_house(pipeline, features):
    # Convert features dict to DataFrame
    df = pd.DataFrame([features])
    prediction = pipeline.predict(df)[0]
    return prediction

def interactive_mode(pipeline):
    print("\n===========================================")
    print("      Property Price Prediction - Interactive CLI     ")
    print("===========================================")
    print("Please enter the details of the property:")
    
    try:
        print("Property Type:")
        print("  1. Apartment")
        print("  2. Independent House")
        print("  3. Villa")
        print("  4. Penthouse")
        print("  5. Agricultural Land (Field/Plot)")
        pt_choice = input("Select property type (1-5): ")
        property_type = 'Apartment'
        if pt_choice == '2':
            property_type = 'Independent House'
        elif pt_choice == '3':
            property_type = 'Villa'
        elif pt_choice == '4':
            property_type = 'Penthouse'
        elif pt_choice == '5':
            property_type = 'Agricultural Land'
            
        # If property is land, bedrooms/bathrooms/floors/garage don't apply
        if property_type == 'Agricultural Land':
            acres = float(input("Land Area in Acres (e.g. 1.5): "))
            sqft = acres * 43560.0
            bedrooms = 0
            bathrooms = 0.0
            year_built = 1950
            floors = 0
            garage = 0
            print(f"Note: Converted {acres} Acres to {sqft:,.1f} sq ft. Building details are set to 0.")
        else:
            sqft = float(input("Square Footage (e.g. 1500): "))
            bedrooms = int(input("Number of Bedrooms (e.g. 3): "))
            bathrooms = float(input("Number of Bathrooms (e.g. 2.5): "))
            year_built = int(input("Year Built (e.g. 2005): "))
            floors = int(input("Number of Floors (1-3): "))
            garage_choice = input("Has garage? (y/n): ").lower()
            garage = 1 if garage_choice.startswith('y') else 0
            
        print("Neighborhood:")
        print("  1. Downtown")
        print("  2. Suburban")
        print("  3. Rural")
        nh_choice = input("Select neighborhood (1-3): ")
        neighborhood = 'Suburban'
        if nh_choice == '1':
            neighborhood = 'Downtown'
        elif nh_choice == '3':
            neighborhood = 'Rural'
            
        features = {
            'SquareFeet': sqft,
            'PropertyType': property_type,
            'Bedrooms': bedrooms,
            'Bathrooms': bathrooms,
            'Neighborhood': neighborhood,
            'YearBuilt': year_built,
            'Floors': floors,
            'Garage': garage
        }
        
        price = predict_single_house(pipeline, features)
        print("\n===========================================")
        print(f" Estimated Property Price: Rs. {price:,.2f}")
        print("===========================================")
        
    except ValueError as e:
        print(f"\nError: Invalid input format. {e}. Please try again.")

def main():
    parser = argparse.ArgumentParser(description="Property Price Prediction CLI")
    parser.add_argument('--sqft', type=float, help='Square footage of the property')
    parser.add_argument('--acres', type=float, help='Acreage if property type is Agricultural Land')
    parser.add_argument('--property-type', type=str, choices=['Apartment', 'Independent House', 'Villa', 'Penthouse', 'Agricultural Land'], help='Property type')
    parser.add_argument('--bedrooms', type=int, help='Number of bedrooms')
    parser.add_argument('--bathrooms', type=float, help='Number of bathrooms')
    parser.add_argument('--neighborhood', type=str, choices=['Downtown', 'Suburban', 'Rural'], help='Neighborhood name')
    parser.add_argument('--year-built', type=int, help='Year the property was built')
    parser.add_argument('--floors', type=int, help='Number of floors')
    parser.add_argument('--garage', type=int, choices=[0, 1], help='Garage presence (0 = No, 1 = Yes)')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive input mode')
    
    args = parser.parse_args()
    
    try:
        pipeline = load_pipeline()
    except FileNotFoundError as e:
        print(e)
        return
        
    # Check if we should run in interactive mode (either explicit flag or if no args passed)
    has_args = any(val is not None for val in [args.sqft, args.acres, args.property_type, args.bedrooms, args.bathrooms, args.neighborhood, args.year_built, args.floors, args.garage])
    
    if args.interactive or not has_args:
        interactive_mode(pipeline)
    else:
        property_type = args.property_type if args.property_type is not None else 'Apartment'
        
        # Override defaults if Agricultural Land
        if property_type == 'Agricultural Land':
            # Resolve size in square feet (prioritize --acres if provided)
            sqft_value = 43560.0 # Default 1.0 Acre
            if args.acres is not None:
                sqft_value = args.acres * 43560.0
            elif args.sqft is not None:
                sqft_value = args.sqft
                
            features = {
                'SquareFeet': sqft_value,
                'PropertyType': 'Agricultural Land',
                'Bedrooms': 0,
                'Bathrooms': 0.0,
                'Neighborhood': args.neighborhood if args.neighborhood is not None else 'Suburban',
                'YearBuilt': 1950,
                'Floors': 0,
                'Garage': 0
            }
        else:
            features = {
                'SquareFeet': args.sqft if args.sqft is not None else 1500.0,
                'PropertyType': property_type,
                'Bedrooms': args.bedrooms if args.bedrooms is not None else 3,
                'Bathrooms': args.bathrooms if args.bathrooms is not None else 2.0,
                'Neighborhood': args.neighborhood if args.neighborhood is not None else 'Suburban',
                'YearBuilt': args.year_built if args.year_built is not None else 2000,
                'Floors': args.floors if args.floors is not None else 1,
                'Garage': args.garage if args.garage is not None else 1
            }
        
        price = predict_single_house(pipeline, features)
        print(f"Predicted Price: Rs. {price:,.2f}")

if __name__ == '__main__':
    main()
