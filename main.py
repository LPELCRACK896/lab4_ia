import numpy as np

def main():
    # Task 1.1
    headers = ["id", "date", "price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view", "condition", "grade", "sqft_above", "sqft_basement", "yr_built", "yr_renovated", "zipcode", "lat", "long", "sqft_living15", "sqft_lot15"]
    data = np.genfromtxt('./data/kc_house_data.csv', delimiter=',', skip_header=1)

if __name__ == "__main__":
    main()