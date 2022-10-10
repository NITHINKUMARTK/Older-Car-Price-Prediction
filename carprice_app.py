import numpy as np
import pickle
import pandas as pd
import streamlit as st 
import xgboost as xgb


from PIL import Image
image = Image.open('carimage.jpg')
st.image(image,width = 600)
st.write("")
st.write("")

from datetime import date
todays_date = date.today()
present_year = todays_date.year


pickle_in = open("carprice_encoding.pkl","rb")
encoder = pickle.load(pickle_in)

pickle_in = open("carprice_scaling.pkl","rb")
scaling_transformer = pickle.load(pickle_in)

pickle_in = open("carprice_predict_sc.pkl","rb")
sc = pickle.load(pickle_in)

#bst = xgb.Booster({"nthread":4})
bst = xgb.Booster({"nthread":4})
bst.load_model("carprice.model")


st.title("OLDER CAR PRICE PREDICTION OF INDIAN CARS")

option = st.selectbox(
     'Please select the brand',
     ('Maruti', 'Hyundai', 'Ford', 'Renault', 'Mini', 'Mercedes-Benz',
       'Toyota', 'Volkswagen', 'Honda', 'Mahindra', 'Datsun', 'Tata',
       'Kia', 'BMW', 'Audi', 'Land Rover', 'Jaguar', 'MG', 'Isuzu',
       'Porsche', 'Skoda', 'Volvo', 'Lexus', 'Jeep', 'Maserati',
       'Bentley', 'Nissan', 'Ferrari', 'Mercedes-AMG','Rolls-Royce', 'Force'))


if option == "Maruti":
	option2 = st.selectbox(
     'Please select the model',
     ('Maruti Alto', 'Maruti Wagon R', 'Maruti Swift', 'Maruti Ciaz',
       'Maruti Baleno', 'Maruti Swift Dzire', 'Maruti Ignis',
       'Maruti Vitara', 'Maruti Celerio', 'Maruti Ertiga', 'Maruti Eeco',
       'Maruti Dzire VXI', 'Maruti XL6', 'Maruti S-Presso',
       'Maruti Dzire LXI', 'Maruti Dzire ZXI'))

elif option == "Hyundai":
	option2 = st.selectbox(
     'Please select the model',
     ('Hyundai Grand', 'Hyundai i20', 'Hyundai i10', 'Hyundai Venue',
       'Hyundai Verna', 'Hyundai Creta', 'Hyundai Santro',
       'Hyundai Elantra', 'Hyundai Aura', 'Hyundai Tucson'))

elif option == "Ford":
	option2 = st.selectbox(
     'Please select the model',
     ('Ford Ecosport', 'Ford Aspire', 'Ford Figo', 'Ford Endeavour',
       'Ford Freestyle'))

elif option == "Renault":
	option2 = st.selectbox(
     'Please select the model',
     ('Renault Duster', 'Renault KWID', 'Renault Triber'))

elif option == "Mini":
	option2 = st.selectbox(
     'Please select the model',
     ('Mini Cooper',"-"))

elif option == 'Mercedes-Benz':
	option2 = st.selectbox(
     'Please select the model',
     ('Mercedes-Benz C-Class', 'Mercedes-Benz E-Class',
       'Mercedes-Benz GL-Class', 'Mercedes-Benz S-Class',
       'Mercedes-Benz CLS', 'Mercedes-Benz GLS'))

elif option == 'Toyota':
	option2 = st.selectbox(
     'Please select the model',
     ('Toyota Innova', 'Toyota Fortuner', 'Toyota Camry', 'Toyota Yaris',
       'Toyota Glanza'))

elif option == 'Volkswagen':
	option2 = st.selectbox(
     'Please select the model',
     ('Volkswagen Vento', 'Volkswagen Polo'))

elif option == 'Honda':
	option2 = st.selectbox(
     'Please select the model',
     ('Honda City', 'Honda Amaze', 'Honda CR-V', 'Honda Jazz',
       'Honda Civic', 'Honda WR-V', 'Honda CR'))

elif option == 'Mahindra':
	option2 = st.selectbox(
     'Please select the model',
     ('Mahindra Bolero', 'Mahindra XUV500', 'Mahindra KUV100',
       'Mahindra Scorpio', 'Mahindra Marazzo', 'Mahindra KUV',
       'Mahindra Thar', 'Mahindra XUV300', 'Mahindra Alturas'))

elif option == 'Datsun':
	option2 = st.selectbox(
     'Please select the model',
     ('Datsun RediGO', 'Datsun GO', 'Datsun redi-GO'))

elif option == 'Tata':
	option2 = st.selectbox(
     'Please select the model',
     ('Tata Tiago', 'Tata Tigor', 'Tata Safari', 'Tata Hexa',
       'Tata Nexon', 'Tata Harrier', 'Tata Altroz'))

elif option == 'Kia':
	option2 = st.selectbox(
     'Please select the model',
     ('Kia Seltos', 'Kia Carnival'))

elif option == 'BMW':
	option2 = st.selectbox(
     'Please select the model',
     ('BMW 5', 'BMW 3', 'BMW Z4', 'BMW 6', 'BMW X5', 'BMW X1', 'BMW 7',
       'BMW X3', 'BMW X4'))

elif option == 'Audi':
	option2 = st.selectbox(
     'Please select the model',
     ('Audi A4', 'Audi A6', 'Audi Q7', 'Audi A8'))

elif option == 'Land Rover':
	option2 = st.selectbox(
     'Please select the model',
     ('Land Rover Rover',"-"))

elif option == 'Jaguar':
	option2 = st.selectbox(
     'Please select the model',
     ('Jaguar XF', 'Jaguar F-PACE', 'Jaguar XE'))

elif option == 'MG':
	option2 = st.selectbox(
     'Please select the model',
     ('MG Hector',"-"))

elif option == 'Isuzu':
	option2 = st.selectbox(
     'Please select the model',
     ('Isuzu D-Max', 'Isuzu MUX'))

elif option == 'Porsche':
	option2 = st.selectbox(
     'Please select the model',
     ('Porsche Cayenne', 'Porsche Macan', 'Porsche Panamera'))

elif option == 'Skoda':
	option2 = st.selectbox(
     'Please select the model',
     ('Skoda Rapid', 'Skoda Superb', 'Skoda Octavia'))

elif option == 'Volvo':
	option2 = st.selectbox(
     'Please select the model',
     ('Volvo S90', 'Volvo XC', 'Volvo XC90', 'Volvo XC60'))

elif option == 'Lexus':
	option2 = st.selectbox(
     'Please select the model',
     ('Lexus ES', 'Lexus NX', 'Lexus RX'))

elif option == 'Jeep':
		option2 = st.selectbox(
     'Please select the model',
     ('Jeep Wrangler', 'Jeep Compass'))

elif option == 'Maserati':
	option2 = st.selectbox(
     'Please select the model',
     ('Maserati Ghibli', 'Maserati Quattroporte'))

elif option == 'Bentley':
	option2 = st.selectbox(
     'Please select the model',
     ('Bentley Continental',"-"))

elif option == 'Nissan':
	option2 = st.selectbox(
     'Please select the model',
     ('Nissan Kicks', 'Nissan X-Trail'))

elif option == 'Ferrari':
	option2 = st.selectbox(
     'Please select the model',
     ('Ferrari GTC4Lusso',"-"))

elif option == 'Mercedes-AMG':
	option2 = st.selectbox(
     'Please select the model',
     ('Mercedes-AMG C',"-"))

elif option == 'Rolls-Royce':
	option2 = st.selectbox(
     'Please select the model',
     ('Rolls-Royce Ghost',"-"))

elif option == 'Force':
	option2 = st.selectbox(
     'Please select the model',
     ('Force Gurkha',"-"))


option3 = st.selectbox(
     'Please select the Seller Type',
     ('Individual', 'Dealer', 'Trustmark Dealer'))

option4 = st.selectbox(
     'Please select the Fuel Type',
     ('Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'))

option5 = st.selectbox(
     'Please select the Transmission Type',
     ('Manual', 'Automatic'))

option6 = st.selectbox(
     'Please select the Number of Seats',
     (1,2,3,4,5,6,7,8,9,10))

cost_price = st.number_input("Enter the purchase value/amount in Rs.")
purchase_year = st.number_input("Enter the year of  purchase")
km_driven = st.number_input("Enter km driven")

car_name = option
brand = option2


cost = cost_price
if cost < 0:
  st.write("Please enter a positive value for max_cost_price")
elif:
   max_cost_price = cost
 


 year = purchase_year
if year < 0:
  st.write("Please enter a positive value for year")
elif:
  year_of_purchase = year
  vehicle_age = present_year - year_of_purchase
  

km = km_driven
if km <0 :
  st.write("Please enter a positive value for km_driven")
elif:
  km_driven  = km
  


seller_type = option3
fuel_type = option4
transmission_type = option5
seats = option6

if   max_cost_price >=0 and year_of_purchase >=0 and km_driven>0:
  input_data = {"car_name":[car_name],"brand": [brand], "max_cost_price":[max_cost_price],
              "vehicle_age": [vehicle_age],"km_driven":[km_driven], "seller_type":[seller_type],
              "fuel_type":[fuel_type],"transmission_type":[transmission_type], "seats":[seats]}

  df = pd.DataFrame(input_data)
  df= encoder.transform(df)
  df = scaling_transformer.transform(df)
  dpred = xgb.DMatrix(df)
  if st.button("Predict"):
    y = bst.predict(dpred)
    y = sc.inverse_transform([y])
    answer = int(y.ravel().ravel())
    st.success("The predicted selling price of the car is Rs.{}".format(answer))




  
  
    

    
    
    



  
  
  
  
  
  
  
































 












