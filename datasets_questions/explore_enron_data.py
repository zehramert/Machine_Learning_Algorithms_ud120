#!/usr/bin/python3

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import joblib

enron_data = joblib.load(open("../final_project/final_project_dataset.pkl", "rb"))


#Size of the dataset
size_of_dataset = len(enron_data)
print(f"Number of dataset: {size_of_dataset} ")

#Features
first_key = list(enron_data.keys())[0]  # gives all the keys and converts them into a list
num_of_features = len(enron_data[first_key])

print(f"Number of features available for each person: {num_of_features}")

#Person of interest num

poi_num = sum(1 for key in enron_data if enron_data[key]["poi"]==1)

#keys = enron_data.keys()
#for key in keys:
    #if(enron_data[key]["poi"]==1):
        #poi_num +=1

print(f"Number of person of interest: {poi_num}")


# Total POI using names file

with open("../final_project/poi_names.txt", "r") as name_data:
    total_poi_num = sum(1 for line in name_data if line.startswith("(y)") or line.startswith("(n)")) # Avoid counting title and empty line


print(f"Total number of POIs: {total_poi_num}")


#######
print(f"Total value of stock belonging to James Prentice:{enron_data['PRENTICE JAMES']['total_stock_value']}")

print(f"Total number of email messages  from Wesley Colwell to persons of interest: {enron_data['COLWELL WESLEY']['from_this_person_to_poi']}")

print(f"Total number of stock options of Jeffrey K Skilling: {enron_data['SKILLING JEFFREY K']['exercised_stock_options']}")


#Follow the Money

# Extract total payments for each individual
total_payments = {
    "Kenneth Lay": enron_data["LAY KENNETH L"]["total_payments"],
    "Jeffrey Skilling": enron_data["SKILLING JEFFREY K"]["total_payments"],
    "Andrew Fastow": enron_data["FASTOW ANDREW S"]["total_payments"]
}

# Find the individual with the maximum total payment
max_p = max(total_payments, key=total_payments.get)
max_payment = total_payments[max_p]
print(f"Largest total payment belongs to {max_p} with the value of {max_payment}")


#Dealing with Unfilled Features
quantified_salary_count = 0
known_email_address_count = 0

for person in enron_data:
    if enron_data[person]["salary"] != "NaN":
        quantified_salary_count += 1
    if enron_data[person]["email_address"] != "NaN":
        known_email_address_count += 1

print(f"Number of people with quantified salary: {quantified_salary_count}")
print(f"Number of people with known email address: {known_email_address_count}")

#######
count_of_people=0
for person in enron_data:
    count_of_people+=1

empty_payment_count=0
for person in enron_data:
    if enron_data[person]["total_payments"] == "NaN":
       empty_payment_count += 1
percentage = (empty_payment_count/count_of_people) *100

print(f"Empyt total payment value percentage is {percentage}")


count_of_pois=0
for person, features in enron_data.items():
    if features.get("poi"):
        count_of_pois+=1
empty_payment_count=0
for person in enron_data:
    if enron_data[person]["total_payments"] == "NaN" and enron_data[person]["poi"] != False:
       empty_payment_count += 1
percentage = (empty_payment_count/count_of_pois) *100

print(f"Empyt total payment value percentage is {percentage}")










