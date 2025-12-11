# Spark-Housing-Project Midterm Report

# Description

This project is part of Spark’s University Accountability Ordinance Project. Our goal is to better understand where students live and evaluate the housing conditions associated with those addresses. We aim to classify which landlords are problematic by training a model to predict if an address is problematic or not. By integrating data on 311 service requests, building violations, property assessments, neighborhood data, and student housing records from 2018-2024, we hope to find patterns and trends that contribute to student housing conditions in Boston.

# How to Build and Run 
1. **Download Data**: [Link](www.https://drive.google.com/drivefolders/1YtY4Z3qbk1JIEX84lFkByQXLLRl_i1DC?usp=sharing.com)
    - Data File Directory: 
      - Boston_Neighborhood_Boundaries.csv
      - neighborhood_sizes.csv
      - building_and_property_violations.csv 
      - live_street_address_management_sam_addresses.csv
      - 311_calls/
        - 2018.csv
        - 2019.csv
        - 2020.csv
        - 2021.csv
        - 2022.csv
        - 2023.csv
        - 2024.csv
      - property_assessment/ 
        - 2018.csv
        - 2019.csv
        - 2020.csv
        - 2021.csv
        - 2022.csv
        - 2023.csv
        - 2024.csv

2. **Run**: `make -f MakeFile all`  

# Data Processing 
- **Data Collection**: We collected data from 2018-2024. Data include [311 calls](https://data.boston.gov/dataset/311-service-requests), known [building and property violations](https://data.boston.gov/dataset/building-and-property-violations1/resource/800a2663-1d6a-46e7-9356-bedb70f5332c), avaliable [building and property assessments](https://data.boston.gov/dataset/property-assessment), [student housing data](https://docs.google.com/spreadsheets/d/11X4VvywkSodvvTk5kkQH7gtNPGovCgBq/edit?usp=drive_link&ouid=107346197263951251461&rtpof=true&sd=true), [SAM addresses](https://data.boston.gov/dataset/live-street-address-management-sam-addresses), and [neighborhood shape files](https://data.boston.gov/dataset/boston-neighborhood-boundaries-approximated-by-2020-census-tracts). 
- **Data Cleaning**: 
  1. We filtered databases to only include data from the years 2018-2024. 
  2. Standardized columns to have the same types and format.
  3. Removed irrelevant columns.  
  4. Combined our data by matching properties by their addresses and the year that the data was sampled from. Now, we have all instances of cases made for each address in their respective years. 

# Modeling 
- **Feature Extraction**: 
  - Filled in NaNs 
  - Standardized Values
  - Text Features: length and word counts of case titles, case reasons, and case type descriptions. 
  - Time Based Features: Dates that cases were opened, closed, and length of active cases 
  - Cyclical Encoding for Month and Day 
  - Number of cases per address per year 
  - VADAR: Sentiment Intensity Analysis to score severity of cases 
  - Frequency of cases 
  - Escalation of severity 
  - Rate of cases being resolved before their deadlines 
  - Scoring of overall, internal, and external conditions 
  - TF-IDF: On case reason, and case title 
  - Truncated SVD on the TF-IDF Features 

- **Model**: 
  - We upsampled because our classes were severely imbalanced. 
  - We trained a Logistic Regression model because we only had two class labels (problematic / not problematic)

# Visualizations / Results 

# Video 