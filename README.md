# Operation Soar
- Steps to reproduce are located at the bottom of this ReadMe.
# Project planning 
## Wrangle
### Acquire:
- Data was acquired from Kaggle
    - https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction?resource=download&select=train.csv
    - https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction?resource=download&select=test.csv
- 1st data set (train) consisted of 103,904 rows and 25 columns
- 2nd data set (test) consisted of 25,976 rows and 25 columns
- Each row represents 1 customer survey
- Each column respresents a unique question or data point
### Prepare:
- Combined the 2 data sets which resulted in 1 data frame consisting of 129,880 rows and 25 columns
- **Nulls:** Dropped 393 nulls
- Replaced special characters in column names with '_' and removed spaces
- Dropped 1 column named 'unnamed:_0'
- Renamed 'class' to 'customer_class' for exploration
- Created dummy columns for columns with strings
- Data frame consisted of 129,487 row and 30 columns
- Split data into train, validate, test
## Data Dictionary
| Feature | Definition |
|:--------|:-----------|
|ID|ID associated with the passenger|
|customer type|Loyal or disloyal passenger|
|age|Age of the passenger who answered the survey|
|type_of_travel|Business or personal reason for travel|
|customer_class|Business, Eco Plus, or Eco seat selection|
|flight_distance|The distance each passenger travels to their destination|
|Survey Questions|(0:Not Applicable;1-5) 1 is the lowest rating for satisfaction and 5 is the highest|
|inflight_wifi_service|Satisfaction level of the inflight wifi service|
|departure_arrival_time_convenient|Satisfaction level of Departure/Arrival time convenient|
|ease_of_online_booking| Satisfaction level of online booking|
|gate_location|Satisfaction level of Gate location|
|food_and_drink|Satisfaction level of Food and drink|
|online_boarding|Satisfaction level of online boarding|
|seat_comfort|Satisfaction level of Seat comfort|
|inflight_entertainment|Satisfaction level of inflight entertainment|
|on_board_service|Satisfaction level of On-board service|
|leg_room_service|Satisfaction level of Leg room|
|baggage_handling|Satisfaction level of baggage handling|
|checkin_service|Satisfaction level of Check-in service|
|inflight_service|Satisfaction level of inflight service|
|cleanliness|Satisfaction level of Cleanliness|
|departure_delay_in_minutes|Minutes delayed for departure|
|arrival_delay_in_minutes|Minutes delayed for Arrival|
|satisfaction(Target Variable)|Airline satisfaction level(Satisfaction, neutral or dissatisfaction)|
## Exploration
- This phase was used to search for key drivers in passenger satisfaction.
- After some initial exploration I asked 5 question to gain additional clarity of the data
    - Does flight duration impact passenger satisfaction?
    - Does gender impact passenger satisfaction?
    - Does customer type (loyal vs disloyal) impact type of travel (business vs personal)?
    - Does type of travel impact passenger satisfaction?
    - Does customer loyalty impact satisfaction?
### Key findings:
- Passengers with shorter flights are dissatisfied at higher rates than customers who have longer flights
- Females are slightly more dissatisfied than males
- Loyal passengers make up 99.5% of the personal travel category
- 90% of loyal passengers who travel for personal reasons are dissatisfied
- Overall, 48% of loyal passengers are satisfied and 24% of disloyal passengers are satisfied
    - Industry average according to a J.D. Power survey fluctuates over the years but is around 79%
## Modeling
- 4 ML algorithms were used during this phase:
    - KNN, Decision Tree, Random Forest, and Logistical Regression
- Multiple iterations were ran for each model. The best model for each algorithm was selected to compare against the rest.
- Evaluation metric: **Accuracy**
- All features were used in the modeling phase with exception to:
    - id
        - Not needed for modeling.
    - gate location
        - Gate locations are typically part of airline contracts with airports. I decided to not use that feature since airlines have minimal control or flexiblility to change gate locations.
    - leg room service
        - Since it isn't feasible to alter legroom for entire fleets, I decided to not use that feature in modeling.
### Modeling Summary:
- Baseline accuracy was 56.55% and all models performed well above.
- Logistic regression had the lowest performance at 87%.
- Best model was decision tree with a max depth of 12, which performed 38% above baseline.
- Used decision tree model on test and the result was 94.57% accuracy which was also 38% above baseline.
    - I would be confident using this model in production to predict passenger satisfaction for airline companies.
## Recommendation
- Prioritize loyal customers when developing a strategy to increase passenger satisfaction
    - Specifically loyal customers who travel for personal reasons
    - This segment accounts for nearly 19% of the total passenger population
    - This segment also accounts for nearly 50% of the toal of neutral or dissatisfied passengers
    - If we convert 50% of dissatisfied to satisfied passengers, the overall satisfaction of loyal customers would increase from 48% to 64%
## Next Steps
- Explore how the business can convert disloyal business travelers to loyal business travelers with incentives
    - Then explore how to convert that segment to loyal personal travelers
- Consider altering the passenger survey to ask some "level of importance" questions
    - This could give the business more insights on what the customer wants and needs are
    - i.e. On a scale 1 - 5, how satisfied were you with inflight entertainment? By selecting important or not important, how do you rate the level of importance?
## Steps to Reproduce
- Clone this repo or download all .py files and final.ipynb
- Acquire both datasets from:
    - **Note:** You need to be logged into your Kaggle account in order to download csv files.
        - Click the black "download" button at the top right of the page
        - It should allow you to download both csv's at the same time
            - Once downloaded, move both csv's to the folder/directory you are going to run the notebook in
    - https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction?resource=download&select=train.csv
    - https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction?resource=download&select=test.csv
- Prepare data using functions in wrangle.py file:
    - get_airline_data() - this function combines both csv files after you insert them into your local directory
    - prep_airline(df)
- Split data using function in wrangle.py file:
    - split_function(df, 'satisfaction')
        - Stratify on satisfaction
- Explore data utilizing visualizations and stats tests location in explore.py file
- Prepare data for modeling and run modeling functions located in model.py file
    - Xy_train_val_test(train, validate, test, 'satisfaction_satisfied')
    - scaled_df(X_train, X_validate, X_test)
    - k_nearest2(X_train_scaled, y_train, X_validate_scaled, y_validate)
    - decision_tree(X_train_scaled, X_validate_scaled, y_train, y_validate)
    - random_forest_scores(X_train_scaled, y_train, X_validate_scaled, y_validate)
    - plot_logistic_regression(X_train_scaled, X_validate_scaled, y_train, y_validate)
    - best_model(X_train_scaled, X_test_scaled, y_train, y_test)
# Thank you for stopping by and reading through this project description! 