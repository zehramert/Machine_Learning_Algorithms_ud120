#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """

    cleaned_data = []
    error = abs(predictions-net_worths)
    error_list = list(zip(ages,net_worths, error))  #zip function combines multiple feature into a single tuple iteration

    error_list.sort( key=lambda x : x[2])

    reminder_num = int(len(error_list)* 0.9)

    cleaned_data = error_list [:reminder_num]





    
    return cleaned_data

