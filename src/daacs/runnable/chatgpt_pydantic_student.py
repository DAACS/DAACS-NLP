from daacs.infrastructure.estimate_service import EstimateService
from daacs.infrastructure.bootstrap import Bootstrap

es = EstimateService()
b = Bootstrap()

essays_and_grades = b.get_essays_and_grades() 

# set the range of daacs id's to query.

low_boundary = 10
high_boundary = 20

filtered_essays = essays_and_grades[(essays_and_grades.index > low_boundary) & (essays_and_grades.index < high_boundary)]

# Define the function to apply estimates
def apply_estimates(row):
    estimates = es.run(row['essay'])
    for key, value in estimates.items():
        row[key] = value
    return row

# Apply the function to each row of the filtered DataFrame
filtered_essays = filtered_essays.apply(apply_estimates, axis=1)

print(filtered_essays.drop(columns=['essay', 'file_name']))



