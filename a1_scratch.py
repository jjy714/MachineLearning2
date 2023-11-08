
# P10 File Clock angle calculation

# Variable initiation
time = input("Enter the time in the format of HH:MM: ")
hour = int(time.split(':')[0])
minute = int(time.split(':')[1])

# Conditions
if hour > 12:
    hour = hour - 12

# functions for degree
def degree(hour,  minute):
    result = abs(30 * hour - 5.5 * minute)
    if result > 180:
        result = 360 - result
    return result


# Print the result
for i in range(5):
    print('{0}:{1} makes {2:0.2f} degrees'.format(hour, minute, degree(hour, minute)))
    minute = minute + 15
    if minute > 60:
        minute = minute - 60
        hour = hour + 1
