
list= [0.27, 0.27, 0.27, 0.27, 0.27, 0.27 ]

firstEst = 0.27
# EstimatedRTT = (0.875 * EstimatedRTT) + (0.125 * SampleRTT)
result =[]
for i in list:
    EstimatedRTT = (0.875 * firstEst) + (0.125 * i)
    firstEst = EstimatedRTT
    result.append(EstimatedRTT)

print(result)
