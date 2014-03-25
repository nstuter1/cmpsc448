from numpy import *
import csv

output_file = csv.DictReader(open("NCAAStats.csv"))

stats = None

for stat_row in output_file:
    new_row = array([[stat_row["Season"], int(stat_row["Team"]), float(stat_row["WinPercent"])]])
    if stats == None:
        stats = new_row
    else:
        stats = concatenate((stats,new_row), axis=0)
print stats

for row in stats:
    count = 0
    average = 0
    lookup_team = 0
    season = row[0]
    team = int(row[1])
    input_file = csv.DictReader(open("NCAARawData.csv"))
    for test_row in input_file:
        if(season == test_row["season"] and (team == int(test_row["wteam"]) or team == int(test_row["lteam"]))):
            if(team == int(test_row["wteam"])):
                lookup_team = int(test_row["lteam"])
            else:
                lookup_team = int(test_row["wteam"])
            
        
