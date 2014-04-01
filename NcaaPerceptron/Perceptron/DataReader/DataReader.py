import numpy as np
import csv
from collections import OrderedDict
stats_file = csv.DictReader(open("NCAAStats.csv"))
ordered_fieldnames = OrderedDict([("season",None),("Team",None),("Win%",None),("OppWin%",None)])

stats = None
output_stats = None
for stat_row in stats_file:
    new_row = np.array([[stat_row["Season"], int(stat_row["Team"]), float(stat_row["WinPercent"])]])
    if stats == None:
        stats = new_row
    else:
        stats = np.concatenate((stats,new_row), axis=0)

new_stats = None

for row in stats:  
    count = 0
    average = 0
    lookup_team = 0
    season = row[0]
    team = int(row[1])
    score = float(row[2])
    if(score == 0.0):
        np.savetxt("NCAAAppendStats.csv",row,delimiter=',', fmt="%s", header="Season,Team,Win%,OppWin%\n")  
        print(row)
        continue
    output_array = None
    new_stats = None
    input_file = csv.DictReader(open("NCAARawData.csv"))
    for test_row in input_file:
        if(season == test_row["season"] and (team == int(test_row["wteam"]) or team == int(test_row["lteam"]))):
            if(team == int(test_row["wteam"])):
                lookup_team = int(test_row["lteam"])
            else:
                lookup_team = int(test_row["wteam"])
            for look_row in stats:
                if int(look_row[1]) == lookup_team and season == look_row[0]:
                    average += float(look_row[2])
                    count += 1
                    break
    if(count > 0):
        final_average = average / count
        output_row = np.append(row, final_average)
    else:
        output_row = row
    if new_stats == None:
        new_stats = output_row
    else:
        new_stats = np.concatenate((new_stats,output_row),axis=0)
    np.savetxt("NCAAAppendStats.csv",new_stats,delimiter=',', fmt="%s", header="Season,Team,Win%,OppWin%\n")  
    print(row)
