import pandas as pd
import numpy as np
import matplotlib.pyplot as matplt

data = pd.read_csv("https://raw.githubusercontent.com/riziry/fuzzyLogic/master/datasheetRestaurant.csv")
print("**Dataset**\n", data)

# membership
# -Precise rule
service_rule = {"sMin" : [30,45], "sAvg_min" : [40, 50, 55, 65], "sAvg_max" : [60, 70, 80, 85], "sMax" : [80, 90]}
    
foods_rule = {"fMin" : [4,4.5], "fAvg_min" : [4, 5, 5.5, 6], "fAvg_max" : [5.5, 6, 7, 7.5], "fMax" : [7, 9]}

# -func to calculate rule
def minimum(x, a, b):
    if a < x and x <= b:
        temp = (b - x) / (b - a)
    elif x <= a:
        temp = 1
    else:
        temp = 0

    return temp

def average(x, a, b, c, d):
    if x < a or x > d:
        temp = 0
    else:
        if a < x and x < b:
            temp = (x - a) / (b - a)
        elif c < x and x <= d:
            temp = (d - x) / (d - c)
        else:
            temp = 1

    return temp

def maximum(x, a, b):
    if a < x and x <= b:
        return (x - a) / (b - a)
    elif x >= b:
        return 1
    else:
        return 0

# -Services
def services_max(x):
    return maximum(x, service_rule["sMax"][0], service_rule["sMax"][1])

def services_avg_max(x):
    return average(x, service_rule["sAvg_max"][0], service_rule["sAvg_max"][1], service_rule["sAvg_max"][2], service_rule["sAvg_max"][3])

def services_avg_min(x):
    return average(x, service_rule["sAvg_min"][0], service_rule["sAvg_min"][1], service_rule["sAvg_min"][2], service_rule["sAvg_min"][3])

def services_min(x):
    return minimum(x, service_rule["sMin"][0], service_rule["sMin"][1])

# -Foods
def foods_max(x):
    return maximum(x, foods_rule["fMax"][0], foods_rule["fMax"][1])

def foods_avg_max(x):
    return average(x, foods_rule["fAvg_max"][0], foods_rule["fAvg_max"][1], foods_rule["fAvg_max"][2], foods_rule["fAvg_max"][3])

def foods_avg_min(x):
    return average(x, foods_rule["fAvg_min"][0], foods_rule["fAvg_min"][1], foods_rule["fAvg_min"][2], foods_rule["fAvg_min"][3])

def foods_min(x):
    return minimum(x, foods_rule["fMin"][0], foods_rule["fMin"][1])

# Plot(service)
def membership_plot():
    s_x = np.linspace(0, 100, 100)
    
    # -adding to matplot
    matplt.plot(s_x, [services_max(x) for x in s_x])
    matplt.plot(s_x, [services_avg_max(x) for x in s_x])
    matplt.plot(s_x, [services_avg_min(x) for x in s_x])
    matplt.plot(s_x, [services_min(x) for x in s_x])
    matplt.title("Membership of services quality")
    # -Print plot
    matplt.show()


    # Plot(foods)
    f_x = np.linspace(0, 10, 100)

    # -adding to matplot
    matplt.plot(f_x, [foods_max(x) for x in f_x])
    matplt.plot(f_x, [foods_avg_max(x) for x in f_x])
    matplt.plot(f_x, [foods_avg_min(x) for x in f_x])
    matplt.plot(f_x, [foods_min(x) for x in f_x])
    matplt.title("Membership of foods quality")
    matplt.show()

# Design fuzzy Rule
"""
[0-5]star
|   **Services**    |   **Foods**   |   **Score**   |
|___________________________________________________|
|   sMax            |   fMax        |   5-Star      |
|   sMax            |   fAvg_max    |   4-star      |
|   sMax            |   fAvg_min    |   3-star      |
|   sMax            |   fMin        |   2-star      |
|   sAvg_max        |   fMax        |   4-Star      |
|   sAvg_max        |   fAvg_max    |   4-star      |
|   sAvg_max        |   fAvg_min    |   3-star      |
|   sAvg_max        |   fMin        |   2-star      |
|   sAvg_min        |   fMax        |   3-Star      |
|   sAvg_min        |   fAvg_max    |   3-star      |
|   sAvg_min        |   fAvg_min    |   2-star      |
|   sAvg_min        |   fMin        |   1-star      |
|   sMin            |   fMax        |   2-Star      |
|   sMin            |   fAvg_max    |   2-star      |
|   sMin            |   fAvg_min    |   1-star      |
|   sMin            |   fMin        |   0-star      |
"""

# rule base for inference engine
def rule_base():
    fuzzyness = []
    
    for service, foods in zip(data["pelayanan"], data["makanan"]):
        fuzzyness.append([
            {"5-Star" : min(services_max(service), foods_max(foods))},
            {"4-Star" : min(services_max(service), foods_avg_max(foods))},
            {"3-star" : min(services_max(service), foods_avg_min(foods))},
            {"2-Star" : min(services_max(service), foods_min(foods))},
            {"4-Star" : min(services_avg_max(service), foods_max(foods))},
            {"4-Star" : min(services_avg_max(service), foods_avg_max(foods))},
            {"3-Star" : min(services_avg_max(service), foods_avg_min(foods))},
            {"2-Star" : min(services_avg_max(service), foods_min(foods))},
            {"3-Star" : min(services_avg_min(service), foods_max(foods))},
            {"3-Star" : min(services_avg_min(service), foods_avg_max(foods))},
            {"2-Star" : min(services_avg_min(service), foods_avg_min(foods))},
            {"1-Star" : min(services_avg_min(service), foods_min(foods))},
            {"2-Star" : min(services_min(service), foods_max(foods))},
            {"2-Star" : min(services_min(service), foods_avg_max(foods))},
            {"1-Star" : min(services_min(service), foods_avg_min(foods))},
            {"0-Star" : min(services_min(service), foods_min(foods))}
            ])

    return fuzzyness

def inference(fuzzyness_data):
    score = []
    counter = 1

    for i in fuzzyness_data:
        max_0star = 0
        max_1star = 0
        max_2star = 0
        max_3star = 0
        max_4star = 0
        max_5star = 0

        for j in i:
            for key, value in j.items():
                if key == "0-Star":
                    max_0star = max(max_0star, value)
                elif key == "1-Star":
                    max_1star = max(max_1star, value)
                elif key == "2-Star":
                    max_2star = max(max_2star, value)
                elif key == "3-Star":
                    max_3star = max(max_3star, value)
                elif key == "4-Star":
                    max_4star = max(max_4star, value)
                else: 
                    max_5star = max(max_5star, value)

        score.append({
                "0-Star" : max_0star,
                "1-Star" : max_1star,
                "2-Star" : max_2star,
                "3-Star" : max_3star,
                "4-Star" : max_4star,
                "5-Star" : max_5star
                })
    
    return score

# Checkpoint
def defuzzy_formula(score):
    return ((score["5-Star"] * 100) + (score["4-Star"] * 80) + (score["3-Star"] * 60) + (score["2-Star"] * 40) + (score["1-Star"] * 20) + (score["0-Star"] * 10)) / (score["5-Star"] + score["4-Star"] + score["3-Star"] + score["2-Star"] + score["1-Star"] + score["0-Star"])

def defuzzy(score):
    final_data = []

    for set_value in score:
        final_data.append(defuzzy_formula(set_value))

    return final_data
        
membership_plot()
data["score"] = defuzzy(inference(rule_base()))

top10 = data.sort_values(by=["score"])
print("\n\n==Top 10 restaurant==\n", top10[100:90:-1])
