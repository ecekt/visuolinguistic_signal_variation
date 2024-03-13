import json

# 4586

with open('fixation_events_DS_2023.json', 'r') as f:
    fx2 = json.load(f)

count = 0
max_x = 0
max_y = 0

for ppn in fx2:
    for im in fx2[ppn]:
        count += 1

        for fx_window in fx2[ppn][im]:
            for fx in fx_window:
                _, _, _, xp, yp = fx
                xp = float(xp)
                yp = float(yp)

                if xp > max_x:
                    max_x = xp
                if yp > max_y:
                    max_y = yp

print(count)


