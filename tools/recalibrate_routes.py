import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv
import json

city_coords = {}
with open('data/cities.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        city_coords[row['City']] = (int(row['X']), int(row['Y']))

with open('data/route_waypoints.json', 'r') as f:
    route_waypoints = json.load(f)

routes_to_calibrate = [
    {'source': 'Berlin', 'target': 'Essen', 'carriages': 2, 'color': 'Blue'},
    {'source': 'Frankfurt', 'target': 'Paris', 'carriages': 3, 'color': 'Orange'},
    {'source': 'London', 'target': 'Edinburgh', 'carriages': 4, 'color': 'Black'},
    {'source': 'Roma', 'target': 'Brindisi', 'carriages': 2, 'color': 'White'},
    {'source': 'Palermo', 'target': 'Smyrna', 'carriages': 6, 'color': 'false'},
    {'source': 'Khobenhaven', 'target': 'Essen', 'carriages': 3, 'color': 'false'},
]

current_route_idx = 0
current_waypoints = []
fig = None
ax = None
scatter_points = []
line_plots = []

def get_route_key(route):
    return f"{route['source']}-{route['target']}-{route['color']}"

def on_click(event):
    global current_waypoints, scatter_points
    if event.xdata is None or event.ydata is None:
        return
    if event.button == 1:
        x, y = int(event.xdata), int(event.ydata)
        current_waypoints.append((x, y))
        point = ax.scatter(x, y, c='red', s=30, zorder=10)
        scatter_points.append(point)
        if len(current_waypoints) > 1:
            xs = [current_waypoints[-2][0], current_waypoints[-1][0]]
            ys = [current_waypoints[-2][1], current_waypoints[-1][1]]
            line, = ax.plot(xs, ys, 'r-', linewidth=2, zorder=9)
            line_plots.append(line)
        fig.canvas.draw()
        print(f"  Point {len(current_waypoints)}: ({x}, {y})")

def on_key(event):
    global current_route_idx, current_waypoints, scatter_points, line_plots

    if event.key == 'enter' or event.key == ' ':
        route = routes_to_calibrate[current_route_idx]
        key = get_route_key(route)

        if len(current_waypoints) == 0:
            src = city_coords.get(route['source'], (0, 0))
            tgt = city_coords.get(route['target'], (0, 0))
            route_waypoints[key] = [list(src), list(tgt)]
            print(f"  Using straight line (no waypoints clicked)")
        else:
            route_waypoints[key] = [list(wp) for wp in current_waypoints]

        print(f"  Saved {len(route_waypoints[key])} points for {key}")

        for p in scatter_points:
            p.remove()
        for l in line_plots:
            l.remove()
        scatter_points = []
        line_plots = []
        current_waypoints = []

        current_route_idx += 1

        if current_route_idx < len(routes_to_calibrate):
            show_current_route()
        else:
            print("\n=== ALL ROUTES DONE ===")
            save_waypoints()
            plt.close()

    elif event.key == 'backspace':
        if current_waypoints:
            current_waypoints.pop()
            if scatter_points:
                scatter_points[-1].remove()
                scatter_points.pop()
            if line_plots:
                line_plots[-1].remove()
                line_plots.pop()
            fig.canvas.draw()
            print("  Removed last point")

    elif event.key == 'escape':
        print("\n=== SAVING AND EXITING ===")
        save_waypoints()
        plt.close()

def show_current_route():
    route = routes_to_calibrate[current_route_idx]
    src_name = route['source']
    tgt_name = route['target']
    color = route['color']
    carriages = route['carriages']

    src = city_coords.get(src_name, (0, 0))
    tgt = city_coords.get(tgt_name, (0, 0))

    ax.set_title(f"Route {current_route_idx + 1}/{len(routes_to_calibrate)}: {src_name} -> {tgt_name} ({color}, {carriages} cars)\n"
                 f"Click waypoints along route, ENTER when done, BACKSPACE to undo, ESC to save & exit")

    for child in ax.get_children():
        if hasattr(child, '_route_highlight'):
            child.remove()

    highlight, = ax.plot([src[0], tgt[0]], [src[1], tgt[1]], 'g--', linewidth=3, alpha=0.5, zorder=6)
    highlight._route_highlight = True

    src_marker = ax.scatter(*src, c='lime', s=200, zorder=7, marker='o', edgecolors='black')
    src_marker._route_highlight = True
    tgt_marker = ax.scatter(*tgt, c='yellow', s=200, zorder=7, marker='s', edgecolors='black')
    tgt_marker._route_highlight = True

    fig.canvas.draw()
    print(f"\nRoute {current_route_idx + 1}/{len(routes_to_calibrate)}: {src_name} -> {tgt_name} ({color})")

def save_waypoints():
    with open('data/route_waypoints.json', 'w') as f:
        json.dump(route_waypoints, f, indent=2)
    print(f"Saved waypoints to data/route_waypoints.json")

img = mpimg.imread('assets/map.png')
fig, ax = plt.subplots(figsize=(16, 11))
ax.imshow(img)
ax.axis('off')

for city, (x, y) in city_coords.items():
    ax.scatter(x, y, c='white', s=50, zorder=5, edgecolors='black')
    ax.annotate(city, (x, y), fontsize=5, ha='center', va='bottom', xytext=(0, 3), textcoords='offset points')

fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('key_press_event', on_key)

print("=== ROUTE RECALIBRATION ===")
print("Click points along each route (in order from source to target)")
print("Press ENTER/SPACE when done with a route")
print("Press BACKSPACE to undo last point")
print("Press ESC to save and exit early")
print("")

show_current_route()
plt.show()
