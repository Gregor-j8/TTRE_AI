import matplotlib.pyplot as plt
import matplotlib.image as mpimg

cities = [
    "Lisboa", "Cadiz", "Madrid", "Barcelona", "Pamplona", "Marseille", "Paris",
    "Brest", "Dieppe", "Zurich", "London", "Edinburgh", "Venezia", "Roma",
    "Munchen", "Frankfurt", "Bruxelles", "Amsterdam", "Essen", "Berlin",
    "Wien", "Zagrab", "Budapest", "Brindisi", "Palermo", "Sarajevo", "Athina",
    "Sofia", "Smyrna", "Constantinople", "Bucuresti", "Kyiv", "Warzawa",
    "Sevastopol", "Angora", "Erzurum", "Sochi", "Rostov", "Kharkov", "Moskva",
    "Smolensk", "Wilno", "Petrograd", "Stockholm", "Riga", "Danzic", "Khobenhavn"
]

coords = {}
current_idx = 0

def on_click(event):
    global current_idx
    if event.xdata is None or event.ydata is None:
        return

    if current_idx < len(cities):
        city = cities[current_idx]
        x, y = int(event.xdata), int(event.ydata)
        coords[city] = (x, y)
        print(f"{city},{x},{y}")

        ax.scatter(x, y, c='red', s=50, zorder=10)
        ax.annotate(city, (x, y), fontsize=6, color='red')
        fig.canvas.draw()

        current_idx += 1

        if current_idx < len(cities):
            ax.set_title(f"Click on: {cities[current_idx]} ({current_idx + 1}/{len(cities)})")
        else:
            ax.set_title("Done! Close window to save.")
            save_csv()

    fig.canvas.draw()

def save_csv():
    print("\n--- CSV OUTPUT ---")
    print("City,X,Y")
    for city in cities:
        if city in coords:
            x, y = coords[city]
            print(f"{city},{x},{y}")

img = mpimg.imread('assets/map.png')
fig, ax = plt.subplots(figsize=(16, 11))
ax.imshow(img)
ax.set_title(f"Click on: {cities[0]} (1/{len(cities)})")
ax.axis('off')

fig.canvas.mpl_connect('button_press_event', on_click)

print("Click on each city in order. Coordinates will be printed.")
print("City,X,Y")
plt.show()
