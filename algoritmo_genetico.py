import tkinter as tk
from tkinter import ttk
import numpy as np
import random

# -------------------
# Configuracion algoritmo generico
# -------------------
population_size = 50
mutation_rate = 0.01
update_interval_ms = 50

# Paleta de colores
colors = [
    (173, 216, 230),  # Azul pastel
    (144, 238, 144),  # Verde pastel
    (255, 255, 224),  # Amarillo pastel
    (255, 255, 255),  # Blanco
    (0, 0, 0)         # Negro
]

# -------------------
# Interfaz
# -------------------
root = tk.Tk()
root.title("Replicador de dibujos con Algoritmo genetico")
root.configure(bg="black")

# Seleccionar tamaño del canvas
sizes = [f"{n}x{n}" for n in range(3, 8, 1)]
size_var = tk.StringVar(value="5x5")
size_dropdown = ttk.Combobox(root, textvariable=size_var, values=sizes, state="readonly")
size_dropdown.pack(pady=10)

# -------------------
# Opciones de colores 
# -------------------
color_frame = tk.Frame(root, bg="black")
color_frame.pack()

current_color = tk.IntVar(value=0)
border_frames = []
color_buttons = []

def set_color(idx):
    current_color.set(idx)
    for i, border in enumerate(border_frames):
        if i == idx:
            border.config(bg="red", bd=3, relief="solid")
        else:
            if colors[i] == (0, 0, 0):
                border.config(bg="white", bd=1, relief="flat")
            else:
                border.config(bg="black", bd=1, relief="flat")

for i, rgb in enumerate(colors):
    hex_color = "#%02x%02x%02x" % rgb

    border_bg = "white" if rgb == (0,0,0) else "black" 
    border = tk.Frame(color_frame, bg=border_bg, bd=1, relief="flat")
    border.grid(row=0, column=i, padx=6, pady=2)

    btn = tk.Button(border, bg=hex_color, width=3, height=2,
                    command=lambda i=i: set_color(i), relief="flat", bd=0)
    btn.pack(padx=2, pady=2)

    border_frames.append(border)
    color_buttons.append(btn)

set_color(current_color.get())

# Canvas de dibujo y de replica
canvas_frame = tk.Frame(root, bg="black")
canvas_frame.pack(pady=10)

target_canvas = tk.Canvas(canvas_frame, width=300, height=300, bg="black", highlightthickness=1, highlightbackground="white") # Canvas de dibujo
target_canvas.grid(row=0, column=0, padx=10)

ga_canvas = tk.Canvas(canvas_frame, width=300, height=300, bg="black", highlightthickness=1, highlightbackground="white") # Canvas de replica
ga_canvas.grid(row=0, column=1, padx=10)

# -------------------
# Dibujo
# -------------------
def draw_grid(canvas, size):
    """Dibujar lineas separadoras blancas."""
    cell_w = canvas.winfo_width() / size
    cell_h = canvas.winfo_height() / size
    for i in range(size + 1):
        canvas.create_line(i * cell_w, 0, i * cell_w, canvas.winfo_height(), fill="white")
        canvas.create_line(0, i * cell_h, canvas.winfo_width(), i * cell_h, fill="white")

def reset_canvases():
    """Aplicar tamaño a ambos canvas."""
    global target, population, best, best_score, generation, ga_running
    w, h = map(int, size_var.get().split("x"))
    target_canvas.delete("all")
    ga_canvas.delete("all")
    draw_grid(target_canvas, w)
    draw_grid(ga_canvas, w)

    # Iniciar con todo el canvas negro
    target = np.full((w, h), 4, dtype=int) 
    population = []
    best = None
    best_score = 0
    generation = 0
    ga_running = False

reset_canvases()

# Pintar
painting = False
def paint(event):
    if not painting:
        return
    w, h = map(int, size_var.get().split("x"))
    cell_w = target_canvas.winfo_width() / w
    cell_h = target_canvas.winfo_height() / h
    col = int(event.x // cell_w)
    row = int(event.y // cell_h)
    if col >= w or row >= h: 
        return
    color_idx = int(current_color.get())
    rgb = colors[color_idx]
    hex_color = "#%02x%02x%02x" % rgb
    target_canvas.create_rectangle(col*cell_w, row*cell_h, (col+1)*cell_w, (row+1)*cell_h,
                                   fill=hex_color, outline="white")
    target[row, col] = color_idx

def start_paint(event):
    global painting
    painting = True
    paint(event)

def stop_paint(event=None):
    global painting
    painting = False

target_canvas.bind("<Button-1>", start_paint)
target_canvas.bind("<B1-Motion>", paint)
target_canvas.bind("<ButtonRelease-1>", stop_paint)

# -------------------
# Algoritmo generativo
# -------------------
def fitness(individual, target):
    return np.sum(individual == target)

def crossover(p1, p2):
    mask = np.random.rand(*p1.shape) < 0.5
    return np.where(mask, p1, p2)

def mutate(child, rate):
    mutation_mask = np.random.rand(*child.shape) < rate
    random_colors = np.random.randint(0, len(colors), child.shape)
    return np.where(mutation_mask, random_colors, child)

def update_canvas_ga(individual):
    ga_canvas.delete("all")
    w, h = individual.shape[1], individual.shape[0]
    cell_w = ga_canvas.winfo_width() / w
    cell_h = ga_canvas.winfo_height() / h
    for row in range(h):
        for col in range(w):
            rgb = colors[individual[row, col]]
            hex_color = "#%02x%02x%02x" % rgb
            ga_canvas.create_rectangle(col*cell_w, row*cell_h,
                                       (col+1)*cell_w, (row+1)*cell_h,
                                       fill=hex_color, outline="white")

def ga_step():
    global population, best, best_score, generation, ga_running
    if not ga_running:
        return

    scores = [fitness(ind, target) for ind in population]
    max_idx = int(np.argmax(scores))

    best, best_score = population[max_idx], scores[max_idx]

    # Mostrar la mejor puntuacion para cada intento
    print(f"Gen {generation} | Best score {best_score}/{target.size}")
    update_canvas_ga(best)

    if best_score == target.size:
        print("Perfect match found!")
        ga_running = False
        return

    # Seleccionar el mejor
    if sum(scores) == 0:
        parents = random.choices(population, k=len(population))
    else:
        parents = random.choices(population, weights=scores, k=len(population))

    next_pop = []
    for i in range(0, len(parents), 2):
        p1 = parents[i]
        p2 = parents[(i+1) % len(parents)]
        c1 = crossover(p1, p2)
        c2 = crossover(p2, p1)
        next_pop.append(mutate(c1, mutation_rate))
        next_pop.append(mutate(c2, mutation_rate))

    population = next_pop
    generation += 1

    root.after(update_interval_ms, ga_step)

def start_ga():
    global population, best, best_score, generation, ga_running
    w, h = map(int, size_var.get().split("x"))
    population = [np.random.randint(0, len(colors), (h, w)) for _ in range(population_size)]
    best = None
    best_score = 0
    generation = 0
    ga_running = True
    ga_step()

# -------------------
# Botones
# -------------------
control_frame = tk.Frame(root, bg="black")
control_frame.pack(pady=5)

size_dropdown.bind("<<ComboboxSelected>>", lambda event: reset_canvases())

replicate_btn = tk.Button(control_frame, text="Replicar", command=start_ga, bg="gray20", fg="white")
replicate_btn.grid(row=0, column=1, padx=5)

# -------------------
# Ejecucion
# -------------------
root.after(100, reset_canvases)
root.mainloop()
