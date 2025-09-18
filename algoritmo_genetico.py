"""
Este script implementa un algoritmo genético para replicar un dibujo simple creado por el usuario en un lienzo. 
El usuario dibuja un patrón en el lienzo izquierdo, y el algoritmo genético en el lado derecho intenta evolucionar 
una población de lienzos generados aleatoriamente hasta que uno de ellos coincide con el dibujo del usuario.

La interfaz gráfica está construida con Tkinter.
"""

# -------------------
# Importación de librerías
# -------------------
import tkinter as tk
from tkinter import ttk
import numpy as np
import random

# -------------------
# Configuración del algoritmo genético
# -------------------

population_size = 50      # Tamaño de la población (número de individuos en cada generación)
mutation_rate = 0.01      # Probabilidad de que un píxel (gen) mute en un individuo
update_interval_ms = 50   # Intervalo en milisegundos para actualizar la interfaz y ejecutar un paso del algoritmo genetico
elitism_rate = 0.1        # Porcentaje de los mejores individuos que pasan a la siguiente generación sin cambios
no_improve_limit = 50     # Límite de generaciones sin mejora antes de aumentar la tasa de mutación

# Paleta de colores (índices de 0 a 4) en RGB
colors = [
    (173, 216, 230),  # Azul pastel
    (144, 238, 144),  # Verde pastel
    (255, 255, 224),  # Amarillo pastel
    (255, 255, 255),  # Blanco
    (0, 0, 0)         # Negro
]

# -------------------
# Interfaz principal
# -------------------

# Se crea la ventana principal de la aplicación.
root = tk.Tk()
root.title("Replicador de dibujos con Algoritmo Genético Optimizado")
root.configure(bg="black")

# Menú desplegable para seleccionar el tamaño del lienzo.
sizes = [f"{n}x{n}" for n in range(3, 8, 1)] # Opciones de 3x3 a 7x7
size_var = tk.StringVar(value="5x5") # Se crea un lienzo 5x5 por defecto
size_dropdown = ttk.Combobox(root, textvariable=size_var, values=sizes, state="readonly")
size_dropdown.pack(pady=10)

# -------------------
# Opciones de colores (paleta de selección)
# -------------------

# Contenedor para los botones de la paleta de colores.
color_frame = tk.Frame(root, bg="black")
color_frame.pack()

# Variable para almacenar el índice del color actualmente seleccionado.
current_color = tk.IntVar(value=0)
border_frames = [] # Lista para guardar los marcos de los botones de color (para el resaltado)
color_buttons = [] # Lista para guardar los botones de color

def set_color(idx):
    """
    Establece el color de dibujo actual y resalta el botón de color seleccionado con un borde rojo. El color negro siempre tiene un borde blanco para ser visible.
    """
    current_color.set(idx)
    for i, border in enumerate(border_frames):
        if i == idx:
            # Resalta el color seleccionado con un borde rojo.
            border.config(bg="red", bd=3, relief="solid")
        else:
            # Quita el resaltado de los otros colores.
            if colors[i] == (0, 0, 0): # Borde blanco para el negro para que sea visible.
                border.config(bg="white", bd=1, relief="flat")
            else:
                border.config(bg="black", bd=1, relief="flat")

# Se crean los botones para cada color en la paleta.
for i, rgb in enumerate(colors):
    hex_color = "#%02x%02x%02x" % rgb # Convierte RGB a formato hexadecimal
    border_bg = "white" if rgb == (0, 0, 0) else "black"
    border = tk.Frame(color_frame, bg=border_bg, bd=1, relief="flat")
    border.grid(row=0, column=i, padx=6, pady=2)
    btn = tk.Button(border, bg=hex_color, width=3, height=2, command=lambda i=i: set_color(i), relief="flat", bd=0)
    btn.pack(padx=2, pady=2)
    border_frames.append(border)
    color_buttons.append(btn)

# Se establece el color inicial al cargar la aplicación.
set_color(current_color.get())

# -------------------
# Lienzos
# -------------------

# Contenedor para contener los dos lienzos.
canvas_frame = tk.Frame(root, bg="black")
canvas_frame.pack(pady=10)

# Lienzo izquierdo: el usuario dibuja el objetivo aquí.
target_canvas = tk.Canvas(canvas_frame, width=300, height=300, bg="black", highlightthickness=1, highlightbackground="white")
target_canvas.grid(row=0, column=0, padx=10)

# Lienzo derecho: muestra el mejor individuo generado por el algoritmo genético.
ga_canvas = tk.Canvas(canvas_frame, width=300, height=300, bg="black", highlightthickness=1, highlightbackground="white")
ga_canvas.grid(row=0, column=1, padx=10)

# -------------------
# Etiquetas de estado
# -------------------

# Contenido para mostrar información sobre el progreso del algoritmo.
status_frame = tk.Frame(root, bg="black")
status_frame.pack(pady=5)

# Etiqueta para mostrar el número de la generación actual.
gen_label = tk.Label(status_frame, text="Generación: 0", fg="white", bg="black")
gen_label.pack(side="left", padx=10)

# Etiqueta para mostrar la puntuación (fitness) del mejor individuo.
best_label = tk.Label(status_frame, text="Mejor puntaje: 0", fg="white", bg="black")
best_label.pack(side="left", padx=10)

# -------------------
# Funciones de dibujo
# -------------------

def draw_grid(canvas, size):
    """Dibuja una cuadrícula blanca sobre un fondo negro en el lienzo especificado, para asi separar cada uno de los cuadros."""
    cell_w = canvas.winfo_width() / size
    cell_h = canvas.winfo_height() / size
    for i in range(size + 1):
        canvas.create_line(i * cell_w, 0, i * cell_w, canvas.winfo_height(), fill="white")
        canvas.create_line(0, i * cell_h, canvas.winfo_width(), i * cell_h, fill="white")

def reset_canvases():
    """Limpia y reinicia los lienzos y las variables del algoritmo genético."""
    global target, population, best, best_score, generation, ga_running, stagnant_gens, mutation_rate
    w, h = map(int, size_var.get().split("x"))
    
    # Limpia los lienzos y dibuja las nuevas cuadrículas.
    target_canvas.delete("all")
    ga_canvas.delete("all")
    draw_grid(target_canvas, w)
    draw_grid(ga_canvas, w)
    
    # Reinicia las estructuras de datos y variables del algoritmo genetico.
    target = np.full((w, h), 4, dtype=int)  # Inicializa el lienzo con fondo negro.
    population = []
    best = None
    best_score = 0
    generation = 0
    stagnant_gens = 0
    mutation_rate = 0.01
    ga_running = False
    
    # Reinicia las etiquetas de estado.
    gen_label.config(text="Generación: 0")
    best_label.config(text="Mejor puntaje: 0")

# Llama a reset_canvases al inicio para preparar la interfaz.
reset_canvases()

# -------------------
# Pintura manual en el lienzo objetivo
# -------------------

painting = False # Bandera para controlar si el usuario está dibujando.

def paint(event):
    """Pinta una celda en el lienzo objetivo con el color seleccionado."""
    if not painting: return
    w, h = map(int, size_var.get().split("x"))
    cell_w = target_canvas.winfo_width() / w
    cell_h = target_canvas.winfo_height() / h
    
    # Calcula la fila y columna basado en la posición del ratón.
    col = int(event.x // cell_w)
    row = int(event.y // cell_h)
    
    if col >= w or row >= h: return # Evita errores si el cursor sale del lienzo.
    
    color_idx = int(current_color.get())
    rgb = colors[color_idx]
    hex_color = "#%02x%02x%02x" % rgb
    
    # Dibuja el rectángulo en la celda correspondiente.
    target_canvas.create_rectangle(col*cell_w, row*cell_h, (col+1)*cell_w, (row+1)*cell_h, fill=hex_color, outline="white")
    # Actualiza la matriz del objetivo con el nuevo color.
    target[row, col] = color_idx

def start_paint(event):
    """Se activa al hacer clic, inicia el modo de pintura."""
    global painting
    painting = True
    paint(event)

def stop_paint(event=None):
    """Se activa al soltar el clic, detiene el modo de pintura."""
    global painting
    painting = False

# Asigna los eventos del ratón a las funciones de pintura.
target_canvas.bind("<Button-1>", start_paint) # Empieza a pintar al hacer clic.
target_canvas.bind("<B1-Motion>", paint) # Si se mueve el mouse con el clic presionado, sigue pintando.
target_canvas.bind("<ButtonRelease-1>", stop_paint) # Al soltar el clic, deja de dibujar.

# -------------------
# Algoritmo genético optimizado
# -------------------

def fitness(individual, target):
    """Calcula la puntuación de un individuo comparándolo con el objetivo.
    La puntuación es el número de píxeles que coinciden."""
    return np.sum(individual == target)

def crossover(p1, p2):
    """
    Realiza el cruce entre dos padres (p1, p2) para crear un hijo.
    Se utiliza un cruce de un punto uniforme: para cada gen (píxel), se elige aleatoriamente si el hijo hereda el gen de p1 o p2.
    """
    mask = np.random.rand(*p1.shape) < 0.5
    return np.where(mask, p1, p2)

def mutate(child, rate):
    """
    Aplica una mutación al individuo (hijo) con una probabilidad 'rate'.
    Si un gen (píxel) muta, se le asigna un color aleatorio de la paleta.
    """
    mutation_mask = np.random.rand(*child.shape) < rate
    random_colors = np.random.randint(0, len(colors), child.shape)
    return np.where(mutation_mask, random_colors, child)

def update_canvas_ga(individual):
    """Dibuja el estado de un individuo en el lienzo del algoritmo genético."""
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
    """Ejecuta un paso (una generación) del algoritmo genético."""
    global population, best, best_score, generation, ga_running, stagnant_gens, mutation_rate
    if not ga_running: return

    # 1. Evaluación: Calcula el fitness (puntuacion) de cada individuo de la población.
    scores = [fitness(ind, target) for ind in population]
    max_idx = int(np.argmax(scores))
    current_best, current_score = population[max_idx], scores[max_idx]

    # Actualiza el mejor individuo global si el de la generación actual es mejor.
    if current_score > best_score:
        best, best_score = current_best, current_score
        stagnant_gens = 0 # Reinicia el contador de estancamiento.
    else:
        stagnant_gens += 1 # Incrementa el contador si no hay mejora, significa que está estancado.

    # Actualiza las etiquetas de la interfaz.
    print(f"Gen {generation} | Best {best_score}/{target.size}")
    gen_label.config(text=f"Generación: {generation}")
    best_label.config(text=f"Mejor puntaje: {best_score}/{target.size}")
    update_canvas_ga(current_best)

    # Parar el algoritmo genetico una vez se encuentre la coincidencia perfecta.
    if best_score == target.size:
        print("¡Coincidencia perfecta encontrada!")
        ga_running = False
        return

    # 2. Selección: Preparar la siguiente generación.
    # Elitismo: Conserva un porcentaje de los mejores individuos.
    elite_count = max(1, int(elitism_rate * population_size))
    elite_idx = np.argsort(scores)[-elite_count:]
    elites = [population[i] for i in elite_idx]

    # Selección por ruleta: selecciona padres basados en su fitness (puntuación).
    if sum(scores) == 0:
        # Si todos tienen score 0, la selección es aleatoria.
        parents = random.choices(population, k=len(population)-elite_count)
    else:
        # Los individuos con mayor score tienen más probabilidad de ser elegidos.
        parents = random.choices(population, weights=scores, k=len(population)-elite_count)

    # 3. Reproducción: Crear la nueva población.
    next_pop = elites.copy() # La nueva población empieza con la élite.
    for i in range(0, len(parents), 2):
        p1 = parents[i]
        p2 = parents[(i+1) % len(parents)] # Asegura que siempre haya un par.
        
        # Cruce
        c1 = crossover(p1, p2)
        c2 = crossover(p2, p1)
        
        # Mutación y adición a la nueva población
        next_pop.append(mutate(c1, mutation_rate))
        next_pop.append(mutate(c2, mutation_rate))

    population = next_pop[:population_size] # Asegura que la población tenga el tamaño correcto.
    generation += 1

    # Mutación adaptativa: si el algoritmo se estanca, aumenta la mutación.
    if stagnant_gens > no_improve_limit:
        mutation_rate = min(0.2, mutation_rate * 1.5) # Aumenta la mutación, con un límite.
        stagnant_gens = 0
        print("Aumentando tasa de mutación:", mutation_rate)

    # Programa el siguiente paso del algoritmo.
    root.after(update_interval_ms, ga_step)

def start_ga():
    """Inicializa y comienza la ejecución del algoritmo genético."""
    global population, best, best_score, generation, ga_running, mutation_rate
    w, h = map(int, size_var.get().split("x"))
    
    # Crea la población inicial con individuos aleatorios.
    population = [np.random.randint(0, len(colors), (h, w)) for _ in range(population_size)]
    
    # Reinicia las variables de estado del algoritmo genético.
    best = None
    best_score = 0
    generation = 0
    mutation_rate = 0.01
    ga_running = True
    
    # Inicia el bucle del algoritmo.
    ga_step()

# -------------------
# Botones de control
# -------------------

control_frame = tk.Frame(root, bg="black")
control_frame.pack(pady=5)

# Cuando se cambia el tamaño en el dropdown, se reinician los lienzos.
size_dropdown.bind("<<ComboboxSelected>>", lambda event: reset_canvases())

# Botón para iniciar el proceso de replicación.
replicate_btn = tk.Button(control_frame, text="Replicar", command=start_ga, bg="gray20", fg="white")
replicate_btn.grid(row=0, column=1, padx=5)

# -------------------
# Ejecución
# -------------------

# Llama a reset_canvases una vez después de que la ventana se haya cargado para asegurar tamaños correctos.
root.after(100, reset_canvases)
# Inicia el bucle principal de la interfaz gráfica.
root.mainloop()
