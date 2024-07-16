from data_structures.compound_pixel_grid import CompoundExpandablePixelGrid
from data_structures.vectors import Position2D
import copy
import math
import skimage

import numpy as np
import cv2 as cv

import os

from flags import SHOW_MAP_AT_END, DO_SAVE_FINAL_MAP, SAVE_FINAL_MAP_DIR, DO_SAVE_DEBUG_GRID, SAVE_DEBUG_GRID_DIR
from mapping import mapper
import time
from collections import deque
import copy
#from executor import executor
#from executor.executor import Executor
#from fixture_detection.fixture_clasification import FixtureClasiffier
#from fixture_detection.fixture_detection import FixtureDetector

class pre_matrix:
    def __init__(self, square_size_px: int):
        self.threshold = 10
        self.__square_size_px = square_size_px

    def preload_matrix(self, wall_array: np.ndarray):
            """
            Transform wall array to boolean node array.
            """
            shape = wall_array.shape
            bool_node_array = np.zeros(shape, dtype=bool)
            
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if wall_array[i, j]:
                        bool_node_array[i, j] = True
            
            #print(type(bool_node_array))
            return bool_node_array
    
    """def correct_preload_victim(self, victims_array: np.ndarray):
        shape = victims_array.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                if victims_array[i, j] == False:
                    countTrue = 0
                    fila = -1
                    columna = -1
                    for a in range (3):
                        for b in range (3):
                            try:
                                if victims_array[i + (fila), j + (columna)] == True:
                                    columna += 1
                                    countTrue += 1
                                else:
                                    columna += 1
                            except IndexError:
                                columna +=1
                                pass
                        fila += 1
                        columna = -1

                    if countTrue >= 3:
                        victims_array[i, j] = True
        return victims_array """

    def correct_preload_victim(self, victims_array: np.ndarray): #agrega mas True para que se marque el espacio de la victima en la matriz
            shape = victims_array.shape
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if victims_array[i, j] == False:
                        countTrue = 0
                        fila = -3
                        columna = 0
                        for a in range (7):
                            for b in range (2):
                                try:
                                    if victims_array[i + (fila), j + (columna)] == True:
                                        columna += 1
                                        countTrue += 1
                                    else:
                                        columna += 1
                                except IndexError:
                                    columna +=1
                                    pass
                            fila += 1
                            columna = 0

                        if countTrue >= 5:
                            victims_array[i, j] = True
            return victims_array
    
    def removing_incorrect_True(self, victims_array: np.ndarray): #elimina True incorrectos
            shape = victims_array.shape
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if victims_array[i, j] == True:
                        countTrue = 0
                        fila = -2
                        columna = -2
                        for a in range (5):
                            for b in range (5):
                                try:
                                    if victims_array[i + (fila), j + (columna)] == True:
                                        columna += 1
                                        countTrue += 1
                                    else:
                                        columna += 1
                                except IndexError:
                                    columna +=1
                                    pass
                            fila += 1
                            columna = -2

                        if countTrue < 5:
                            victims_array[i, j] = False
            return victims_array



    def preload_final_matrix(self, walls_array: np.ndarray, victims_array: np.ndarray):
        shape = walls_array.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                if victims_array[i, j]:
                    walls_array[i-2, j-3] = False

        return walls_array
    
    def preload_final_matrix2(self, walls_array: np.ndarray, victims_array: np.ndarray):
        shape = walls_array.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                if victims_array[i, j]:
                    walls_array[i, j-3] = False

        return walls_array

    
#    def victim_in_grid(self):
#        letter_position = FixtureDetector.get_fixture_positions_in_image("Y") * "ancho_de_matriz" + FixtureDetector.get_fixture_positions_in_image("X")
#        return letter_position

class WallMatrixCreator:
    def __init__(self, square_size_px: int):
        self.threshold = 10
        self.__square_size_px = square_size_px

        #plantillas:
        straight = [
            [0, 0, 1, 2, 2, 2, 2, 1, 0, 0],
            [0, 0, 1, 2, 2, 2, 2, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
        
        self.straight_template = np.array(straight)

        
        vortex = [
            [3, 3, 3, 0, 0, 0, 0, 0, 0, 0],
            [3, 3, 3, 0, 0, 0, 0, 0, 0, 0],
            [3, 3, 3, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
        
        self.vortex_template = np.array(vortex)
        


        self.templates = {}

        for i, name in enumerate([(-1, 0), (0,-1), (1,0), (0,1)]):
            self.templates[name] = np.rot90(self.straight_template, i)
        
        for i, name in enumerate([(-1,-1), (1, -1), (1, 1), (-1, 1)]):
           self.templates[name] = np.rot90(self.vortex_template, i)
        
        #rota y define la orientacion de matriz

    def __get_tile_status(self, min_x, min_y, max_x, max_y, wall_array: np.ndarray) -> list:
        counts = {name: 0 for name in self.templates}
        square = wall_array[min_x:max_x, min_y:max_y]
        if square.shape != (self.__square_size_px, self.__square_size_px):
            return []

        non_zero_indices = np.where(square != 0)
        #print("imprimo non_zeros en status real")
        #print(non_zero_indices)
        for orientation, template in self.templates.items():
            counts[orientation] = np.sum(template[non_zero_indices])

        status = []
        for orientation, count in counts.items():
            if count >= self.threshold:
                status.append(orientation)

        return status

        
    def __get_tile_status_victim(self, min_x, min_y, max_x, max_y, fixture_array: np.ndarray) -> list:
        square = fixture_array[min_x:max_x, min_y:max_y]
        indice_a = 0
        indice_b = 0
        status = []
        format_element = ""
        for i in range ((square.shape[0])):
            for a in range((square.shape[1])):
                element = square[indice_a, indice_b]
                if element != (""):
                    format_element = element
                    indice_b +=1
                else:
                    indice_b +=1
            if format_element != (""):
                status.append(format_element)
                break
            indice_b = 0
            indice_a += 1
        if format_element == (""):
            status.append(format_element)


        return status
        
    
    def transform_wall_array_to_bool_node_array(self, wall_array: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        grid = []
        if SHOW_MAP_AT_END:
            bool_array_copy = wall_array.astype(np.uint8) * 100
        for x in range(offsets[0], wall_array.shape[0] - self.__square_size_px, self.__square_size_px):
            row = []
            for y in range(offsets[1], wall_array.shape[1] - self.__square_size_px, self.__square_size_px):
                min_x = x
                min_y = y
                max_x = x + self.__square_size_px
                max_y = y + self.__square_size_px
                #print(f"min_x: {min_x} min_y: {min_y} max_x: {max_x} max_y: {max_y}")
                if SHOW_MAP_AT_END:
                    bool_array_copy = cv.rectangle(bool_array_copy, (min_y, min_x), (max_y, max_x), (255,), 1)
                
                val = self.__get_tile_status(min_x, min_y, max_x, max_y, wall_array)
                
                row.append(list(val))
            grid.append(row)

        
        if SHOW_MAP_AT_END:
            cv.imshow("point_cloud_with_squares", cv.resize(bool_array_copy, (0, 0), fx=1, fy=1, interpolation=cv.INTER_AREA))
        #print("grid real para ver que pasa")
        #print(grid)
        grid = self.__orientation_grid_to_final_wall_grid(grid)
        return grid
    
    def transform_robot_detected_to_string_node_array(self, fixture_array: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        grid = []
        for x in range(offsets[0], fixture_array.shape[0] - self.__square_size_px, self.__square_size_px):
            row = []
            for y in range(offsets[1], fixture_array.shape[1] - self.__square_size_px, self.__square_size_px):
                min_x = x
                min_y = y
                max_x = x + self.__square_size_px
                max_y = y + self.__square_size_px
                #print(min_x, min_y, max_x, max_y)
                
                val = self.__get_tile_status_victim(min_x, min_y, max_x, max_y, fixture_array)
                
                row.append(list(val))
            grid.append(row)

        
        #print(grid)
        grid = self.__orientation_grid_to_final_fixture_grid(grid)
        #print("transform fuction")
        #print(grid)
        return grid
    


    def __orientation_grid_to_final_fixture_grid(self, orientation_grid: list) -> np.ndarray:
        shape = np.array([len(orientation_grid), len(orientation_grid[0])])
        shape *= 2

        final_wall_grid = np.empty(shape, dtype=object)
        final_wall_grid[:] = ""
        
        for y, row in enumerate(orientation_grid):
            for x, value in enumerate(row):
                x1 = x * 2
                y1 = y * 2

                for orientation in value:
                    if orientation:
                        if len(orientation) > 1:
                            final_x = x1 + int(orientation[1])
                        else:
                            final_x = x1
                        
                        if len(orientation) > 0:
                            final_y = y1 + int(orientation[0])
                        else:
                            final_y = y1

                        final_wall_grid[final_y, final_x] = orientation

        return final_wall_grid

    def __orientation_grid_to_final_wall_grid(self, orientation_grid: list) -> np.ndarray:
        shape = np.array([len(orientation_grid), len(orientation_grid[0])])
        shape *= 2

        final_wall_grid = np.zeros(shape, dtype=np.bool_)
        
        for y, row in enumerate(orientation_grid):
            for x, value in enumerate(row):
                x1 = x * 2
                y1 = y * 2

                for orientation in value:
                    final_x = x1 + orientation[1]
                    final_y = y1 + orientation[0]

                    final_wall_grid[final_y, final_x] = True
        
        return final_wall_grid
    
#    def victim_in_grid(self):
#        letter_position = FixtureDetector.get_fixture_positions_in_image("Y") * "ancho_de_matriz" + FixtureDetector.get_fixture_positions_in_image("X")
#        return letter_position


class FloorMatrixCreator:
    def __init__(self, square_size_px: int) -> None:
        self.contadorimage = 0
        self.contadorimagenes2 = 0
        self.__square_size_px = square_size_px * 2
        self.__floor_color_ranges = {

                    "0": # Normal
                        {   
                            "range":   ((0, 0, 38), (0, 0, 192)), 
                            "threshold":0.1},

                    "0": # Void
                        {
                            "range":((100, 0, 0), (101, 1, 1)),
                            "threshold":0.9},
                    
                    "4": # Checkpoint
                        {
                            "range":((113, 77, 62), (114, 84, 77)),
                            "threshold":0.01},

                    "2": # Hole
                        {
                            "range":((0, 0, 28), (0, 0, 38)),
                            "threshold":0.1},

                    "3": # swamp
                        {
                            "range":((19, 112, 38), (19, 141, 166)),
                            "threshold":0.3},

                    "b": # Connection 1-2
                        {
                            "range":((120, 182, 230), (120, 204, 232)),
                            "threshold":0.2},

                    "y": #Connection 1-3
                        {
                            "range":((30, 205, 233), (30, 205, 233)),
                            "threshold":0.2},
                        
                    "g": # Connection 1-4
                        {
                            "range":((58, 211, 221), (60, 228, 225)),
                            "threshold":0.3},
                    
                    "p": # Connection 2-3
                        {
                            "range":((128, 160, 172), (133, 192, 187)), #(63, 28, 72), (67, 30, 75)
                            "threshold":0.2},

                    "o": # Connection 2-4
                        {
                            "range":((22, 177, 221), (22, 205, 233)),
                            "threshold":0.2},


                    "r": # connection 3-4
                        {
                            "range":((0, 190, 213), (0, 205, 233)),
                            "threshold":0.3},
                    }
        
                    #TODO Add Connection 1-4
                    
        self.final_image = np.zeros((700, 700, 3), np.uint8)
        
    def __get_square_color(self, min_x, min_y, max_x, max_y, floor_array: np.ndarray) -> str:
        square = floor_array[min_x:max_x, min_y:max_y]

        square = cv.cvtColor(square, cv.COLOR_BGR2HSV)
        square_image = square.copy()
        """image_dir = "C:/Users/nacho/Documents/Programacion/webots_2023/RCJ-2024-Rescue-Simulation-Team-ABC/example/imageneswebots"
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        image_name = os.path.join(image_dir, f"IMAGENDESCONOCIDA{self.contadorimagenes2}_square.png")
        self.contadorimagenes2 += 1
        cv.imwrite(image_name, square_image)
        print(f"Square image saved as {image_name}")"""
        
        color_counts = {}
        for color_key, color_range in self.__floor_color_ranges.items():
            colour_count = np.count_nonzero(cv.inRange(square, color_range["range"][0], color_range["range"][1]))
            if colour_count > color_range["threshold"] * square.shape[0] * square.shape[1]:
                color_counts[color_key] = colour_count
        
        if len(color_counts) == 0:
            return "0"
        else:
            """dominant_color_label = max(color_counts, key=color_counts.get)
        
            #Save the square image
            image_dir = "C:/Users/nacho/Documents/Programacion/webots_2023/RCJ-2024-Rescue-Simulation-Team-ABC/example/imageneswebots"
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
            image_name = os.path.join(image_dir, f"{dominant_color_label}_{self.contadorimage}_square.png")
            self.contadorimage += 1
            cv.imwrite(image_name, square_image)
            print(f"Square image saved as {image_name}")
            
            return dominant_color_label"""
            return max(color_counts, key=color_counts.get)


    def get_floor_colors(self, floor_array: np.ndarray, offsets: np.ndarray) -> np.ndarray:

        if SHOW_MAP_AT_END:
            array_copy = copy.deepcopy(floor_array)

        grid = []

        for x in range(offsets[0], floor_array.shape[0] - self.__square_size_px, self.__square_size_px):
            row = []
            for y in range(offsets[1], floor_array.shape[1] - self.__square_size_px, self.__square_size_px):
                min_x = x
                min_y = y
                max_x = x + self.__square_size_px
                max_y = y + self.__square_size_px
                
                if SHOW_MAP_AT_END:
                    array_copy = cv.rectangle(array_copy, (min_y, min_x), (max_y, max_x), (255, 255, 255), 1)
                
                color_key = self.__get_square_color(min_x, min_y, max_x, max_y, floor_array)

                row.append(color_key)
            grid.append(row)

        if SHOW_MAP_AT_END:
            cv.imshow("array copy", array_copy)

        return grid
        

class FinalMatrixCreator:
    def __init__(self, tile_size: float, resolution: float):
        self.__square_size_px = round(tile_size / 2 * resolution)

        self.pre_matrix = pre_matrix(self.__square_size_px)
        self.wall_matrix_creator = WallMatrixCreator(self.__square_size_px)
        self.floor_matrix_creator = FloorMatrixCreator(self.__square_size_px)

    def stringMatriz(self, matriz):
        #matriz de string a int
        intmatriz = []

        for fila in matriz:
            aux = []
            for elemento in fila:
                a = int(elemento)
                aux.append(a)
            intmatriz.append(aux)

        for i in intmatriz:
            print(i)
        
        return intmatriz
    #def diferencia_mayor_a_10(coordenada1, coordenada2):
        #return abs(coordenada1[0] - coordenada2[0]) > 10 or abs(coordenada1[1] - coordenada2[1]) > 10

    #def punto_victim(self):
        #lista1 = mapper.Mapper.victim_position()
        #lista2 = []
        #indiceuno = 0
        #indicedos = 0
        #for i in range (len(lista1)):
            #for i in range (len(lista1)-1):
                #if self.diferencia_mayor_a_10(lista1[indiceuno], lista1[indicedos]) == True:
                    #lista2.append(lista1[indiceuno])
                    #indicedos += 1
        #indiceuno += 1

    def stringMatrizreverse(self, matriz):
        #matriz de int a str
            stringmatriz = []

            for fila in matriz:
                aux = []
                for elemento in fila:
                    a = str(elemento)
                    aux.append(a)
                stringmatriz.append(aux)

            for i in stringmatriz:
                print(i)
            
            return stringmatriz

    def delete_row(self, matriz_procesar):
    #Quita las filas que contengan valores innecesarios
        columnastotales = len(matriz_procesar[0])
        column_reference = ["0"]*columnastotales
        
    
        result = [elem for elem in matriz_procesar if elem != column_reference]
        return result
    

    def transposed_matriz2(self, matriz):
        nueva_matriz = [list(columnas) for columnas in zip(*matriz)]
        return nueva_matriz

    def correccion_de_bordes_filas(self, matriz):
        cant_c = (len(matriz[0]))
        columna = 0
        for i in range (0, cant_c):
            lmatrix = matriz[0][columna]
            if lmatrix != "0":
                columna += 1
            elif lmatrix == "0":        
                matriz[0][columna] = "1"
                columna += 1

        columna = 0
        for i in range (0, cant_c):
            lmatrix = matriz[-1][columna]
            if lmatrix != "0":
                columna += 1
            elif lmatrix == "0":        
                matriz[-1][columna] = "1"
                columna += 1
        columna = 0
        return matriz

    def correccion_de_bordes_columnas(self, matriz):
        matriz = [list(columnas) for columnas in zip(*matriz)]
        cant_c = (len(matriz[0]))
        columna = 0
        for i in range (0, cant_c):
            lmatrix = matriz[0][columna]
            if lmatrix != "0":
                columna += 1
            elif lmatrix == "0":        
                matriz[0][columna] = "1"
                columna += 1

        columna = 0
        for i in range (0, cant_c):
            lmatrix = matriz[-1][columna]
            if lmatrix != "0":
                columna += 1
            elif lmatrix == "0":        
                matriz[-1][columna] = "1"
                columna += 1
        columna = 0
        matriz = [list(columnas) for columnas in zip(*matriz)]
        return matriz

    def correccion_de_interioresA(self, matriz):
        #corrige casos en que haya un 0 incorrecto en medio de dos 1
        fila = 0
        columna = 0
        cant_f = (len(matriz))
        cant_c = (len(matriz[0]))
        for i in range (0, cant_f):
            for j in range (0, cant_c):
                lmatrix = matriz[fila][columna]
                if lmatrix != "0":
                    columna += 1
                elif lmatrix == "0":
                    try:
                        if ((matriz[fila][columna + 1]) == "1") and ((matriz[fila][columna - 1]) == "1"):
                            matriz[fila][columna] = "1"
                            columna += 1

                        elif ((matriz[fila + 1][columna]) == "1") and ((matriz[fila - 1][columna]) == "1"):
                            matriz[fila][columna] = "1"
                            columna += 1
                
                        else:
                            columna+= 1
                    except IndexError:
                            pass
            columna = 0
            fila += 1
        return matriz

    def correccion_de_interioresB(self, matriz):
        fila = 0
        columna = 0
        cant_f = len(matriz)
        cant_c = len(matriz[0])
        
        for i in range(cant_f):
            for j in range(cant_c):
                lmatrix = matriz[fila][columna]
                
                if lmatrix == "0":
                    if fila == 2 or fila == (cant_f - 2):
                        pass
                    else:
                        try:
                            if ((matriz[fila][columna + 1] == "1") and (matriz[fila][columna - 1] != "1") and (matriz[fila][columna + 2] == "1") and (matriz[fila + 1][columna] == "1") and (matriz[fila + 2][columna] == "1")): 
                                if (((matriz[fila - 1][columna + 4] != "1") and (matriz[fila + 2][columna + 4] != "1")) and \
                                ((matriz[fila + 4][columna - 1] != "1") and (matriz[fila + 4][columna - 2] != "1"))):
                                    
                                    matriz[fila][columna] = "1"
                        except IndexError:
                            pass

                        try:
                            if ((matriz[fila][columna - 1] == "1") and (matriz[fila][columna + 1] != "1") and (matriz[fila][columna - 2] == "1") and (matriz[fila + 1][columna] == "1") and (matriz[fila + 2][columna] == "1")):
                                if (((matriz[fila - 1][columna - 4] != "1") and (matriz[fila - 2][columna - 4] != "1")) and \
                                ((matriz[fila + 4][columna + 1] != "1") and (matriz[fila + 4][columna + 2] != "1"))):
                                    
                                    matriz[fila][columna] = "1"
                        except IndexError:
                            pass

                columna += 1

            columna = 0
            fila += 1

        return matriz


    def correccion_de_interioresC(self, matriz):
        #corrige vertices interiores hacia abajo
        fila = 0
        columna = 0
        cant_f = len(matriz)
        cant_c = len(matriz[0])
        
        for i in range(cant_f):
            for j in range(cant_c):
                lmatrix = matriz[fila][columna]
                
                if lmatrix == "0":
                    if fila == 1 or fila == (cant_f - 2):
                        pass
                    else:
                        try:
                            if ((matriz[fila][columna + 1] == "1") and (matriz[fila][columna - 1] != "1") and (matriz[fila][columna + 2] == "1") and (matriz[fila - 1][columna] == "1") and (matriz[fila - 2][columna] == "1")):
                                if (((matriz[fila + 1][columna + 4] != "1") and (matriz[fila + 2][columna + 4] != "1")) and \
                                ((matriz[fila - 4][columna - 1] != "1") and (matriz[fila - 4][columna - 2] != "1"))):
                                
                                    matriz[fila][columna] = "1"
                        except IndexError:
                            pass

                        try:
                            if ((matriz[fila][columna - 1] == "1") and (matriz[fila][columna + 1] != "1") and (matriz[fila][columna - 2] == "1") and (matriz[fila - 1][columna] == "1") and (matriz[fila - 2][columna] == "1")):
                                if (((matriz[fila + 1][columna-4] != "1") and (matriz[fila + 2][columna - 4] != "1")) and \
                                ((matriz[fila - 4][columna + 1] != "1") and (matriz[fila - 4][columna + 2] != "1"))):
                            
                                    matriz[fila][columna] = "1"
                        except IndexError:
                            pass

                columna += 1

            columna = 0
            fila += 1

        return matriz
    
    def correccion_de_interioresD(self, matriz):
        fila = 0
        columna = 0
        cant_f = len(matriz)
        cant_c = len(matriz[0])
        
        for i in range(cant_f):
            for j in range(cant_c):
                lmatrix = matriz[fila][columna]
                if lmatrix == "0":
                    if (fila == 1 or fila == (cant_f - 2)) or (columna == 1 or columna == (columna == (cant_c - 2))):
                        pass
                    else:
                        try:
                            if ((matriz[fila][columna + 1] == "1") and (matriz[fila][columna +2] == "1") and (matriz[fila][columna + 3] != "1")):
                                if ((matriz[fila - 1][columna] != "1") and (matriz[fila + 1][columna] != "1") and (matriz[fila - 1][columna + 3] != "1") and (matriz[fila + 1][columna + 3] != "1")):

                                    matriz[fila][columna] = "1"
                        except IndexError:
                            pass
                        try:
                            if ((matriz[fila][columna - 1] == "1") and (matriz[fila][columna - 2] == "1") and (matriz[fila][columna -3] != "1")) :
                                if ((matriz[fila - 1][columna] != "1") and (matriz[fila + 1][columna] != "1") and (matriz[fila - 1][columna - 3] != "1") and (matriz[fila + 1][columna - 3] != "1")):
                            
                                    matriz[fila][columna] = "1"
                        except IndexError:
                            pass

                        try:
                            if ((matriz[fila + 1][columna] == "1") and (matriz[fila + 2][columna] == "1") and (matriz[fila + 3][columna ] != "1")):
                                if ((matriz[fila][columna - 1] != "1") and (matriz[fila][columna + 1] != "1") and (matriz[fila + 3][columna - 1] != "1") and (matriz[fila + 3][columna + 1] != "1")):
                                    
                                    matriz[fila][columna] = "1"
                        except IndexError:
                            pass

                        try:
                            if ((matriz[fila - 1][columna] == "1") and (matriz[fila - 2][columna] == "1") and (matriz[fila - 3 ][columna] != "1")):
                                if ((matriz[fila][columna - 1] != "1") and (matriz[fila][columna + 1] != "1") and (matriz[fila - 3][columna - 1] != "1") and (matriz[fila - 3][columna + 1] != "1")):
                            
                                    matriz[fila][columna] = "1"
                        except IndexError:
                            pass

                columna += 1

            columna = 0
            fila += 1

    def baldoza_zona_4(self, matriz):
        fila = 0
        columna = 0
        cant_f = len(matriz)
        cant_c = len(matriz[0])
        
        for i in range(cant_f):
            for j in range(cant_c):
                lmatrix = matriz[fila][columna]
                if lmatrix == "0":
                    try:
                        if (((matriz[fila -1][columna - 1]) == ("r")) and ((matriz[fila -1][columna +1]) == ("r")) and ((matriz[fila +1][columna - 1]) == ("r")) and ((matriz[fila +1][columna + 1]) == ("r"))):
                            matriz[fila-1][columna-1] = "r"
                            matriz[fila-1][columna] = "r"
                            matriz[fila-1][columna+1] = "r"
                            matriz[fila][columna-1] = "r"
                            matriz[fila][columna+1] = "r"
                            matriz[fila+1][columna-1] = "r"
                            matriz[fila+1][columna] = "r"
                            matriz[fila+1][columna+1] = "r"

                        elif (((matriz[fila -1][columna - 1]) == ("g")) and ((matriz[fila -1][columna +1]) == ("g")) and ((matriz[fila +1][columna - 1]) == ("g")) and ((matriz[fila +1][columna + 1]) == ("g"))):
                            matriz[fila-1][columna-1] = "g"
                            matriz[fila-1][columna] = "g"
                            matriz[fila-1][columna+1] = "g"
                            matriz[fila][columna-1] = "g"
                            matriz[fila][columna+1] = "g"
                            matriz[fila+1][columna-1] = "g"
                            matriz[fila+1][columna] = "g"
                            matriz[fila+1][columna+1] = "g"
                        elif (((matriz[fila -1][columna - 1]) == ("o")) and ((matriz[fila -1][columna +1]) == ("o")) and ((matriz[fila +1][columna - 1]) == ("o")) and ((matriz[fila +1][columna + 1]) == ("o"))):
                            matriz[fila-1][columna-1] = "o"
                            matriz[fila-1][columna] = "o"
                            matriz[fila-1][columna+1] = "o"
                            matriz[fila][columna-1] = "o"
                            matriz[fila][columna+1] = "o"
                            matriz[fila+1][columna-1] = "o"
                            matriz[fila+1][columna] = "o"
                            matriz[fila+1][columna+1] = "o"
                    except IndexError:
                            pass
                columna += 1
               

            columna = 0
            fila += 1

        return matriz


    def direccion_de_zona4(self, matriz):
        
        fila = 0
        columna = 0
        cant_f = len(matriz)
        cant_c = len(matriz[0])
        contador = 0
            
        for i in range(cant_f):
            for j in range(cant_c):
                lmatrix = matriz[fila][columna]
                if lmatrix == "0":
                    try:
                        if ((((matriz[fila -1][columna - 1]) == ("r")) and ((matriz[fila -1][columna +1]) == ("r")) and ((matriz[fila +1][columna - 1]) == ("r")) and ((matriz[fila +1][columna + 1]) == ("r"))) or (((matriz[fila -1][columna - 1]) == ("g")) and ((matriz[fila -1][columna +1]) == ("g")) and ((matriz[fila +1][columna - 1]) == ("g")) and ((matriz[fila +1][columna + 1]) == ("g"))) or (((matriz[fila -1][columna - 1]) == ("o")) and ((matriz[fila -1][columna +1]) == ("o")) and ((matriz[fila +1][columna - 1]) == ("o")) and ((matriz[fila +1][columna + 1]) == ("o")))):  
                            if contador == 0:
                                fila_fija = fila
                                columna_fija = columna
                                primera_baldosaC = columna
                                contador += 1
                            elif contador == 1:
                                segunda_baldosaC = columna
                                contador+=1
                            else:
                                pass 


                    except IndexError:
                            pass
                        
                                    
                columna +=1

            columna = 0
            fila += 1
        if contador == 2:
            dist_entre_columnas = segunda_baldosaC - primera_baldosaC

            if dist_entre_columnas > 0:         #segunda columna mayor que la primera (avanzar hacia la derecha tomando como base la primera)
                direccion = "derecha"
                return direccion, fila_fija, columna_fija

                                                    

            else:
                direccion = "izquierda"
                return direccion, fila_fija, columna_fija      #primera columna mayor que la segunda (avanzar hacia la izquierda tomando como base la primera)
            
        elif contador == 1:
                direccion = "indefinido"
                return direccion, fila_fija, columna_fija
        

    def al_lado_1(self, matrix, row, col, directions):
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < len(matrix) and 0 <= c < len(matrix[0]) and matrix[r][c] == '1':
                return True
        return False

    def bfszone4(self, matrix, direccion, fila_inicio, columna_inicio):
        matrixinicio = copy.deepcopy(matrix)  # Crear una copia profunda de la matriz original
        rows = len(matrix)
        cols = len(matrix[0])
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        start_row = fila_inicio + 3
        start_col = columna_inicio
        state = 0

        if direccion == "izquierda":
            start_col = start_col - 1
            state = "normal"
        elif direccion == "derecha":
            start_col = start_col + 1
            state = "normal"
        else:
            state = "indefinido"
            
        if state == "normal":
            matrixprueba = copy.deepcopy(matrixinicio)  # Crear una copia profunda para evitar modificar matrixinicio
            queue = deque([(start_row, start_col)])
            visited = set()
            visited.add((start_row, start_col))

            while queue:
                row, col = queue.popleft()
                
                # Check and mark the current cell if it is '0'
                if matrixprueba[row][col] == '0':
                    matrixprueba[row][col] = '*'

                # Traverse adjacent cells
                for dr, dc in directions:
                    r, c = row + dr, col + dc
                    if 0 <= r < rows and 0 <= c < cols and (r, c) not in visited and matrixprueba[r][c] == '0':
                        if not self.al_lado_1(matrixprueba, r, c, directions):
                            queue.append((r, c))
                            visited.add((r, c))
            
            return matrixprueba
        
        else:
            if (rows - fila_inicio) < 6:
                start_row = start_row - 6
                start_col = start_col + 1
                matrixprueba = copy.deepcopy(matrixinicio)
                queue = deque([(start_row, start_col)])
                visited = set()
                visited.add((start_row, start_col))

                while queue:
                    row, col = queue.popleft()
                    
                    if matrixprueba[row][col] == '0':
                        matrixprueba[row][col] = '*'

                    for dr, dc in directions:
                        r, c = row + dr, col + dc
                        if 0 <= r < rows and 0 <= c < cols and (r, c) not in visited and matrixprueba[r][c] == '0':
                            if not self.al_lado_1(matrixprueba, r, c, directions):
                                queue.append((r, c))
                                visited.add((r, c))

                porcentage = self.calculate_percentage(matrixprueba)
                if porcentage < 50:
                    return matrixprueba
                
                else:
                    start_row = start_row + 6
                    start_col = start_col - 1
                    matrixprueba = copy.deepcopy(matrixinicio)
                    queue = deque([(start_row, start_col)])
                    visited = set()
                    visited.add((start_row, start_col))

                    while queue:
                        row, col = queue.popleft()
                        
                        # Check and mark the current cell if it is '0'
                        if matrixprueba[row][col] == '0':
                            matrixprueba[row][col] = '*'

                        # Traverse adjacent cells
                        for dr, dc in directions:
                            r, c = row + dr, col + dc
                            if 0 <= r < rows and 0 <= c < cols and (r, c) not in visited and matrixprueba[r][c] == '0':
                                if not self.al_lado_1(matrixprueba, r, c, directions):
                                    queue.append((r, c))
                                    visited.add((r, c))
                    porcentage = self.calculate_percentage(matrixprueba)
                    if porcentage < 50:
                        return matrixprueba
                    else:
                        return matrixinicio
            else:
                start_row = start_row
                start_col = start_col + 1
                matrixprueba = copy.deepcopy(matrixinicio)
                queue = deque([(start_row, start_col)])
                visited = set()
                visited.add((start_row, start_col))
                while queue:
                    row, col = queue.popleft()
                    
                    # Check and mark the current cell if it is '0'
                    if matrixprueba[row][col] == '0':
                        matrixprueba[row][col] = '*'
                    
                    # Traverse adjacent cells
                    for dr, dc in directions:
                        r, c = row + dr, col + dc
                        if 0 <= r < rows and 0 <= c < cols and (r, c) not in visited and matrixprueba[r][c] == '0':
                            if not self.al_lado_1(matrixprueba, r, c, directions):
                                queue.append((r, c))
                                visited.add((r, c))

                porcentage = self.calculate_percentage(matrixprueba)
                if porcentage < 50:
                    return matrixprueba
                
                else:
                    start_row = start_row + 6
                    start_col = start_col - 1
                    matrixprueba = copy.deepcopy(matrixinicio)
                    queue = deque([(start_row, start_col)])
                    visited = set()
                    visited.add((start_row, start_col))

                    while queue:
                        row, col = queue.popleft()
                        
                        # Check and mark the current cell if it is '0'
                        if matrixprueba[row][col] == '0':
                            matrixprueba[row][col] = '*'

                        # Traverse adjacent cells
                        for dr, dc in directions:
                            r, c = row + dr, col + dc
                            if 0 <= r < rows and 0 <= c < cols and (r, c) not in visited and matrixprueba[r][c] == '0':
                                if not self.al_lado_1(matrixprueba, r, c, directions):
                                    queue.append((r, c))
                                    visited.add((r, c))
                    porcentage = self.calculate_percentage(matrixprueba)
                    if porcentage < 50:
                        return matrixprueba
                    else:
                        return matrixinicio

    def calculate_percentage(self, matrix):
        total_elements = 0
        contador = 0

        for row in matrix:
            for element in row:
                total_elements += 1
                if element == "*":
                    contador += 1

        if total_elements == 0:
            return 0 
        
        percentage = (contador / total_elements) * 100
        return percentage

    def expandzone4(self, matrix):
        
        rows = len(matrix)
        cols = len(matrix[0])
        
        new_matrix = [row.copy() for row in matrix]
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        for i in range(rows):
            for j in range(cols):
                if matrix[i][j] == '*':
                    for di, dj in directions:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols: 
                            if new_matrix[ni][nj] not in ('1', 'r', 'g', 'o', '*'):
                                new_matrix[ni][nj] = '*'
        
        return new_matrix
    
    def expandzone4_v2(self, matrix):
        
        rows = len(matrix)
        cols = len(matrix[0])
        
        new_matrix = [row.copy() for row in matrix]
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        for i in range(rows):
            for j in range(cols):
                if matrix[i][j] == '*':
                    for di, dj in directions:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols: 
                            if new_matrix[ni][nj] not in ('0', 'r', 'g', 'o', '*'):
                                new_matrix[ni][nj] = '*'
        
        return new_matrix


    def combinar_matriz(self, matriz_original, matriz_prueba):
        filas = len(matriz_original)
        columnas = len(matriz_original[0])
        
        for i in range(filas):
            for j in range(columnas):
                if matriz_prueba[i][j] == "*":
                    matriz_original[i][j] = "*"
        
        return matriz_original
    
    def baldozazona4_correccion(self, matriz):
        fila = 0
        columna = 0
        cant_f = len(matriz)
        cant_c = len(matriz[0])
            
        for i in range(cant_f):
            for j in range(cant_c):
                lmatrix = matriz[fila][columna]
                if lmatrix == "0":
                    try:
                        if ((((matriz[fila -1][columna - 1]) == ("r")) and ((matriz[fila -1][columna +1]) == ("r")) and ((matriz[fila +1][columna - 1]) == ("r")) and ((matriz[fila +1][columna + 1]) == ("r"))) or (((matriz[fila -1][columna - 1]) == ("g")) and ((matriz[fila -1][columna +1]) == ("g")) and ((matriz[fila +1][columna - 1]) == ("g")) and ((matriz[fila +1][columna + 1]) == ("g"))) or (((matriz[fila -1][columna - 1]) == ("o")) and ((matriz[fila -1][columna +1]) == ("o")) and ((matriz[fila +1][columna - 1]) == ("o")) and ((matriz[fila +1][columna + 1]) == ("o")))):  
                                matriz[fila-1][columna] = "0"
                                matriz[fila][columna-1] = "0"
                                matriz[fila][columna+1] = "0"
                                matriz[fila+1][columna] = "0"
                        

                    except IndexError:
                            pass
                        
                                    
                columna +=1

            columna = 0
            fila += 1
        return matriz

    def pixel_grid_to_final_grid(self, pixel_grid: CompoundExpandablePixelGrid, robot_start_position: np.ndarray, victimas, timerun) -> np.ndarray: #pasar parametro de victimas
        np.set_printoptions(linewidth=1000000000000, threshold=100000000000000)
        #linewidth: ancho maximo de impresion
        #threshold: limite de elementos que se imprimen
        wall_array = pixel_grid.arrays["walls"]
        color_array = pixel_grid.arrays["floor_color"]
        victims_array = pixel_grid.arrays["victims"]
        victims_type_array = pixel_grid.arrays["victims_type"]
        fixture_array = pixel_grid.arrays["robot_detected_fixture_from"]

        if DO_SAVE_FINAL_MAP:
            cv.imwrite(f"{SAVE_FINAL_MAP_DIR}/WALL_PIXEL_GRID{str(time.time()).rjust(50)}.png", wall_array.astype(np.uint8) * 255)

        if DO_SAVE_DEBUG_GRID:
            cv.imwrite(f"{SAVE_DEBUG_GRID_DIR}/DEBUG_GRID{str(time.time()).rjust(50)}.png", (pixel_grid.get_colored_grid() * 255).astype(np.uint8))

                
        # Walls
        #wall_node_array = self.wall_matrix_creator.transform_wall_array_to_bool_node_array(wall_array, offsets)
        #print("----------- PRECARGA DE PAREDES -----------")
        
        pre_walls = self.pre_matrix.preload_matrix(wall_array)

        # Victim
        #print("----------- PRECARGA DE VICTIMAS -----------")
        pre_victims = self.pre_matrix.preload_matrix(victims_array)
        #print("first pre victims")
        #print(pre_victims)
        pre_victims = self.pre_matrix.correct_preload_victim(pre_victims)
        #print("second pre victims")
        #print(pre_victims)
        pre_victims = self.pre_matrix.removing_incorrect_True(pre_victims)
        #print("third pre victims")
        #print(pre_victims)

        #print("----------- MATRIZ FINAL PRECARGADA -----------")
        new = self.pre_matrix.preload_final_matrix(pre_walls, pre_victims)
        #print("first new")
        #print(new)
        #new = self.pre_matrix.preload_final_matrix2(pre_walls, pre_victims)
        #print("second new")
        #print(new)

        #print(victims_type_array)

        #print("SEPARACION")

        #print(fixture_array)
        
        # Walls & Victims
        offsets = self.__get_offsets(self.__square_size_px, pixel_grid.offsets)
        wall_node_array = self.wall_matrix_creator.transform_wall_array_to_bool_node_array(new, offsets)
        robot_detected_array = self.wall_matrix_creator.transform_robot_detected_to_string_node_array(fixture_array, offsets)
        #print(robot_detected_array)



        # Floor
        floor_offsets = self.__get_offsets(self.__square_size_px * 2, pixel_grid.offsets + self.__square_size_px)
        floor_string_array = self.floor_matrix_creator.get_floor_colors(color_array, floor_offsets)

        #prueba:
        #new_array = self

        # Start tile
        if robot_start_position is None:
            return np.array([])

        """offsets = self.__get_offsets(self.__square_size_px, pixel_grid.offsets)
        
        # Walls
        wall_node_array = self.wall_matrix_creator.transform_wall_array_to_bool_node_array(wall_array, offsets)


        floor_offsets = self.__get_offsets(self.__square_size_px * 2, pixel_grid.offsets + self.__square_size_px)

        # Floor
        floor_string_array = self.floor_matrix_creator.get_floor_colors(color_array, floor_offsets)

        # Start tile
        if robot_start_position is None:
            return np.array([])
        """
        start_array_index = pixel_grid.coordinates_to_array_index(robot_start_position)
        start_array_index -= offsets
        robot_node = np.round((start_array_index / self.__square_size_px) * 2).astype(int) - 1


        # Mix everything togehter
        #matrix_walls = pixel_grid.matrix_to_arrays((pixel_grid.arrays["walls"]))
        #matrix_victims = pixel_grid.matrix_to_arrays((pixel_grid.arrays["victims"]))
        
        #print(pixel_grid.arrays["walls"])
        #print("SEPARACION")
        #print(pixel_grid.arrays["victims"])
        vict_grid = self.__get_victims_text_grid(robot_detected_array, victimas)
        #print("acaaaa")
        #print(vict_grid)
        text_grid = self.__get_final_text_grid(wall_node_array, floor_string_array, robot_node)
        text_grid = self.unificador_de_matrices(vict_grid, text_grid)
        print("el programa")
        if timerun == False:
            text_grid = self.delete_row(text_grid)
            text_grid = self.transposed_matriz2(text_grid)
            text_grid = self.delete_row(text_grid)
            text_grid = self.transposed_matriz2(text_grid)
            text_grid = self.correccion_de_interioresA(text_grid)
            text_grid_zone4 = text_grid
            text_grid_zone4 = self.baldoza_zona_4(text_grid_zone4)
            datos_importantes = self.direccion_de_zona4(text_grid_zone4)
            direccion_zona4 = datos_importantes[0]
            fila_fija_zona4 = datos_importantes[1]
            columna_fija_zona4 = datos_importantes[2]
            text_grid_zone4 = self.bfszone4(text_grid_zone4, direccion_zona4, fila_fija_zona4, columna_fija_zona4)
            text_grid_zone4 = self.expandzone4(text_grid_zone4)
            text_grid_zone4 = self.expandzone4_v2(text_grid_zone4)
            text_grid = self.combinar_matriz(text_grid, text_grid_zone4)
            text_grid = self.baldozazona4_correccion(text_grid)
        """text_grid = self.correccion_de_bordes_filas(text_grid)
        text_grid = self.correccion_de_bordes_columnas(text_grid)
        text_grid = self.correccion_de_interioresA(text_grid)
        text_grid = self.correccion_de_interioresB(text_grid)
        text_grid = self.correccion_de_interioresC(text_grid)
        text_grid = self.correccion_de_interioresD(text_grid)"""
        return np.array(text_grid)
        

        #wall_array = self.offset_array(wall_array, self.square_size_px, pixel_grid.offsets)
        #color_array = self.offset_array(color_array, self.square_size_px, pixel_grid.offsets)

    def __get_final_text_grid(self, wall_node_array: np.ndarray, floor_type_array: np.ndarray, robot_node: np.ndarray) -> list:
        if SHOW_MAP_AT_END:
            cv.imshow("final_grid", cv.resize(wall_node_array.astype(np.uint8) * 255, (0, 0), fx=10, fy=10, interpolation=cv.INTER_AREA))


        if DO_SAVE_FINAL_MAP:
            cv.imwrite(f"{SAVE_FINAL_MAP_DIR}/WALL_GRID{str(time.time()).rjust(50)}.png", wall_node_array.astype(np.uint8) * 255)

        
        final_text_grid = []

        # set walls
        for row in wall_node_array:
            f_row = []
            for val in row:
                if val:
                    f_row.append("1")
                else:
                    f_row.append("0")
            final_text_grid.append(f_row)

#        if WallMatrixCreator.victim_in_grid():
#            if FixtureClasiffier.classify_fixture == "U":
#                f_row.append("U")
#            if FixtureClasiffier.classify_fixture == "S":
#                f_row.append("S")
#            if FixtureClasiffier.classify_fixture == "H":
#                f_row.append("H")
#            final_text_grid.append(f_row)

        #set floor
        for y, row in enumerate(floor_type_array):
            for x, val in enumerate(row):
                x1 = x * 4 + 3
                y1 = y * 4 + 3
                self.__set_node_as_character(final_text_grid, np.array([y1, x1]), val)

        
        self.__set_node_as_character(final_text_grid, robot_node, "5")
        return final_text_grid
    
    def __get_victims_text_grid(self, robot_detected_array: np.ndarray, executorvariable ) -> list:       #executorvariable = executor.Executor

        victims_grid = []

        # set walls
        for row in robot_detected_array:
            f_row = []
            for val in row:
                if val == (""):
                    f_row.append("0")
                else:
                    new_letter = executorvariable[int(val)]
                    #print(executorvariable)
                    #f_row.append(val)
                    print(new_letter)
                    f_row.append(new_letter)   #new_letter
            victims_grid.append(f_row)

        return victims_grid
        
    
    def __get_offsets(self, square_size: float, raw_offsets: np.array) -> np.ndarray:
        return np.round(raw_offsets % square_size).astype(int)
    

    def __set_node_as_character(self, final_text_grid: list, node: np.ndarray, character: str) -> list:
        for diagonal in np.array(((1, 1), (-1, 1), (-1, -1), (1, -1))):
            n = node + diagonal
            try:
                final_text_grid[n[0]][n[1]] = character
            except IndexError:
                pass

        return final_text_grid

    def unificador_de_matrices(self, victim_matrix, original_matrix):
        fila = 0
        columna = 0

        cant_f = len(victim_matrix)
        cant_c = len(victim_matrix[0])
        for i in range(cant_f):
            for j in range(cant_c):
                #print(f"fila: {fila} y columna {columna}")
                lmatrix = victim_matrix[fila][columna]
                if lmatrix != ("0"):
                    #print(f"fila: {fila} y columna {columna}")
                    #print(lmatrix)
                    fila2 = (-3)
                    columna2 = (-3)
                    for a in range (7):
                        for b in range (7):
                            #print(fila2, columna2)   
                            positon_matrix = original_matrix[fila + fila2][columna + columna2]
                            #print(f"fila2 {[fila + fila2]} y columna2{[columna + columna2]}")
                            if positon_matrix == ("0"):
                                #print(columna2)
                                #print(f"fila -1 = {(fila + fila2) - 1} y fila +1 {(fila + fila2) + 1}")
                                try:
                                    if (((original_matrix[(fila + fila2) - 1][columna + columna2]) == ("1")) and ((original_matrix[(fila + fila2) + 1][columna + columna2]) == ("1"))): #or (((original_matrix[(fila + fila2) - 2][columna + columna2]) == ("1")) and ((original_matrix[(fila + fila2) + 1][columna + columna2]) == ("1"))) or (((original_matrix[(fila + fila2) - 1][columna + columna2]) == ("1")) and ((original_matrix[(fila + fila2) + 22][columna + columna2]) == ("1"))):
                                        #print("entrooo")
                                        #print(f"fila2 {[fila + fila2]} y columna2{[columna + columna2]}")
                                        original_matrix[fila + fila2][columna + columna2] = lmatrix
                                        columna2 += 1
                                except IndexError:
                                    pass

                                try:
                                    if (((original_matrix[fila + fila2][(columna + columna2) - 1]) == ("1")) and ((original_matrix[fila + fila2][(columna + columna2) + 1]) == ("1"))): #or (((original_matrix[fila + fila2][(columna + columna2) - 2]) == ("1")) and ((original_matrix[fila + fila2][(columna + columna2) + 1])) or (((original_matrix[fila + fila2][(columna + columna2) - 1]) == ("1")) and ((original_matrix[fila + fila2][(columna + columna2) + 2])))):
                                        #print("entrooo2")
                                        #print(f"fila2 {[fila + fila2]} y columna2{[columna + columna2]}")
                                        original_matrix[fila + fila2][columna + columna2] = lmatrix 
                                        columna2 += 1
                                except IndexError:
                                    pass
                                columna2 += 1
                            else:
                                columna2 += 1
                        columna2 = -3
                        fila2 += 1
                    columna += 1
                else:
                    columna += 1
            columna = 0
            fila +=1

        return original_matrix
