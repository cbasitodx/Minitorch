from minitorch.engine.Autograd import Value
from typing import List


#TODO
'''
ESTO ACEPTA UNA LISTA DE FLOATS O UNA LISTA DE INTS. TAMBIEN
ACEPTA UNA LISTA DE VALUES, PERO ESO NO LO DEBERIA USAR EL PUBLICO GENERAL
(SOLO PARA CALCULOS INTERMEDIOS, PARA NO PERDER EL ARBOL DE COMPUTOS)
'''
class Tensor:
    def __init__(self, 
                 data : int | float | Value | List | List[List] | List[List[List]]): # TODO: ESTOS SON LISTAS DE LISTAS DE NUMEROS O VALUES O NUMEROS

        self.shape : List[int]
        
        # TODO: PONER QUE ASUMIMOS QUE ASUMIMOS UNIFORMIDAD A LO LARGO DE TODAS LAS DIMENSIONES
        def get_dims(data : List):
            if type(data) != list:
                return []
            else:
                return [len(data)] + get_dims(data[0]) 
        
        # TODO: PONER ERROR SI LA SHAPE ES >= 3 O SI NO SE INTRODUJO UN VALOR ENTONCES ERROR
        self.shape = get_dims(data)

        self.__data : Value | List[Value] | List[List[Value]] | List[List[List[Value]]]

        def get_value_list(data : List):
            if isinstance(data, list):
                return [get_value_list(item) for item in data]
            elif isinstance(data, (float,int)): 
                return Value(data)
            elif isinstance(data, Value):
                return data
            # TODO PONER ERRORES AQUI!!! (EL 'else')
        
        self.__data = get_value_list(data)
        
    def __iter__(self):
        return iter(self.__data)
    
    def __mul__(self, other : 'Tensor') -> 'Tensor':

        res : Tensor

        # Scalar multiplication
        if len(self.shape) == 0:
            res = Tensor(self.__data * other.__data)
        
        # Vector or matrix multiplication
        else:
            # TODO: SI N° COLS DE SELF != N° FILAS DE OTHER ENTONCES NO!!!
            # TODO: PONER COMPROBACION DE QUE SI SHAPE > 2 ENTONCES NO
            self_rows = 1 if len(self.shape) == 1 else self.shape[0]

            res = Tensor([[Value(0) for _ in range(other.shape[1])]for _ in range(self_rows)])

            for self_row in range(self_rows):
                for other_col in range(other.shape[1]):
                    for other_row in range(other.shape[0]):

                        if len(self.shape) == 1:
                            res.__data[self_row][other_col] += self.__data[other_row] * other.__data[other_row][other_col]
                        else:
                            res.__data[self_row][other_col] += self.__data[self_row][other_row] * other.__data[other_row][other_col]

            # TODO: ESTO ES PARA DROPEAR DIMENSIONES TIPO EN [[1]] -> [1]
            while len(res.shape) != 1 and res.shape[0] == 1:
                res = res[0]

        return res
    
    def __add__(self, other : 'Tensor') -> 'Tensor':
        
        # TODO: PONER COMPROBACION DE QUE SI SHAPE DE CUALQUIERA DE LOS DOS
        #       ES MAYOR A 2, ENTONCES NO !!! (ES UN ELSE Y UNA COMPROBACION INICIAL DE Q COINCIDEN LAS DIMS)

        res : Tensor

        # Scalar addition
        if len(self.shape) == 0:
            res = Tensor(self.__data + other.__data)

        # Vector addition
        if len(self.shape) == 1:
            res = Tensor([val_1 + val_2 for val_1, val_2 in zip(self.__data, other.__data)])

        # Matrix addition
        elif len(self.shape) == 2:
            res = Tensor([[Value(0) for _ in range(self.shape[1])] for _ in range(self.shape[0])])

            for row in range(self.shape[0]):
                for col in range(self.shape[1]):
                    res[row][col].__data = self.__data[row][col] + other.__data[row][col]
        
        return res

    def __repr__(self):
        return f"Tensor({self.__data})"
    
    def __getitem__(self, index):
        return Tensor(self.__data[index])
    
    def __len__(self):
        return len(self.__data)
    
    def transpose(self):
        # TODO: PONER AQUI QUE NO SE PUEDE PARA TENSORES QUE NO TENGAN DIMENSION 2 (SHAPE != 2 => ERROR)

        new_data = [list(row) for row in zip(*self.__data)]

        return Tensor(new_data)
    
    # TODO: QUITAR ESTO. ES SOLO PARA ENTRENAR A LA RED EASY CUANDO DEFINO EL MSE
    def get_data(self):
        return self.__data