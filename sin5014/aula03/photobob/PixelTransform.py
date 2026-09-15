from abc import ABC, abstractmethod

# 1. Definição da Interface
class PixelTransform(ABC):
    
    @abstractmethod
    def transformar(self, mensagem: str) -> bool:
        """Este método deve ser obrigatoriamente implementado."""
        pass