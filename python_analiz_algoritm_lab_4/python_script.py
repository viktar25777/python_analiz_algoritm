import logging
import graphviz
from graphviz import Digraph
dot = Digraph(node_attr={'shape': 'box'})
dot.node('A', label='Клиент старше 18 лет?')
dot.node('B', label='Превышает ли его зароботок 50 тысяч рублей?')
dot.node('C', label='Отказать')
dot.node('D', label='Были ли у клиента просроченные кредиты ранее?')
dot.node('E', label='Отказать')
dot.node('F', label='Отказать')
dot.node('G', label='Выдать')
dot.edge('A', 'B', label='да')
dot.edge('A', 'C', label='нет')
dot.edge('B', 'D', label='да')
dot.edge('B', 'E', label='нет')
dot.edge('D', 'F', label='да')
dot.edge('D', 'G', label='нет')
print(dot)

