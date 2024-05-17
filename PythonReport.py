import pandas as pd
import matplotlib.pyplot as plt

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4



#Abrir excel
data = pd.read_excel('CoffeeMachineData.xlsx',sheet_name='Sheet1')

# Transformar datos a listas, se extraen las columnas
MachineName = data['Machine Name'].tolist()
MachinePrices = data['Machine Price'].tolist()
MachineTypes = data['Type'].tolist()
MachineSpecialFeatures = data['Special Feature'].tolist()

# Ver cuales son los productos que se tienen
ListaProductos = []
CantidadProdutos = []
for i in MachineName:
    if i not in ListaProductos:
        ListaProductos.append(i)

# Ver cuántos de cada uno se tienen
for i in ListaProductos:
    CantidadProdutos.append(MachineName.count(i))


# Ver cuantos y cuales tipos se compra más
ListaTipos = []
CantidadTipos = []

for i in MachineTypes:
    if i not in ListaTipos:
        ListaTipos.append(i)

for i in ListaTipos:
    CantidadTipos.append(MachineTypes.count(i))
    
## Graficar con Matplotlib los resúmenes
# Productos
fig1, ax1 = plt.subplots()
ax1.bar(ListaProductos,CantidadProdutos)
plt.xticks(rotation = 90)
plt.xlabel('Productos vendidos')
plt.ylabel('Cantidad de productos vendidos')

#tipos de productos
fig2, ax2 = plt.subplots()
ax2.pie(CantidadTipos, labels=ListaTipos)

plt.show()

# Crear pdf 
c = canvas.Canvas("reportlab_pdf.pdf",pagesize=letter)
width,height = letter

c.setFont("Times New roman",14)
c.drawString(0,0,"Hello World")


c.showPage()
c.save()