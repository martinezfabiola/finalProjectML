def moda(filename):
    dataset = open(filename, "r")
    ident = []

    lines=dataset.readlines()

    nombre = " "
    #k=1
    t=0
    f=0
    #aux=0
    for line in lines:
        #if ((nombre != line.split(',')[0]) and (nombre[0:(len(line.split(',')[0])-1)]!=line.split(',')[0][0:(len(line.split(',')[0])-1)])):
        if ((nombre != line.split(',')[0])):
            if (t>=f):
                ident.append([nombre, 1])
            if (t<f):
                ident.append([nombre, 0])
            t = 0
            f = 0
            nombre = line.split(',')[0]
            #print(line.split(',')[1])

        if (line.split(',')[1].rstrip()=="FALSE"):
            f+=1
        if (line.split(',')[1].rstrip()=="TRUE"):
            t+=1
    dataset.close()
    return ident

def enfermedades(filename):
    dataset = open(filename, "r")
    ident = []

    lines=dataset.readlines()

    nombre = " "
    #k=1
    ovder=0
    ovizq=0
    utero=0
    #aux=0
    for line in lines:
        #if ((nombre != line.split(',')[0]) and (nombre[0:(len(line.split(',')[0])-1)]!=line.split(',')[0][0:(len(line.split(',')[0])-1)])):
        if ((nombre != line.split(',')[0])):
            ident.append([nombre, ovder, ovizq, utero])
            ovder=0
            ovizq=0
            utero=0
            nombre = line.split(',')[0]

        if (line.split(',')[2].rstrip()!=""):
            ovder=1
        if (line.split(',')[3].rstrip()!=""):
            ovizq=1
        if (line.split(',')[4].rstrip()!=""):
            utero=1

    dataset.close()
    return ident

datos = enfermedades('PregnancyRate.csv')
archivo = open('Vacadata_enfermedades.csv', 'w')
archivo.write("ID, ovder, ovizq, utero\n")
for i in range(2,len(datos)):
    archivo.write(datos[i][0] + ", " + str(datos[i][1])+ ", " + str(datos[i][2])+ ", " + str(datos[i][3]) + "\n") 
#datos = moda('Y_sin_espacios.csv')
#archivo = open("Vacadata_Y.csv", "w")
#archivo.write("ID, Y\n")
#for i in range(2,len(datos)):
    #archivo.write(datos[i][0] + ", " + str(datos[i][1])+"\n")