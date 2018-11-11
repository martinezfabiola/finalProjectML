def read_dataset(filename):
    dataset = open(filename, "r")
    ident = []

    lines=dataset.readlines()
    rate = 0.0

    nombre = " "
    k=1
    aux=0
    for line in lines:
        #if ((nombre != line.split(',')[0]) and (nombre[0:(len(line.split(',')[0])-1)]!=line.split(',')[0][0:(len(line.split(',')[0])-1)])):
        if ((nombre != line.split(',')[0])):
            if (aux<1):
                aux += 1
                continue
            if aux>=1: 
                ident.append([nombre, rate/k])
            rate = 0.0
            k = 0
            nombre = line.split(',')[0]

        rate += float(line.split(',')[1])
        k += 1
    dataset.close()
    return ident

datos = read_dataset('salida_dim.csv')
archivo = open("Vacadata_DIM.csv", "w")
archivo.write("ID, Preg. Rate\n")
for i in range(1,len(datos)):
    print(datos[i][0],datos[i][1])
    archivo.write(datos[i][0] + ", " + str(datos[i][1])+"\n")