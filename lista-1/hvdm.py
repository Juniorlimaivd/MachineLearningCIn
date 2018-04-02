def HVDM(instance1, instance2, classDict, attrFrequecy, attrFrequecyByClass, q):
    if len(instance1) != len(instance2):
        raise ValueError

    distance = 0
    for i in len(instance1):

        if type(instance1[i]) is str:
            partialsum = 0

            for j in classDict:
                partialsum += pow(
                    attrFrequecyByClass[j][instance1[i]] /
                    attrFrequecy[instance1[i]] - attrFrequecyByClass[j]
                    [instance2[i]] / attrFrequecy[instance2[i]], q)

            distance += partialsum
        else:
            distance += pow(instance1[i] - instance2[i], 2)
