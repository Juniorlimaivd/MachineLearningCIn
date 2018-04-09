import math

def is_number(n):
    try:
        float(n)   # Type-casting the string to `float`.
                   # If string is not a valid `float`, 
                   # it'll raise `ValueError` exception
    except ValueError:
        return False
    return True

def HVDM(instance1, instance2, classDict, attrFrequency, attrFrequencyByClass, ranges, q):
    if len(instance1) != len(instance2):
        raise ValueError
#
    distance = 0
    for i in range(len(instance1) - 1):
        
        if isinstance(instance1[i],str):
            partialsum = 0          
                
            for j in classDict.keys():
                if not (j, i, instance1[i]) in attrFrequencyByClass:
                    attrFrequencyByClass[(j, i, instance1[i])] = 0
                if not (j, i, instance2[i]) in attrFrequencyByClass:
                    attrFrequencyByClass[(j, i, instance2[i])] = 0

                partialsum += pow(abs((float(attrFrequencyByClass[(j, i, instance1[i])]) / float(attrFrequency[(i, instance1[i])])) - (float(attrFrequencyByClass[(j, i, instance2[i])]) / float(attrFrequency[(i, instance2[i])]))), q)
            distance += pow(partialsum, 2)
        else:
            distance += pow( abs(float(instance1[i]) - float(instance2[i])) / float(ranges[i]), 2)

    return math.sqrt(distance)
