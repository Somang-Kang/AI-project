import math


def oneLine(lines):
    xLines = [0]*26
    yLines = [0]*26
    a = 0
    for a in range(len(lines)):
        #x좌표정사영
        x1L = lines[a][0][0]
        y1L = lines[a][0][1]
        x2L = lines[a][1][0]
        y2L = lines[a][1][1]
        print(a)
        if x1L%4 != 0:
            startX = int(x1L/4) +1
        else:
            startX = int(x1L/4)
        endX = x2L/4
        while startX <= endX:
            xLines[startX] = 1
            startX+=1

        #y좌표 정사영
        if y1L % 4 != 0:
            startY = int(y1L / 4) + 1
        else:
            startY = int(y1L / 4)
        endY = y2L / 4
        while startY <= endY:
            xLines[startY] = 1
            startY += 1

    countX =0
    for i in range(len(xLines)):
        if xLines[i] == 1:
            countX +=1
        else:
            continue
    print(countX)
    countY = 0
    for i in range(len(yLines)):
        if yLines[i] == 1:
            countY += 1
        else:
            continue
    print(countY)
    xProp = countX / len(xLines)
    yProp = countY / len(yLines)
    print(xProp)
    print(yProp)
    result = math.sqrt(xProp**2 + yProp**2)

    return result

range_list = [[[20,30],[40,50]],
              [[30,30],[50,70]],
              [[60,80],[80,100]]]
result = oneLine(range_list)
print(result)