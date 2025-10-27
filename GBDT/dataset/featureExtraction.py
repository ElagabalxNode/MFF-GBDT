import pandas as pd
import numpy as  np
import cv2
import os
import xlrd

def getWeightFromExcel():
    """
    # Get the weight dictionary from the excel file in each folder
    :return: dir_weightDict
    """
    dir_weightDict = dict()
    for imgDir in os.listdir('E:/chicken-mean/20210911mean/'):
        excelPath = 'E:/chicken-mean/20210911mean/' + imgDir
        pathList = os.listdir(excelPath)

        excelName = ''
        weightDict = dict()

        for p in pathList:
            if p.endswith('xlsx'):
                excelName = p

        wk = xlrd.open_workbook(os.path.join(excelPath,excelName))
        sheet = wk.sheet_by_index(0)
        # Iterate through the rows
        for row_num in range(2, sheet.nrows): # From the second row
            id = sheet.row_values(row_num)[0]  # The first column is the id
            weight = sheet.row_values(row_num)[1] # The second column is the weight/kg
            print(id, weight)
            weightDict[id] = weight

        dir_weightDict[imgDir] = weightDict

    return dir_weightDict


# TODO Change to extract features from multiple segmentation result folders


imgDir = ['1-21','2-35','34-7']
# imgDir = ['56-110']
saveCSVname = 'csvData/20210911-90_2D_3D_features.csv'

imgNameList = []

# 2D features
areaList = []
perimeterList = []
min_rect_widthList = []
min_rect_highList = []
approx_areaList = []
approx_perimeterList = []
extentList = []
hull_perimeterList = []
hull_areaList = []
solidityList = []
max_defect_distList = []
sum_defect_distList = []
equi_diameterList = []
ellipse_shortList = []
ellipse_longList = []
eccentricityList = []
weightList = []

# 3D features
volumeList = []
maxHeightList = []
minHeightList = []
max2minList = []
meanHeightList = []
mean2minList = []
mean2maxList = []
stdHeightList = []
heightSumList = []


dataDirt = {'weight': weightList, 'imgName': imgNameList, 'area': areaList, 'perimeter': perimeterList, 'min_rect_width': min_rect_widthList,
            'min_rect_high': min_rect_highList, 'approx_area': approx_areaList, 'approx_perimeter': approx_perimeterList,
            'extent': extentList, 'hull_perimeter': hull_perimeterList, 'hull_area': hull_areaList, 'solidity': solidityList,
            'max_defect_dist': max_defect_distList, 'sum_defect_dist': sum_defect_distList, 'equi_diameter': equi_diameterList,
            'ellipse_long': ellipse_longList, 'ellipse_short': ellipse_shortList, 'eccentricity': eccentricityList,

            'volume': volumeList, 'maxHeight': maxHeightList, 'minHeight': minHeightList,'max2min':max2minList,
            'meanHeight': meanHeightList,'mean2min': mean2minList, 'mean2max': mean2maxList,'stdHeight': stdHeightList,
            'heightSum': heightSumList,
            }

dir_weightDict = getWeightFromExcel()

def getdep(rawImgPath,format,resolution):
    """
    Get the original raw depth data and convert it to a single-channel numpy array
    :param rawImgPath: raw file path
    :param format: 'Z16' or 'DISPARITY32'
    :param resolution: (w,h)
    :return: numpy_arrary(w,h)
    """
    if format == 'Z16':
        type = np.int16 # format = Z16
        width, height = resolution
    # width, height = (720, 1280)
    elif format == 'DISPARITY32':
        type = np.float32 # format = DISPARITY32
        width, height = resolution

    imgData = np.fromfile(rawImgPath, dtype=type)
    imgData = imgData.reshape(width, height)

    return imgData

for dir in imgDir:

    dirPath = 'data/20210911/' + dir + '/maskImg/'
    maskPath = 'data/20210911/' + dir + '/mask/'
    rawPath = 'data/20210911/' + dir + '/raw/'

    imgNames = os.listdir(dirPath)
    imgNameList.extend(imgNames)

    for name in imgNames:
        print(name)
        weight = dir_weightDict[dir][int(name.split('.')[0])]
        print('The weight of the chicken:',weight)
        weightList.append(weight)

        # imgPath = os.path.join(dirPath, name)
        # img = cv2.imread(imgPath)
        # name = '20210312166-6.png'

        dataItem = []
        imgPath = os.path.join(maskPath, name)
        print(imgPath)
        mask = cv2.imread(imgPath)
        ret, thresh = cv2.threshold(mask[:,:,0], 200, 1,0) # Threshold processing the predicted mask


        # print(type(mask),mask.shape)
        # cv2.imshow('thresh',thresh)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Find the contours
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        print(len(contours))
        areas = []
        for c in range(len(contours)):
            areas.append(cv2.contourArea(contours[c]))
        # The largest area contour
        area = maxAreas = max(areas)  # Projected area
        print('Projected area', area)
        areaList.append(area)

        maxArea_id = areas.index(maxAreas)
        cnt = contours[maxArea_id] #cnt is the contour with the largest area

        perimeter = cv2.arcLength(cnt,True)  # Contour perimeter
        print('Contour perimeter', perimeter)
        perimeterList.append(perimeter)

        maxArea_min_rect = cv2.minAreaRect(cnt) # The smallest rectangle of the contour with the largest area
        # print("The smallest rectangle",maxArea_min_rect)  # The coordinates of the left-upper corner point (x,y), the width and height of the rectangle (w,h), and the rotation angle

        print("The width of the smallest rectangle",maxArea_min_rect[1][0])  # The width of the smallest rectangle
        print("The height of the smallest rectangle",maxArea_min_rect[1][1])  # The height of the smallest rectangle
        min_rect_widthList.append(maxArea_min_rect[1][0])
        min_rect_highList.append(maxArea_min_rect[1][1])

        epsilon = 0.01*cv2.arcLength(cnt,True) # 0.01 times the contour length as the parameter for approximate calculation
        approx = cv2.approxPolyDP(cnt,epsilon,True)  # Approximate contour
        approx_area = cv2.contourArea(approx)  # Approximate contour area
        approx_perimeter = cv2.arcLength(approx,True) # Approximate contour perimeter
        print('Approximate contour area',approx_area)
        print('Approximate contour perimeter',approx_perimeter)
        approx_areaList.append(approx_area)
        approx_perimeterList.append(approx_perimeter)

        x, y, w, h = cv2.boundingRect(cnt)  # Straight bounding rectangle (no rotation)
        rect_area = w * h
        extent = float(maxAreas) / rect_area  # Ratio of contour area to bounding rectangle area; reflects how much of the rectangle is filled by the contour
        print('Ratio of contour area to bounding rectangle area:', extent)
        extentList.append(extent)

        hull = cv2.convexHull(cnt)  # Convex hull
        hull_perimeter = cv2.arcLength(hull,True) # Convex hull perimeter
        hull_area = cv2.contourArea(hull)  # Convex hull area
        solidity = float(area)/hull_area  # Ratio of contour area to convex hull area; reflects how much of the convex hull is filled by the contour
        print('Convex hull perimeter',hull_perimeter)
        print('Convex hull area', hull_area)
        print('Ratio of contour area to convex hull area', solidity)
        hull_perimeterList.append(hull_perimeter)
        hull_areaList.append(hull_area)
        solidityList.append(solidity)

        hull = cv2.convexHull(cnt,returnPoints = False)
        defects = cv2.convexityDefects(cnt,hull) # Convex hull defects
        # print(defects.shape)

        distenceList = []  # The approximate distance from each convex defect to the farthest point
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            # print(s,e,f,d)
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            distenceList.append(d)
            # cv2.line(mask,start,end,[0,255,0],1)
            # cv2.circle(mask,far,3,[0,0,255],-1)
        # cv2.imshow('mask',mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        max_defect_dist = max(distenceList) # The maximum approximate distance from the convex defect to the farthest point
        sum_defect_dist = sum(distenceList) # The sum of the approximate distances from the farthest points of all convex defects
        print('The maximum approximate distance from the convex defect to the farthest point', max_defect_dist)
        print('The sum of the approximate distances from the farthest points of all convex defects', sum_defect_dist)
        max_defect_distList.append(max_defect_dist)
        sum_defect_distList.append(sum_defect_dist)

        equi_diameter = np.sqrt(4*area/np.pi)  # The diameter of the circle with the same area as the contour
        print('The diameter of the circle with the same area as the contour', equi_diameter)
        equi_diameterList.append(equi_diameter)

        M = cv2.moments(cnt) # The moment of the contour
        # print (M)
        ellipse = cv2.fitEllipse(cnt) # Ellipse fitting  The long axis and short axis
        # print(ellipse)  # The center point (x,y), the short axis diameter, the long axis diameter, and the rotation angle
        print('The short axis diameter of the fitted ellipse', ellipse[1][0])
        print('The long axis diameter of the fitted ellipse', ellipse[1][1])
        ellipse_shortList.append(ellipse[1][0])
        ellipse_longList.append(ellipse[1][1])

        denominator = np.sqrt(pow(2 * M['mu11'], 2) + pow(M['mu20'] - M['mu02'], 2))
        eps = 1e-4   # Define a very small value
        if (denominator > eps):
            # cosmin and sinmin are used to calculate the smaller eigenvalue λ2 of the image covariance matrix
            cosmin = (M['mu20'] - M['mu02']) / denominator
            sinmin = 2 * M['mu11'] / denominator
            # cosmin and sinmax are used to calculate the larger eigenvalue λ1 of the image covariance matrix
            cosmax = -cosmin
            sinmax = -sinmin
             # imin is the λ2 multiplied by the zeroth-order central moment μ00
            imin = 0.5 * (M['mu20'] + M['mu02']) - 0.5 * (M['mu20'] - M['mu02']) * cosmin - M['mu11'] * sinmin
             # imax is the λ1 multiplied by the zeroth-order central moment μ00
            imax = 0.5 * (M['mu20'] + M['mu02']) - 0.5 * (M['mu20'] - M['mu02']) * cosmax - M['mu11'] * sinmax
            ratio = imin / imax   # The inertia ratio of the ellipse

            eccentricity  = np.sqrt(1- ratio*ratio)  # The eccentricity of the ellipse
        else:
            eccentricity  = 0  # The eccentricity of the ellipse is 0 for a circle

        print("The eccentricity of the ellipse",eccentricity)
        eccentricityList.append(eccentricity)


        #################  Extract 3D features ##############
        rawImgPath = os.path.join(rawPath, name.split('-')[0] + '.raw')
        print(rawImgPath)

        """
        Modify to use conImg instead of deepImg to extract 3D features
        """
        if dir == '34-7' or dir == '56-110':
            deepImg = getdep(rawImgPath,'DISPARITY32',(360,640)) # Depth matrix
            """
            When maxD = 1.5, alpha = 0.17; alpha = 255/(maxD*10**3)
            """
            conImg = cv2.convertScaleAbs(deepImg, alpha=0.17)
            conImg = 255 - conImg
            conImg = np.where(conImg==255,0,conImg)
        else:
            deepImg = getdep(rawImgPath,'Z16',(360,640)) # Depth matrix
            conImg = cv2.convertScaleAbs(deepImg, alpha=0.17)

        print(conImg.dtype)
        # mul_img = cv2.multiply(thresh.astype(np.int16), conImg)  # Keep the original 16bit int16 data
        mul_img = cv2.multiply(thresh.astype(np.uint8), conImg)  # Keep the original 16bit int16 data
        deepArr = mul_img[mul_img>0]
        print(deepArr)

        maxHeight = np.max(mul_img)
        minHeight = np.min(deepArr) # The minimum value of the non-zero elements
        meanHeight = np.mean(deepArr)
        max2min = maxHeight - minHeight
        mean2min = meanHeight - minHeight
        mean2max = maxHeight - meanHeight
        stdHeight = np.std(deepArr)
        heightSum=np.sum(mul_img)
        print('Maximum depth:',maxHeight)
        print('Minimum depth:',minHeight)
        print('Mean depth:',meanHeight)
        print('Depth difference:', max2min)
        print('Distance from mean to minimum:', mean2min)
        print('Distance from mean to maximum:', mean2max)
        print('Standard deviation of height:', stdHeight)
        print('Depth sum:',heightSum)

        volumeZhu = area * maxHeight
        # print('Vzhu = ',Vzhu)
        volume = volumeZhu - heightSum    # The approximate volume of the chicken
        print('Volume:', volume)

        maxHeightList.append(maxHeight)
        minHeightList.append(minHeight)
        max2minList.append(max2min)
        meanHeightList.append(meanHeight)
        mean2minList.append(mean2min)
        mean2maxList.append(mean2max)
        stdHeightList.append(stdHeight)
        heightSumList.append(heightSum)
        volumeList.append(volume)

        # serier = pd.Series(dataItem)
        # print(serier)

        # break


print(dataDirt)
df = pd.DataFrame(dataDirt)
print(df.info)

df.to_csv(saveCSVname,index=False)

