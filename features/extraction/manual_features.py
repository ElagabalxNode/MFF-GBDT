import pandas as pd
import numpy as  np
import cv2
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime

# def getWeightFromExcel():
#     """
#     Get weight dictionary from excel file in each folder
#     :return: dir_weightDict
#     """
#     dirRoot = 'E:\BeanDataset'
#     df = pd.read_excel("data/bean399-30/bean-weight-all.xlsx")
#     dir2weightDict = {}
#     for i in range(len(df)):
#         type = str(df.loc[i,'品种'])
#         name = df.loc[i,'name']
#         id = df.loc[i,'id']
#         pos = df.loc[i, 'pos']
#         weight = df.loc[i,'weight']
#         path = os.path.join(dirRoot,type,name)
#
#         dir2weightDict[name] = weight
#
#         item = list(df.iloc[i,2:5])
#         item.insert(0,type)
#         print(item)
#         print(list(dir2weightDict.values()))
#
#         break
#     # split
#     return dir2weightDict



def chickenFeatureExt():

    imgDir = ['1-21']
    # imgDir = ['56-110']

    date_str = datetime.now().strftime('%Y%m%d')
    saveCSVname = f'GBDT/csvData/{date_str}-mixData/{date_str}-mixData_2D_3D_features.csv'

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
    df = pd.read_excel("coco_sets/mixData/20210206-200-weight.xlsx")
    idx_weightDict = {}
    for i in range(len(df)):
        idx = df.loc[i,'序号']  # serial number
        weight = df.loc[i,'体重/kg']  # weight/kg
        idx_weightDict[idx] = weight

    def getdep(rawImgPath,format,resolution):
        """
        Get raw depth data and convert to numpy format single-channel image
        :param rawImgPath: raw file path
        :param format: 'Z16' or 'DISPARITY32'
        :param resolution: (w,h)
        :return: numpy_array(w,h)
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

        dirPath = 'exps/data_20210206-200_weight_100-model-73-50.pth-result/maskImg' # Extract features only from images in maskImg
        maskPath = 'exps/data_20210206-200_weight_100-model-73-50.pth-result/mask'
        rawPath = 'data/20210206-200/raw/'

        imgNames = os.listdir(dirPath)
        imgNameList.extend(imgNames)

        for name in imgNames:
            print(name)
            weight = idx_weightDict[int(name.split('.')[0])]
            print('Broiler weight:',weight)
            weightList.append(weight)

            # imgPath = os.path.join(dirPath, name)
            # img = cv2.imread(imgPath)
            # name = '20210312166-6.png'

            dataItem = []
            imgPath = os.path.join(maskPath, name)
            print(imgPath)
            mask = cv2.imread(imgPath)
            ret, thresh = cv2.threshold(mask[:,:,0], 200, 1,0) # Threshold segmentation of predicted mask


            # print(type(mask),mask.shape)
            # cv2.imshow('thresh',thresh)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # Find contours
            contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

            print(len(contours))
            areas = []
            for c in range(len(contours)):
                areas.append(cv2.contourArea(contours[c]))
            # Contour with maximum area
            area = maxAreas = max(areas)  # Projected area
            print('Projected area', area)
            areaList.append(area)

            maxArea_id = areas.index(maxAreas)
            cnt = contours[maxArea_id] # cnt is the contour with maximum area

            perimeter = cv2.arcLength(cnt,True)  # Contour perimeter
            print('Contour perimeter', perimeter)
            perimeterList.append(perimeter)

            maxArea_min_rect = cv2.minAreaRect(cnt) # Minimum bounding rectangle of the contour with maximum area
            # print("Minimum rectangle",maxArea_min_rect)  # Coordinates of top-left corner (x,y), width and height (w,h), and rotation angle

            print("Minimum rectangle width",maxArea_min_rect[1][0])  # Coordinates of top-left corner (x,y), width and height (w,h), and rotation angle
            print("Minimum rectangle height",maxArea_min_rect[1][1])  # Coordinates of top-left corner (x,y), width and height (w,h), and rotation angle
            min_rect_widthList.append(maxArea_min_rect[1][0])
            min_rect_highList.append(maxArea_min_rect[1][1])

            epsilon = 0.01*cv2.arcLength(cnt,True) # 0.01 times the contour length as parameter for approximation
            approx = cv2.approxPolyDP(cnt,epsilon,True)  # Approximated contour
            approx_area = cv2.contourArea(approx)  # Area of approximated contour
            approx_perimeter = cv2.arcLength(approx,True) # Perimeter of approximated contour
            print('Approximated contour area',approx_area)
            print('Approximated contour perimeter',approx_perimeter)
            approx_areaList.append(approx_area)
            approx_perimeterList.append(approx_perimeter)

            x,y,w,h = cv2.boundingRect(cnt) # Axis-aligned bounding rectangle (non-rotated rectangle)
            rect_area = w*h
            extent = float(maxAreas)/rect_area  # Ratio of contour area to bounding rectangle area, reflects the extent of region expansion
            print('Ratio of contour area to bounding rectangle area', extent)
            extentList.append(extent)

            hull = cv2.convexHull(cnt)  # Convex hull of contour
            hull_perimeter = cv2.arcLength(hull,True) # Convex hull perimeter
            hull_area = cv2.contourArea(hull)  # Convex hull area
            solidity = float(area)/hull_area  # Ratio of contour area to convex hull area, reflects the solidity of the region
            print('Convex hull perimeter',hull_perimeter)
            print('Convex hull area', hull_area)
            print('Ratio of contour area to convex hull area', solidity)
            hull_perimeterList.append(hull_perimeter)
            hull_areaList.append(hull_area)
            solidityList.append(solidity)

            hull = cv2.convexHull(cnt,returnPoints = False)
            defects = cv2.convexityDefects(cnt,hull) # Convexity defects of convex hull
            # print(defects.shape)

            distenceList = []  # Approximate distance from each convexity defect to the farthest point
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

            max_defect_dist = max(distenceList) # Maximum approximate distance from convexity defect to farthest point
            sum_defect_dist = sum(distenceList) # Sum of approximate distances from all convexity defects to farthest points
            print('Maximum approximate distance from convexity defect to farthest point', max_defect_dist)
            print('Sum of approximate distances from all convexity defects to farthest points', sum_defect_dist)
            max_defect_distList.append(max_defect_dist)
            sum_defect_distList.append(sum_defect_dist)

            equi_diameter = np.sqrt(4*area/np.pi)  # Diameter of circle with area equal to contour area
            print('Diameter of circle with area equal to contour area', equi_diameter)
            equi_diameterList.append(equi_diameter)

            M = cv2.moments(cnt) # Contour moments
            # print (M)
            ellipse = cv2.fitEllipse(cnt) # Ellipse fitting, major and minor axes
            # print(ellipse)  # ((center x,y), (minor axis diameter, major axis diameter), rotation angle)
            print('Ellipse fitting minor axis diameter', ellipse[1][0])
            print('Ellipse fitting major axis diameter', ellipse[1][1])
            ellipse_shortList.append(ellipse[1][0])
            ellipse_longList.append(ellipse[1][1])

            denominator = np.sqrt(pow(2 * M['mu11'], 2) + pow(M['mu20'] - M['mu02'], 2))
            eps = 1e-4   # Define a very small value
            if (denominator > eps):
                # cosmin and sinmin are used to calculate the smaller eigenvalue λ2 of the image covariance matrix
                cosmin = (M['mu20'] - M['mu02']) / denominator
                sinmin = 2 * M['mu11'] / denominator
                # cosmax and sinmax are used to calculate the larger eigenvalue λ1 of the image covariance matrix
                cosmax = -cosmin
                sinmax = -sinmin
                 # imin is λ2 multiplied by zero-order central moment μ00
                imin = 0.5 * (M['mu20'] + M['mu02']) - 0.5 * (M['mu20'] - M['mu02']) * cosmin - M['mu11'] * sinmin
                 # imax is λ1 multiplied by zero-order central moment μ00
                imax = 0.5 * (M['mu20'] + M['mu02']) - 0.5 * (M['mu20'] - M['mu02']) * cosmax - M['mu11'] * sinmax
                ratio = imin / imax   # Ellipse inertia ratio

                eccentricity  = np.sqrt(1- ratio*ratio)  # Ellipse eccentricity
            else:
                eccentricity  = 0  # Ellipse eccentricity is 0 (a circle)

            print("Ellipse eccentricity",eccentricity)
            eccentricityList.append(eccentricity)


            #################  Extract 3D features ##############
            rawImgPath = os.path.join(rawPath, name.split('-')[0] + '.raw')
            print(rawImgPath)

            """
            Modified to uniformly use conImg instead of directly using deepImg to extract 3D features
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
                # deepImg = getdep(rawImgPath,'Z16',(360,640)) # Depth matrix
                deepImg = getdep(rawImgPath,'Z16',(720,1280)) # Depth matrix
                conImg = cv2.convertScaleAbs(deepImg, alpha=0.17)

            print(conImg.dtype)
            # mul_img = cv2.multiply(thresh.astype(np.int16), conImg)  # Preserves original 16-bit int16 data
            mul_img = cv2.multiply(thresh.astype(np.uint8), conImg)  # Preserves original 16-bit int16 data
            deepArr = mul_img[mul_img>0]
            print(deepArr)

            maxHeight = np.max(mul_img)
            minHeight = np.min(deepArr) # Minimum non-zero value
            meanHeight = np.mean(deepArr)
            max2min = maxHeight - minHeight
            mean2min = meanHeight - minHeight
            mean2max = maxHeight - meanHeight
            stdHeight = np.std(deepArr)
            heightSum=np.sum(mul_img)
            print('Maximum depth:',maxHeight)
            print('Minimum depth:',minHeight)
            print('Mean depth:',meanHeight)
            print('Depth range:', max2min)
            print('Distance from mean to minimum:', mean2min)
            print('Distance from mean to maximum:', mean2max)
            print('Standard deviation of depth:', stdHeight)
            print('Sum of depths:',heightSum)

            volumeZhu = area * maxHeight
            # print('Vzhu = ',Vzhu)
            volume = volumeZhu - heightSum    # Approximate volume of broiler
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

    # Create directory if it doesn't exist
    save_dir = os.path.dirname(saveCSVname)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"Created directory: {save_dir}")

    df.to_csv(saveCSVname,index=False)

# chickenFeatureExt()

def chicken1198_split():
    data = pd.read_csv('GBDT/csvData/20210206-200-1198-manuals/20210206-1198_2D_3D_features.csv')

    head = data.columns.values
    print(type(head))
    print(head)
    # print(data.head().to_numpy())
    # dataNumpy = data.to_numpy()

    print(np.mean(data['weight'])) #166.6583541147132
    print(np.std(data['weight'])) #52.54880630114353

    """
       "id,path,pos,BeanType,weight,imgName,area,perimeter,min_rect_width,min_rect_high,approx_area,approx_perimeter,extent,"
       "hull_perimeter,hull_area,solidity,max_defect_dist,sum_defect_dist,equi_diameter,ellipse_long,ellipse_short,eccentricity"
    """

    """60% training, 20% validation, 20% test"""
    # x_train,x_test = train_test_split(data.to_numpy() , test_size=0.2, random_state=43, shuffle=True)
    # x_train,x_val = train_test_split(x_train , test_size=0.25, random_state=43, shuffle=True)
    # print(len(x_train),len(x_val),len(x_test))

    # pd.DataFrame(x_train).to_csv("csvData/BeanCSV/allPos/Bean-train-60.csv", index=False, header=head)
    # pd.DataFrame(x_val).to_csv("csvData/BeanCSV/allPos/Bean-val-20.csv",index=False, header=head)
    # pd.DataFrame(x_test).to_csv("csvData/BeanCSV/allPos/Bean-test-20.csv",index=False, header=head)

    """80% training, 20% test"""
    #  39, 42,43,#45xgb#,
    x_train,x_test = train_test_split(data.to_numpy() , test_size=0.2, random_state=45, shuffle=True)
    # x_train,x_val = train_test_split(x_train , test_size=0.25, random_state=42, shuffle=True)
    # print(len(x_train),len(x_val),len(x_test))
    print(len(x_train),len(x_test))
    pd.DataFrame(x_train).to_csv("GBDT/csvData/20210206-200-1198-manuals/20210206-1198-train-0.8.csv", index=False, header=head)
    # pd.DataFrame(x_val).to_csv("GBDT/csvData/csvData/20210206-200-1198/Bean-val-20.csv",index=False, header=head)
    pd.DataFrame(x_test).to_csv("GBDT/csvData/20210206-200-1198-manuals/20210206-1198-test-0.2.csv",index=False, header=head)

def normalFeature():
    """Normalize hand-crafted features"""
    data = pd.read_csv('GBDT/csvData/20210206-200-1198-manuals/20210206-1198_2D_3D_features.csv')
    imgName = data['imgName']
    data = data.drop(['weight','imgName'],axis=1)

    scaler = MinMaxScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    print(type(data))

    df = pd.DataFrame(data)
    df = pd.concat([imgName,df],axis=1)
    print(df.head())

    df.to_csv('GBDT/csvData/20210206-200-1198-manuals/20210206-1198_2D_3D_normal_features.csv',index=False)
# chicken1198_split()
# normalFeature()

def chicken1198_normal_split():
    data = pd.read_csv('GBDT/csvData/20210206-200-1198-manuals/20210206-1198_2D_3D_features.csv')
    df = pd.read_csv('GBDT/csvData/20210206-200-1198-manuals/20210206-1198_2D_3D_normal_features.csv')
    df.insert(0,'weight',data['weight'])

    head = data.columns.values
    print(type(head))
    print(head)
    # print(data.head().to_numpy())
    # dataNumpy = data.to_numpy()

    print(np.mean(data['weight'])) #1.5932470784641068
    print(np.std(data['weight'])) #0.27380102700920916

    #  39, 42,43,#45xgb#,
    x_train,x_test = train_test_split(df.to_numpy() , test_size=0.2, random_state=45, shuffle=True)
    print(len(x_train),len(x_test))
    pd.DataFrame(x_train).to_csv("GBDT/csvData/20210206-200-1198-manuals/20210206-1198-normal-train-0.8.csv", index=False, header=head)
    # pd.DataFrame(x_val).to_csv("GBDT/csvData/csvData/20210206-200-1198/Bean-val-20.csv",index=False, header=head)
    pd.DataFrame(x_test).to_csv("GBDT/csvData/20210206-200-1198-manuals/20210206-1198-normal-test-0.2.csv",index=False, header=head)

chicken1198_normal_split()
