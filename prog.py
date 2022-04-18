import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils
import copy
import sys

image_name = input()

im = cv2.imread(image_name+".jpg")
RGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
blue = np.array([[j[2] for j in i] for i in RGB], dtype = "uint8")
all_colors = sum([gray[i] for i in range(len(gray))])
quant = sorted(all_colors)[len(all_colors)* 7//8]
RGB1 = np.array([[1 if ((j[0] < j[2]) and (j[1] < j[2])and (j[1]//2+j[0]//2<j[2]*0.7) and (j[2]<quant)) else 0 for j in i] for i in RGB], dtype='uint8')

def area(im, point, points):
    if  1 <= point[0] < len(im)-1:
        if 1 <= point[1] < len(im[0])-1:
            points.append(point)
            im[point[0], point[1]] = 0
            point_id = 0
            while point_id < len(points):
                if  1 <= points[point_id][0] < len(im)-1:
                    if 1 <= points[point_id][1] < len(im[0])-1:

                        if im[points[point_id][0], points[point_id][1] + 1] == 1:
                            points.append((points[point_id][0], points[point_id][1] + 1))
                            im[points[point_id][0], points[point_id][1] + 1] = 0

                        if im[points[point_id][0], points[point_id][1] - 1] == 1:
                            points.append((points[point_id][0], points[point_id][1] - 1))
                            im[points[point_id][0], points[point_id][1] - 1] = 0

                        if im[points[point_id][0] + 1, points[point_id][1]] == 1:
                            points.append((points[point_id][0] + 1, points[point_id][1]))
                            im[points[point_id][0] + 1, points[point_id][1]] = 0

                        if im[points[point_id][0] - 1, points[point_id][1]] == 1:
                            points.append((points[point_id][0] - 1, points[point_id][1]))
                            im[points[point_id][0] - 1, points[point_id][1]] = 0
                
                point_id+=1
                
im_area = copy.deepcopy(RGB1)
pointss = []
for i in range(len(im_area)):
    for j in range(len(im_area[0])):
        if im_area[i,j] == 1:
            points = []
            area(im_area, (i,j), points)
            pointss.append(points)
            
            
i = 0
while i < len(pointss):
    if len(pointss[i]) < 4000:
        pointss.pop(i)
    else:
        i+=1
        
        
def x_avg_point(points, x=True):
    if x:
        return sum([i[0] for i in points])/len([i[0] for i in points])
    else:
        return sum([i[1] for i in points])/len([i[1] for i in points])
        
        
lin_reg_data = [(x_avg_point(i), len(i), i) for i in pointss]
lin_reg_data_x = [x_avg_point(i) for i in pointss]
lin_reg_data_y = [len(i) for i in pointss]



lin_reg_data = sorted(lin_reg_data)



lin_reg_data1 = copy.deepcopy(lin_reg_data)



for j in range(len(pointss)):
    i = 0
    while i < len(lin_reg_data)-1:
        if lin_reg_data[i][1] * 0.8 > lin_reg_data[i+1][1]:
            lin_reg_data.pop(i)
            i-=1
        i+=1
        
        
lin_reg_data_x = [i[0] for i in lin_reg_data]
lin_reg_data_y = [i[1]**(1/2) for i in lin_reg_data]


from scipy.stats import linregress
model = linregress(lin_reg_data_x, lin_reg_data_y)



def dist(point1, point2):
    return ((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)**(1/2)

def dist_to_center(card_in_pixels, pixel):
    center = (len(card_in_pixels)//2, len(card_in_pixels[0])//2)
    return ((center[0]-pixel[0])**2 + (center[1]-pixel[1])**2)**(1/2)

def center(points):
    N = len(points)
    x_center = sum([point[0] for point in points])/N
    y_center = sum([point[1] for point in points])/N
    return x_center, y_center

def farthest_point(points, point):
    max_dist = 0
    my_point = point
    for some_point in points:
        if dist(some_point, point) > max_dist:
            max_dist = dist(some_point, point)
            my_point = some_point
            
    return my_point

def nearests_points(im, n_points, point, points, our_points):
    
    if  1 <= point[0] < len(im)-1:
        if 1 <= point[1] < len(im[0])-1:
            points.append(point)
            im[point[0], point[1]] = 0
            point_id = 0
            our_points.remove((point[0], point[1]))

            while point_id < len(points):
                if len(points)< n_points:
                    if  1 <= points[point_id][0] < len(im)-1:
                        if 1 <= points[point_id][1] < len(im[0])-1:

                            if im[points[point_id][0], points[point_id][1] + 1] == 1:
                                points.append((points[point_id][0], points[point_id][1] + 1))
                                im[points[point_id][0], points[point_id][1] + 1] = 0
                                our_points.remove((points[point_id][0], points[point_id][1] + 1))

                            if im[points[point_id][0], points[point_id][1] - 1] == 1:
                                points.append((points[point_id][0], points[point_id][1] - 1))
                                im[points[point_id][0], points[point_id][1] - 1] = 0
                                our_points.remove((points[point_id][0], points[point_id][1] - 1))

                            if im[points[point_id][0] + 1, points[point_id][1]] == 1:
                                points.append((points[point_id][0] + 1, points[point_id][1]))
                                im[points[point_id][0] + 1, points[point_id][1]] = 0
                                our_points.remove((points[point_id][0] + 1, points[point_id][1]))

                            if im[points[point_id][0] - 1, points[point_id][1]] == 1:
                                points.append((points[point_id][0] - 1, points[point_id][1]))
                                im[points[point_id][0] - 1, points[point_id][1]] = 0
                                our_points.remove((points[point_id][0] - 1, points[point_id][1]))
                
                point_id+=1
                
                
qkrq = []
new_im = copy.deepcopy(RGB1)

for i in range(len(new_im)):
    for j in range(len(new_im[0])):
        new_im[i,j]=0

for i in lin_reg_data1:
    for j in i[2]:
        new_im[j[0],j[1]]=1
        

for q in range(max([round(len(i[2])/(model.intercept + model.slope*i[0])**2) for i in lin_reg_data1])):
    for j,i in enumerate(lin_reg_data1):
        if round(len(lin_reg_data1[j][2])/(model.intercept + model.slope*lin_reg_data1[j][0])**2) > 1:
            qkrq_points = []
            nearests_points(new_im, 
                            len(lin_reg_data1[j][2])//round(len(lin_reg_data1[j][2])/(model.intercept + model.slope*lin_reg_data1[j][0])**2), 
                            farthest_point(lin_reg_data1[j][2], center(lin_reg_data1[j][2])), 
                            qkrq_points, 
                            lin_reg_data1[j][2])
            qkrq.append(qkrq_points)
            
def inPolygon(x, y, xp, yp):
    c=0
    for i in range(len(xp)):
        if (((yp[i]<=y and y<yp[i-1]) or (yp[i-1]<=y and y<yp[i])) and (x > (xp[i-1] - xp[i]) * (y - yp[i]) / (yp[i-1] - yp[i]) + xp[i])):
            c = 1 - c
    if c == 0:
        return True
    else:
        return False
                    
l = 0
number_of_cards = 0
qkrq1 = qkrq + [i[2] for i in lin_reg_data1]
for i in range(len(qkrq1)):
    if len(qkrq1[i])>1:
        number_of_cards += 1
        card = qkrq1[i]
        bottom = min([i[0] for i in card])
        top = max([i[0] for i in card])
        left = min([i[1] for i in card])
        right = max([i[1] for i in card])
        card_in_pixels = np.zeros(shape = (top-bottom + 2*l, right-left + 2*l), dtype = "uint8")
        for i in range(bottom-l, top+l):
            for j in range(left-l, right+l):
                card_in_pixels[i-bottom, j-left] = gray[i,j]


        card_in_pixels = cv2.GaussianBlur(card_in_pixels, (3,3), 0)
        edges = cv2.Canny(card_in_pixels, 10, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        cnt = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = imutils.grab_contours(cnt)

        min_dist = None
        cont = []

        for c in cnt:
            p = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02* p, True)
            mean_x = sum([i[0][1]for i in approx])/len([i[0][1]for i in approx])
            mean_y = sum([i[0][0]for i in approx])/len([i[0][0]for i in approx])
            
            
            our_dist = dist_to_center(card_in_pixels, (mean_x,mean_y))
            if min_dist is not None:
                if our_dist < min_dist:
                    min_dist = our_dist
                    cont = approx
            else:
                min_dist = our_dist
                cont = approx
        #     cv2.drawContours(card_in_pixels, [approx], -1, (0, 255, 0), 1)
        flag = True
        for j in range(len(cont)):
            cont1_x = []
            cont1_y = []
            for q, i in enumerate(cont):
                if q != j:
                    cont1_x.append(i[0][0])
                    cont1_y.append(i[0][1])
            flag = flag and inPolygon(cont[j][0][0], cont[j][0][1], cont1_x, cont1_y)
        cv2.putText(RGB, f"P{len(cont)}" + flag*"C", ((left*2+right)//3,(bottom*2+top)//3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
BGR = cv2.cvtColor(RGB, cv2.COLOR_RGB2BGR)
cv2.imwrite(image_name+"_parsed.jpg", BGR)
print(number_of_cards)





