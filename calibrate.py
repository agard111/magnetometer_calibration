import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt
import pandas as pd
import math
import warnings

warnings.filterwarnings("ignore")



class Magnetometer(object):

    plot_count = 0
    
    '''
        To obtain Gravitation Field (raw format):
    1) get the Total Field for your location from here:
       http://www.ngdc.noaa.gov/geomag-web (tab Magnetic Field)
       es. Total Field = 47,241.3 nT | my val :47'789.7
    2) Convert this values to Gauss (1nT = 10E-5G)
       es. Total Field = 47,241.3 nT = 0.47241G
    3) Convert Total Field to Raw value Total Field, which is the
       Raw Gravitation Field we are searching for
       Read your magnetometer datasheet and find your gain value,
       Which should be the same of the collected raw points
       es. on HMC5883L, given +_ 1.3 Ga as Sensor Field Range settings
           Gain (LSB/Gauss) = 1090 
           Raw Total Field = Gain * Total Field
           0.47241 * 1090 = ~515  |
           
        -----------------------------------------------
         gain (LSB/Gauss) values for HMC5883L
            0.88 Ga => 1370 
            1.3 Ga => 1090 
            1.9 Ga => 820
            2.5 Ga => 660 
            4.0 Ga => 440
            4.7 Ga => 390 
            5.6 Ga => 330
            8.1 Ga => 230 
        -----------------------------------------------

     references :
        -  https://teslabs.com/articles/magnetometer-calibration/      
        -  https://www.best-microcontroller-projects.com/hmc5883l.html

    '''
    MField = 300

    def __init__(self, F=MField): 

        # initialize values
        self.F   = F
        self.b   = np.zeros([3, 1])
        self.A_1 = np.eye(3)

    def split(self, data):
        """
        Break out the x, y, and z into it's own array for plotting
        """
        xx = []
        yy = []
        zz = []
        for v in data:
            xx.append(v[0])
            yy.append(v[1])
            zz.append(v[2])
        return xx, yy, zz

    def plotMagnetometer3D(self, data, title=None):

        X, Y, Z = self.split(data)
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        ax.plot(X, Y, Z, '.b')
        ax.set_xlabel('$\mu$T')
        ax.set_ylabel('$\mu$T')
        ax.set_zlabel('$\mu$T')
        ax.set_box_aspect(aspect = (1,0.67,0.67))

        max_range = np.array([max(X) - min(X), max(Y) - min(Y), max(Z) - min(Z)]).max()
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (max(X) - min(X))
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (max(Y) - min(Y))
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (max(Z) - min(Z))


        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')


        if title:
            plt.title(title)

        plt.savefig(str(self.plot_count) + '.png')
        plt.clf()
        self.plot_count += 1

    def plotMagnetometer(self, data, title=None):
        x, y, z = self.split(data)
        plt.plot(x, y, '.b', x, z, '.r', z, y, '.g')
        plt.xlabel('$\mu$T')
        plt.ylabel('$\mu$T')

        plt.axis('equal')
        plt.grid(True)
        if title:
            plt.title(title)

        plt.savefig(str(self.plot_count) + '.png')
        plt.clf()
        self.plot_count += 1

    def loadRawData(self):
        data = np.loadtxt("/Users/agarde/Desktop/SmartFinData/smartfin-tools/smartfin/first_edit.csv", delimiter=',')
        print("shape of data:", data.shape)
        # print("datatype of data:",data.dtype)
        print("First 5 rows raw:\n", data[:5])
        return data

    def createMatrices(self):
        data = self.loadRawData()

        self.plotMagnetometer(data,"Uncalibrated 2D")
        self.plotMagnetometer3D(data, "Uncalibrated 3D")


        # ellipsoid fit
        s = np.array(data).T
        M, n, d = self.__ellipsoid_fit(s)

        # calibration parameters
        M_1 = linalg.inv(M)
        self.b = -np.dot(M_1, n)
        self.A_1 = np.real(self.F / np.sqrt(np.dot(n.T, np.dot(M_1, n)) - d) * linalg.sqrtm(M))
        

        print("Soft iron transformation matrix:\n",self.A_1)
        print("Hard iron bias:\n", self.b)

        pd.DataFrame(self.A_1).to_csv("SoftIron.csv", index=False)
        pd.DataFrame(self.b).to_csv("HardIron.csv",  index=False)


        result = self.applyMatrices()

        self.plotMagnetometer(result,"Calibrated 2D")
        self.plotMagnetometer3D(result, "Calibrated 3D")


        print("First 5 rows calibrated:\n", result[:5])

        print("*************************" )        
        print("code to paste : " )
        print("*************************" )  
        print("float hard_iron_bias_x = ", float(self.b[0]), ";")
        print("float hard_iron_bias_y = " , float(self.b[1]), ";")
        print("float hard_iron_bias_z = " , float(self.b[2]), ";")
        print("\n")
        print("double soft_iron_bias_xx = " , float(self.A_1[0,0]), ";")
        print("double soft_iron_bias_xy = " , float(self.A_1[1,0]), ";")
        print("double soft_iron_bias_xz = " , float(self.A_1[2,0]), ";")
        print("\n")
        print("double soft_iron_bias_yx = " , float(self.A_1[0,1]), ";")
        print("double soft_iron_bias_yy = " , float(self.A_1[1,1]), ";")
        print("double soft_iron_bias_yz = " , float(self.A_1[2,1]), ";")
        print("\n")
        print("double soft_iron_bias_zx = " , float(self.A_1[0,2]), ";")
        print("double soft_iron_bias_zy = " , float(self.A_1[1,2]), ";")
        print("double soft_iron_bias_zz = " , float(self.A_1[2,2]), ";")
        print("\n")

        return result

    def getHeading(self, result):
        x, y, z = self.split(result)

        headings = []

        with open("heading.txt", 'w') as df:
            for i in range(len(x)):
               #heading = math.atan2(x[i], z[i]) * (180 / math.pi)
                heading = math.atan2(z[i], x[i])
               # If heading is negative, convert to positive, 2 x pi is a full circle in Radians
                if heading < 0:
                    heading += 2 * math.pi

                # Convert heading from Radians to Degrees
                heading = math.degrees(heading)
                # Round heading to nearest full degree
                heading = round(heading)
                if(heading == 1):
                    continue
                headings.append(heading)
                df.write(str(heading) + '\n')


    def plotHeadings(self):
        headings = np.loadtxt("heading"
                              ".txt").astype(np.int)
        #headings = list(headings)
        y_ax = []
        [y_ax.append(i) for i in range(len(headings))]

        plt.scatter(y_ax, headings, s=10) #Cartesian Graph
        plt.title('Heading vs Time')
        plt.xlabel('Time')
        plt.ylabel('Degrees')
        plt.savefig("scatterHeadings.png", dpi=300)

        plt.clf()

        r = np.arange(len(headings)) #Polar graph
        arr = np.array(headings)
        theta = (arr / 180) * math.pi + math.pi / 2
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        labels = [item.get_text() for item in ax.get_xticklabels()]
        labels[0], labels[1], labels[2], labels[3], \
        labels[4], labels[5], labels[6], labels[7] = 'E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'
        ax.set_xticklabels(labels)
        ax.set_yticklabels([])
        ax.scatter(theta, r, marker=',', s=1, cmap = 'winter')
        plt.savefig("polarHeadings.png", dpi=1000)






    def applyMatrices(self):
        data = self.loadRawData()
        result = []
        hardIron = pd.read_csv("HardIron.csv").to_numpy()
        softIron = pd.read_csv("SoftIron.csv").to_numpy()

        for row in data:
            # subtract the hard iron offset
            xm_off = row[0] - hardIron[0]
            ym_off = row[1] - hardIron[1]
            zm_off = row[2] - hardIron[2]

            # multiply by the inverse soft iron offset
            xm_cal = xm_off * softIron[0, 0] + ym_off * softIron[0, 1] + zm_off * softIron[0, 2]
            ym_cal = xm_off * softIron[1, 0] + ym_off * softIron[1, 1] + zm_off * softIron[1, 2]
            zm_cal = xm_off * softIron[2, 0] + ym_off * softIron[2, 1] + zm_off * softIron[2, 2]

            result = np.append(result, np.array([xm_cal, ym_cal, zm_cal]))  # , axis=0 )
            # result_hard_iron_bias = np.append(result, np.array([xm_off, ym_off, zm_off]) )

        result = result.reshape(-1, 3)
        result = result / 100


        np.savetxt('rawResult.txt', result, fmt='%f', delimiter=' ,')

        return result


    def calculateAverageDirection(self, numIntervals, sessionTime):
        timePerInterval = sessionTime / numIntervals
        current_interval = 0
        headingSum = 0
        headingIndex = 0
        avgHeading = []

        with open("heading.txt", 'r') as hFile:

            totalHeadings = hFile.readlines()
            fileLength = len(totalHeadings)
            linesPerInterval = int(fileLength / numIntervals)
            print(linesPerInterval)
            while headingIndex < fileLength:
                headingSum += int(totalHeadings[headingIndex])
                if headingIndex % linesPerInterval == 0 and headingIndex != 0:
                    avgHeading.append(headingSum / linesPerInterval)
                    headingSum = 0
                headingIndex += 1

            for avg in avgHeading:
                text = "The average heading from {}s to {}s is {:.2f}:".format(int(current_interval), current_interval + timePerInterval, avg)
                print(text)
                current_interval += timePerInterval

    def giveDirection(self, heading):
        pass








    def __ellipsoid_fit(self, s):
        ''' Estimate ellipsoid parameters from a set of points.

            Parameters
            ----------
            s : array_like
              The samples (M,N) where M=3 (x,y,z) and N=number of samples.

            Returns
            -------
            M, n, d : array_like, array_like, float
              The ellipsoid parameters M, n, d.

            References
            ----------
            .. [1] Qingde Li; Griffiths, J.G., "Least squares ellipsoid specific
               fitting," in Geometric Modeling and Processing, 2004.
               Proceedings, vol., no., pp.335-340, 2004
        '''

        # D (samples)
        D = np.array([s[0]**2., s[1]**2., s[2]**2.,
                      2.*s[1]*s[2], 2.*s[0]*s[2], 2.*s[0]*s[1],
                      2.*s[0], 2.*s[1], 2.*s[2], np.ones_like(s[0])])

        # S, S_11, S_12, S_21, S_22 (eq. 11)
        S = np.dot(D, D.T)
        S_11 = S[:6,:6]
        S_12 = S[:6,6:]
        S_21 = S[6:,:6]
        S_22 = S[6:,6:]

        # C (Eq. 8, k=4)
        C = np.array([[-1,  1,  1,  0,  0,  0],
                      [ 1, -1,  1,  0,  0,  0],
                      [ 1,  1, -1,  0,  0,  0],
                      [ 0,  0,  0, -4,  0,  0],
                      [ 0,  0,  0,  0, -4,  0],
                      [ 0,  0,  0,  0,  0, -4]])

        # v_1 (eq. 15, solution)
        E = np.dot(linalg.inv(C),
                   S_11 - np.dot(S_12, np.dot(linalg.inv(S_22), S_21)))

        E_w, E_v = np.linalg.eig(E)

        v_1 = E_v[:, np.argmax(E_w)]
        if v_1[0] < 0: v_1 = -v_1

        # v_2 (eq. 13, solution)
        v_2 = np.dot(np.dot(-np.linalg.inv(S_22), S_21), v_1)

        # quadric-form parameters
        M = np.array([[v_1[0], v_1[3], v_1[4]],
                      [v_1[3], v_1[1], v_1[5]],
                      [v_1[4], v_1[5], v_1[2]]])
        n = np.array([[v_2[0]],
                      [v_2[1]],
                      [v_2[2]]])
        d = v_2[3]

        return M, n, d
        
        
        
if __name__=='__main__':

        #Magnetometer().createMatrices() DO NOT RUN THIS UNLESS YOU WANT TO CREATE NEW MATRICES
        result = Magnetometer().applyMatrices()
        Magnetometer().getHeading(result)
        Magnetometer().plotHeadings()
        Magnetometer().calculateAverageDirection(50, 1100)

