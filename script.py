'''
    Trabalho feito por: Fernando da Costa Kudrna (78181) e Leonardo Pellegrini Silva (78159)
'''

try:
    import cv2
    import time
    import math
    import matplotlib
    import numpy as np
    import PIL.Image, PIL.ImageTk
    from tkinter import *
    from tkinter import ttk
    from tkinter import messagebox
    from tkinter import filedialog
    from matplotlib import pyplot as plt
except:
    print("Este script requer Python 3.x e as bibliotecas OpenCV, TkInter, MatPlotLib, NumPy e Pillow")
    exit(0)

class Application():
    '''Classe principal da aplicação gráfica'''

    def __init__(self):
        '''Construtor da classe'''
        self.root = Tk()
        self.initComponents()        
        self.openImg()
        self.showImg(self.img)

    def openImg(self):
        '''Abre a imagem'''
        try:
            # Prompt user to open file    
            self.test = open(filedialog.askopenfilename(title="Imagem"), 'r')
        
            # Carregando a imagem
            self.img = cv2.resize(cv2.cvtColor(cv2.imread(self.test.name), cv2.COLOR_BGR2RGB),(800, 600))
            #self.img = cv2.cvtColor(cv2.resize(cv2.imread("data/mandioca1.jpg"), (600, 400)), cv2.COLOR_BGR2RGB)
            self.img_original = self.img.copy()
            self.height, self.width, channels = self.img.shape
            self.cv_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
            self.cv_cont = cv2.cvtColor(cv2.cvtColor(self.cv_img, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
        except:
            messagebox.showinfo(icon="error",title='Erro',message="Formato de arquivo inválido.")
            self.openImg()

    def initComponents(self):
        '''Inicialização dos componentes da aplicação'''
        self.canvas = Canvas(self.root, width = 800, height = 600)
        self.canvas.grid(row=0,column=0,rowspan=3,columnspan=2)
        self.btn = Button(self.root,text="Detectar as Bordas do Objeto")
        self.btn.configure(command=self.detectBorder)
        self.btn.grid(row=3,column=0,columnspan=2)

    def configComponents(self):
        '''Configuração da segunda parte do programa'''
        self.btn.destroy()
        self.ctrlP1 = Scale(self.root, label="Coordenada do Ponto 1", from_=self.pointsW[2] + 1, to=self.cx - 1, length=self.width / 2, orient=HORIZONTAL)
        self.ctrlP1.set((self.cx - self.pointsW[2])/2 + self.pointsW[2])
        self.ctrlP2 = Scale(self.root, label="Coordenada do Ponto 2", from_=self.cx + 1, to=self.pointsW[3] - 1, length=self.width / 2, orient=HORIZONTAL)
        self.ctrlP2.set((self.pointsW[3] - self.cx)/2 + self.cx)
        self.ctrlS1 = Scale(self.root, label="Densidade da supefície", from_=1, to=50, length=self.width / 2, orient=HORIZONTAL)
        self.btnB = Button(self.root,text="Gerar curva",command=self.calcBezier)
        self.btnS = Button(self.root,text="Gerar superfície", command=self.surfaceMapping, state=DISABLED)
        self.ctrlP1.grid(row=0,column=2)
        self.ctrlP2.grid(row=1,column=2)
        self.ctrlS1.grid(row=2,column=2)
        self.btnB.grid(row=3,column=0)
        self.btnS.grid(row=3,column=1)

    def showImg(self, image):
        '''Exibição da imagem na aplicação'''
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
    
    def detectBorder(self):
        '''Detecção de bordas'''
        lower_brown = np.array([0, 60, 0])
        upper_brown = np.array([255, 255, 255])
        mask = cv2.inRange(self.cv_img, lower_brown, upper_brown)
        self.cv_cont = cv2.bitwise_and(self.cv_cont, self.cv_cont, mask=mask)
        self.cv_cont = cv2.Canny(self.cv_cont, 100, 250)
        for i in range(self.height-1):
            for j in range(self.width-1):
                if self.cv_cont[i,j] == 255:
                    self.img[i, j] = [255, 0, 0]
        self.cv_curve = self.cv_cont.copy()
        self.showImg(self.img)
        self.btn.configure(text="Detectar as extremidades",command=self.detectExtP)

    def detectExtP(self):
        '''Detecta os pontos extremos e centróide'''
        # detecção da centróide
        ret,thresh = cv2.threshold(self.cv_cont,127,255,0)
        contours = cv2.findContours(thresh, 1, 2)
        cnt = contours[0]
        M = cv2.moments(cnt)
        self.cx = int(M['m10']/M['m00'])
        self.cy = int(M['m01']/M['m00'])
        #cv2.circle(self.img, (self.cx,self.cy), 8, (0, 255, 255), -1)

        # detecção dos pontos extremos
        first = False
        self.pointsW = [self.cx,self.cx,0,0]
        self.pointsH = [0,0,self.cy,self.cy]
        for i in range(self.height-1):        
            if self.cv_cont[i,self.cx] == 255:
                if not first:
                    first = True
                    self.pointsH[0] = i
                self.pointsH[1] = i
        first = False
        for i in range(self.width-1):      
            if self.cv_cont[self.cy,i] == 255:
                if not first:
                    first = True
                    self.pointsW[2] = i
                self.pointsW[3] = i
        
        # Expansão dos pontos extremos
        self.pointsH[0], self.pointsW[0] = self.aproxExtP1(self.exceedBound(self.pointsH[0],self.height,False),self.pointsH[0]+1,1,self.exceedBound(self.pointsW[2],self.width,False),self.exceedBound(self.pointsW[3],self.width,True),1)
        self.pointsH[1],self.pointsW[1] = self.aproxExtP1(self.exceedBound(self.pointsH[1],self.height,True),self.pointsH[1]-1,-1,self.exceedBound(self.pointsW[2],self.width,False),self.exceedBound(self.pointsW[3],self.width,True),1)
        self.pointsH[2], self.pointsW[2] = self.aproxExtP2(self.exceedBound(self.pointsW[2],self.width,False),self.pointsW[2]+1,1,self.exceedBound(self.pointsH[0],self.height,False),self.exceedBound(self.pointsH[1],self.height,True),1)
        self.pointsH[3], self.pointsW[3] = self.aproxExtP2(self.exceedBound(self.pointsW[3],self.width,True),self.pointsW[3]-1,-1,self.exceedBound(self.pointsH[0],self.height,False),self.exceedBound(self.pointsH[1],self.height,True),1)

        # Localização dos pontos extremos na imagem
        cv2.circle(self.img, (self.pointsW[0],self.pointsH[0]), 4, (0, 0, 255), -1)
        cv2.circle(self.img, (self.pointsW[1],self.pointsH[1]), 4, (255, 0, 255), -1)
        cv2.circle(self.img, (self.pointsW[2],self.pointsH[2]), 4, (255, 0, 0), -1)
        cv2.circle(self.img, (self.pointsW[3], self.pointsH[3]), 4, (0, 0, 0), -1)
        
        # Retângulo contendo o objeto
        self.drawContainer()

        # configurações da interface
        self.configComponents()

        self.img_original = self.img
        self.showImg(self.img)

    def aproxExtP1(self, from1, to1, step1, from2, to2, step2):
        '''Aproximação dos pontos horizontais'''
        for i in range(from1, to1, step1):
            for j in range(from2, to2, step2):
                if self.cv_cont[i, j] == 255:
                    return i, j

    def aproxExtP2(self, from1, to1, step1, from2, to2, step2):
        '''Aproximação dos pontos verticais'''
        for i in range(from1, to1, step1):
            for j in range(from2, to2, step2):
                if self.cv_cont[j, i] == 255:
                    return j, i

    def aproxCurveP(self, point1, point2, sum):
        '''Ajusta os limites dos pontos de limite da curva'''
        if sum:
            value = int((point1-point2)/2)
            arg = point1 + value
            if arg >= self.height:
                return self.height-1
            else:
                return arg
        else:
            value = int((point2-point1)/2)
            arg = point1 - value
            if arg < 0:
                return 0
            else:
                return arg
        
    def exceedBound(self, point, direction, plus):
        '''Ajusta os limites dos pontos extremos do objeto'''
        if plus:
            arg = point + int(direction / 10)
            if arg >= direction:
                return direction-1
            else:
                return arg
        else:
            arg = point - int(direction / 10)
            if arg < 0:
                return 0
            else:
                return arg

    def drawContainer(self):
        '''Desenha um retângulo que contém o objeto'''
        for i in range(self.pointsH[0], self.pointsH[1]):
            self.img[i,self.pointsW[2]] = 1
            self.img[i,self.pointsW[3]] = 1
        for i in range(self.pointsW[2], self.pointsW[3]):
            self.img[self.pointsH[0],i] = 1
            self.img[self.pointsH[1],i] = 1

    def cubicBezierSum(self, t, points):
        '''Cálculo de cada ponto da curva '''
        # Precisa de 4 pontos para gerar a curva
        # t varia do ponto inicial ao ponto final, utilizando a reta entre os pontos extremos como base
        t1 = points[0]*(1-t)**3
        t2 = 3*points[1]*(1-t)**2*t
        t3 = 3*points[2]*(1-t)*(t**2)
        t4 = points[3]*(t**3)
        return t1 + t2 + t3 + t4

    def calcCurve(self):
        '''Calcula a curva de beziér para os dois conjuntos de pontos'''
        curve1 = []
        curve2 = []
        t = 0
        while (t < 1):
            curve1.append((int(self.cubicBezierSum(t, self.curve1PointsH)), int(self.cubicBezierSum(t, self.curvePointsW))))
            curve2.append((int(self.cubicBezierSum(t, self.curve2PointsH)), int(self.cubicBezierSum(t, self.curvePointsW))))
            t += self.autoDetectSizeToDraw()
        return curve1,curve2

    def autoDetectSizeToDraw(self):
        '''Ajusta a quantidade de pontos na curva baseado na resolução'''
        zeros = '10'
        for i in range(len(str(self.height))):
            zeros += '0'
        return 1/int(zeros)

    def calcBezier(self):
        '''Calcula a curva de Bezier baseado nos pontos definidos'''
        self.curve2_height = self.pointsH[1] - self.pointsH[2]
        # Pontos da curva expandidos
        self.curveLimits = [self.aproxCurveP(self.pointsH[0],self.pointsH[2],False),self.aproxCurveP(self.pointsH[2] + self.curve2_height,self.pointsH[2],True),self.pointsW[2],self.pointsW[3]]
        # Pontos da curva normais
        #self.curveLimits = [self.pointsH[0],self.pointsH[2] + self.curve2_height,self.pointsW[2],self.pointsW[3]]
        self.curvePointsW = [self.pointsW[2],self.ctrlP1.get(),self.ctrlP2.get(),self.pointsW[3]]
        self.curve1PointsH = [self.pointsH[2], self.curveLimits[0], self.curveLimits[0], self.pointsH[3]]
        self.curve2PointsH = [self.pointsH[2], self.curveLimits[1], self.curveLimits[1], self.pointsH[3]]
        self.img = self.img_original.copy()
        self.cv_curve = self.cv_cont.copy()
        
        # Exibição dos pontos de controle da curva
        cv2.circle(self.img, (self.curvePointsW[0], self.curve1PointsH[0]), 4, (0, 0, 0), -1)
        cv2.circle(self.img, (self.curvePointsW[1], self.curve1PointsH[1]), 4, (0, 0, 0), -1)
        cv2.circle(self.img, (self.curvePointsW[1], self.curve2PointsH[1]), 4, (0, 0, 0), -1)
        cv2.circle(self.img, (self.curvePointsW[2], self.curve2PointsH[2]), 4, (0, 0, 0), -1)
        cv2.circle(self.img, (self.curvePointsW[2], self.curve1PointsH[2]), 4, (0, 0, 0), -1)
        cv2.circle(self.img, (self.curvePointsW[3], self.curve1PointsH[3]), 4, (0, 0, 0), -1)

        # Construção da curva
        self.curve1, self.curve2 = self.calcCurve()
        self.maxH1 = self.curve1[0][0]
        self.maxH2 = self.curve2[0][0]
        for i in range(0, len(self.curve1)):
            if self.curve1[i][0] < self.maxH1:
                self.maxH1 = self.curve1[i][0]
            if self.curve2[i][0] > self.maxH2:
                self.maxH2 = self.curve2[i][0]
            self.img[self.curve1[i]] = (0,150,0)
            self.img[self.curve2[i]] = (0,150,0)
            self.cv_curve[self.curve1[i]] = 100
            self.cv_curve[self.curve2[i]] = 100

        # Criando backups para restaurar após seleção de novos pontos
        self.cv_img = self.cv_curve.copy()
        self.cv_img_copy = self.img.copy()
        self.btnS.configure(state=NORMAL)
        self.showImg(self.img)

    def surfaceMapping(self):
        '''Mapeamendo da superfície'''
        self.img = self.cv_img_copy.copy()
        surfaceDensity = self.ctrlS1.get()
        
        # determinando limites da superfície Letf+Right
        pointsLeft = []
        pointsRight = []
        for i in range(self.curveLimits[1]):
            for j in range(self.curveLimits[3]):
                if self.cv_curve[i, j] == 100:
                    pointsLeft.append((i,j))
                    break
            for j in range(self.curveLimits[3],self.curveLimits[2], -1):
                if self.cv_curve[i, j] == 100:
                    pointsRight.append(j)
                    break
        
        # determinando limites da superfície Up+Down
        pointsUp = []
        pointsDown = []
        for i in range(self.curveLimits[3]):
            for j in range(self.curveLimits[1]):
                if self.cv_curve[j, i] == 100:
                    pointsDown.append((i,j))
                    break
            for j in range(self.curveLimits[1],self.curveLimits[0], -1):
                if self.cv_curve[j, i] == 100:
                    pointsUp.append(j)
                    break

        # preenchendo a superfície
        for x in range(len(pointsLeft)):
            if (pointsLeft[x][0] % surfaceDensity == 0):
                h , w  = pointsLeft[x]
                while (w < pointsRight[x]):
                    self.img[h,w] = (0,150,0)
                    w += 1
        for x in range(len(pointsDown)):
            if (pointsDown[x][0] % surfaceDensity == 0):
                w , h  = pointsDown[x]
                while (h < pointsUp[x]):
                    self.img[h,w] = (0,150,0)
                    h += 1

        self.showImg(self.img)

    def start(self):
        '''Inicializa a interface gráfica'''
        self.root.mainloop()

if __name__ == "__main__":
    '''Inicialização do programa'''
    Application().start()