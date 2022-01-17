class data():
    def __init__(self, L,background=None, boxes=[], chars=[], alphas=[]):
        self.background=background
        self.boxes=boxes
        self.chars=chars
        self.alphas=alphas
        self.L=L
    def generate_backround(self,h=110,w=470,d=5):
        
        def polynom(C,x,y):
            z=0
            for i in range(len(C)):
                for j in range(i,len(C[0])):
                    z+=C[i,j] * np.cos((x*i + y*j)*2* np.pi)
            return (100+z)
        
        im=np.zeros((h,w))
        C=10*(np.random.rand(d,d)-1/2)
        for i in range(h):
            for j in range(w):
                im[i,j]=polynom(C,i/h,j/w)
        self.background=im
        
    def draw_shape(self,shape,xy,alpha):
        b=cv2.resize(shape, (xy[1][1]-xy[1][0],xy[0][1]-xy[0][0]))
        if random.choices([True,False],[0.2,0.8])[0]:
            kh=cv2.resize(random.choice(L[23]), (xy[1][1]-xy[1][0],xy[0][1]-xy[0][0]))
            b=255*(1-b/255)*(1-kh/255) + b
        crop=self.background[xy[0][0]:xy[0][1],xy[1][0]:xy[1][1]]*(b/255) + (1- b/255)*alpha
        self.background[xy[0][0]:xy[0][1],xy[1][0]:xy[1][1]]=crop
        
    def generate_boxes(self,n):
        Bh,Bw=self.background.shape[0],self.background.shape[1]
        for i in range(n):
            h,w=random.randrange(20,100),random.randrange(50,70)
            x_1,y_1=random.randrange(Bh-30), random.randrange(int(i*Bw/n),int((i+1)*Bw/n)-30)
            x_2,y_2=min([Bh,x_1+h]), min([int((i+1)*Bw/n),y_1+w])
            self.boxes.append([(x_1,x_2),(y_1,y_2)])
            self.chars.append(random.randrange(0,22))
            self.alphas.append(random.randrange(0,70))
        
    def generate_noise(self,n):
        Bh,Bw=self.background.shape[0],self.background.shape[1]
        for i in range(n):
            h,w=random.randrange(50,100),random.randrange(50,70)
            x_1,y_1=random.randrange(Bh), random.randrange(Bw)
            x_2,y_2=min([Bh,x_1+h]), min([Bw,y_1+w])
            alpha=random.randrange(73,100)
            shape=random.choice(L[22])
            self.draw_shape(shape,[(x_1,x_2),(y_1,y_2)],alpha)
            
    def reset(self):
        self.background=None
        self.boxes=[]
        self.chars=[]
        self.alphas=[]
        
    def create_image(self):
        self.reset()
        self.generate_backround()
        self.generate_noise(random.randrange(0,20))
        self.generate_boxes(random.randrange(5,11))
        for i in range(len(self.boxes)):
            xy=self.boxes[i]
            alpha=self.alphas[i]
            shape=random.choice(L[self.chars[i]])
            self.draw_shape(shape,xy,alpha)




H=[(0,100),(150,300),(320,460),(480,620),(650,794)]
W=[(0,80),(110,186),(200,300),(310,410),(420,520),(530,630),(630,740),(740,850),(850,950),(950,1060) ]
im=cv2.imread('../input/numbers-and-letters/all_digits.png')
L=[]
for i in range(len(W)):
    L.append([])
    for j in range(len(H)):
        m=im[H[j][0]:H[j][1],W[i][0]:W[i][1]]
        #m=cv2.resize(m,(50,70))[:,:,0]
        L[i].append(m[:,:,0])
        
im=cv2.imread('../input/numbers-and-letters/all_letters.png')
W=[(0,240),(250,420),(440,610),(610,710),(720,790),
   (800,850),(850,900),(900,970),(970,1070),(1070,1100),(1100,1300)]

L.append([])
for w in W:
    m=im[:,w[0]:w[1],0]
    L[-1].append(m)
    L.append([])

L[-1].append(cv2.imread('../input/numbers-and-letters/alif.png')[:,:,0])
    
L.append([])
for f in os.listdir("../input/numbers-and-letters"):
    if 'b' in f:
        m=cv2.imread("../input/numbers-and-letters/"+f)[:,:,0]
        L[-1].append(m)
        
L.append([])
for f in os.listdir("../input/numbers-and-letters"):
    if 'khrb' in f:
        m=cv2.imread("../input/numbers-and-letters/"+f)[:,:,0]
        L[-1].append(m)
