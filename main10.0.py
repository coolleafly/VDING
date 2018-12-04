from itertools import combinations
from itertools import permutations
import xlrd,xlwt
import numpy as np
import argparse
import sys

class XYpoint:
    def __init__(self,x=0,y=0,z=0,Mainname='O',Subname='0'):
        self.x = x
        self.y = y
        self.z = z
        self.mn=Mainname
        self.sn=Subname
        self.tmp=0
        self.name=str(self.mn)+'_'+str(self.sn)

    def setname(self):
        self.name=str(self.mn)+'_'+str(self.sn)

    def length0(self):
        tmp=(self.x**2+self.y**2+self.z**2)**0.5
        return tmp

    def lengthFromPoint(self,p):
        tmp = ((self.x-p.x) ** 2 + (self.y-p.y) ** 2 + (self.z-p.z) ** 2) ** 0.5
        return tmp

    def sumlength(self,*pp):
        s=0
        for p in pp:
            s+=self.lengthFromPoint(p)
        return s

class XYimplant:
    def __init__(self,Ic,Ia):
        self.Ic=Ic
        self.Ia=Ia

class XYtransfer:
    def __init__(self,p1,p2,p3):
        s=self
        self.x1 = p1.x
        self.y1 = p1.y
        self.z1 = p1.z

        self.x2 = p2.x
        self.y2 = p2.y
        self.z2 = p2.z

        self.x3 = p3.x
        self.y3 = p3.y
        self.z3 = p3.z

        self.xx = (s.x2 - s.x1) / (((s.x2 - s.x1) ** 2 + (s.y2 - s.y1) ** 2 + (s.z2 - s.z1) ** 2) ** 0.5)
        self.xy = (s.y2 - s.y1) / (((s.x2 - s.x1) ** 2 + (s.y2 - s.y1) ** 2 + (s.z2 - s.z1) ** 2) ** 0.5)
        self.xz = (s.z2 - s.z1) / (((s.x2 - s.x1) ** 2 + (s.y2 - s.y1) ** 2 + (s.z2 - s.z1) ** 2) ** 0.5)
        self.h = ((s.x2 - s.x1) * (s.x3 - s.x1) + (s.y2 - s.y1) * (s.y3 - s.y1) + (s.z2 - s.z1) * (s.z3 - s.z1)) / ((((s.x2 - s.x1) ** 2 + (s.y2 - s.y1) ** 2 + ( s.z2 - s.z1) ** 2) * ((s.x3 - s.x1) ** 2 + (s.y3 - s.y1) ** 2 + (s.z3 - s.z1) ** 2)) ** 0.5)
        self.j = (((s.x3 - s.x1) ** 2 + (s.y3 - s.y1) ** 2 + (s.z3 - s.z1) ** 2) / ((s.x2 - s.x1) ** 2 + (s.y2 - s.y1) ** 2 + (s.z2 - s.z1) ** 2)) ** 0.5 * s.h
        self.yx = ((s.x3 - s.x1) + (s.x1 - s.x2) * s.j) / ((((s.x3 - s.x1) ** 2 + (s.y3 - s.y1) ** 2 + (s.z3 - s.z1) ** 2) * (1 - s.h ** 2)) ** 0.5)
        self.yy = ((s.y3 - s.y1) + (s.y1 - s.y2) * s.j) / ((((s.x3 - s.x1) ** 2 + (s.y3 - s.y1) ** 2 + (s.z3 - s.z1) ** 2) * (1 - s.h ** 2)) ** 0.5)
        self.yz = ((s.z3 - s.z1) + (s.z1 - s.z2) * s.j) / ((((s.x3 - s.x1) ** 2 + (s.y3 - s.y1) ** 2 + (s.z3 - s.z1) ** 2) * (1 - s.h ** 2)) ** 0.5)
        self.zx = ((s.y2 - s.y1) * (s.z3 - s.z1) - (s.y3 - s.y1) * (s.z2 - s.z1)) / ((((s.y2 - s.y1) * (s.z3 - s.z1) - (s.y3 - s.y1) * (s.z2 - s.z1)) ** 2 + ((s.x3 - s.x1) * (s.z2 - s.z1) - (s.x2 - s.x1) * (s.z3 - s.z1)) ** 2 + ((s.x2 - s.x1) * (s.y3 - s.y1) - (s.x3 - s.x1) * (s.y2 - s.y1)) ** 2) ** 0.5)
        self.zy = ((s.x3 - s.x1) * (s.z2 - s.z1) - (s.x2 - s.x1) * (s.z3 - s.z1)) / ((((s.y2 - s.y1) * (s.z3 - s.z1) - (s.y3 - s.y1) * (s.z2 - s.z1)) ** 2 + ((s.x3 - s.x1) * (s.z2 - s.z1) - (s.x2 - s.x1) * (s.z3 - s.z1)) ** 2 + ((s.x2 - s.x1) * (s.y3 - s.y1) - (s.x3 - s.x1) * (s.y2 - s.y1)) ** 2) ** 0.5)
        self.zz = ((s.x2 - s.x1) * (s.y3 - s.y1) - (s.x3 - s.x1) * (s.y2 - s.y1)) / ((((s.y2 - s.y1) * (s.z3 - s.z1) - (s.y3 - s.y1) * (s.z2 - s.z1)) ** 2 + ((s.x3 - s.x1) * (s.z2 - s.z1) - (s.x2 - s.x1) * (s.z3 - s.z1)) ** 2 + ((s.x2 - s.x1) * (s.y3 - s.y1) - (s.x3 - s.x1) * (s.y2 - s.y1)) ** 2) ** 0.5)

    def p(self,p=XYpoint()):
        s=self
        x5 = p.x
        y5 = p.y
        z5 = p.z
        a = -(s.x1 * s.yy * s.zz - s.x1 * s.yz * s.zy - x5 * s.yy * s.zz + x5 * s.yz * s.zy - s.y1 * s.yx * s.zz + s.y1 * s.yz * s.zx + y5 * s.yx * s.zz - y5 * s.yz * s.zx + s.yx * s.z1 * s.zy - s.yy * s.z1 * s.zx - s.yx * z5 * s.zy + s.yy * z5 * s.zx) / (s.xx * s.yy * s.zz - s.xx * s.yz * s.zy - s.xy * s.yx * s.zz + s.xy * s.yz * s.zx + s.xz * s.yx * s.zy - s.xz * s.yy * s.zx)
        b = (s.x1 * s.xy * s.zz - s.x1 * s.xz * s.zy - x5 * s.xy * s.zz + x5 * s.xz * s.zy - s.xx * s.y1 * s.zz + s.xz * s.y1 * s.zx + s.xx * y5 * s.zz - s.xz * y5 * s.zx + s.xx * s.z1 * s.zy - s.xy * s.z1 * s.zx - s.xx * z5 * s.zy + s.xy * z5 * s.zx) / (s.xx * s.yy * s.zz - s.xx * s.yz * s.zy - s.xy * s.yx * s.zz + s.xy * s.yz * s.zx + s.xz * s.yx * s.zy - s.xz * s.yy * s.zx)
        c = -(s.x1 * s.xy * s.yz - s.x1 * s.xz * s.yy - x5 * s.xy * s.yz + x5 * s.xz * s.yy - s.xx * s.y1 * s.yz + s.xz * s.y1 * s.yx + s.xx * y5 * s.yz - s.xz * y5 * s.yx + s.xx * s.yy * s.z1 - s.xy * s.yx * s.z1 - s.xx * s.yy * z5 + s.xy * s.yx * z5) / (s.xx * s.yy * s.zz - s.xx * s.yz * s.zy - s.xy * s.yx * s.zz + s.xy * s.yz * s.zx + s.xz * s.yx * s.zy - s.xz * s.yy * s.zx)
        p2=XYpoint(a,b,c,'N',str(p.mn)+str(p.sn))
        return p2

class XYimplant:
    def __init__(self):
        self.Ic=XYpoint()
        self.Ia=XYpoint()

    def ILength(self):
        return self.Ic.lengthFromPoint(self.Ia)

class XYrecord(XYimplant):
    def __init__(self):
        #global LandmarkN,ImplantPointN
        self.Landmark=[XYpoint() for i in range(LandmarkN+1)]
        self.Implantpoint=[XYpoint() for i in range(ImplantPointN+1)]
        self.Implantpointi=2
        super(XYrecord, self).__init__()
        self.Implantpoint[1] = self.Ic
        self.Implantpoint[2] = self.Ia

class XYcase:
    def __init__(self):
        self.plan=XYrecord()
        self.real=XYrecord()
        self.bestZuhe = []
        self.bestUnzuhe = []
        self.AveEU,self.MaxEU,self.MinEU,self.deltaSum=[0,0,0,0]
        self.pImplant = XYimplant()
        self.rImplant = XYimplant()
        self.r2Implant = XYimplant()
        self.minErrPoint=[]


    def abserr(self,L1,L2):
        length1=self.plan.Landmark[L1].lengthFromPoint(self.plan.Landmark[L2])
        length2=self.real.Landmark[L1].lengthFromPoint(self.real.Landmark[L2])
        return abs(length1-length2)/length1
        # return abs(length1-length2)/(0.5*(length1+length2))

    def abstheta(self,L,L2,L3):
        a1=self.plan.Landmark[L].lengthFromPoint(self.plan.Landmark[L2])
        a2=self.real.Landmark[L].lengthFromPoint(self.real.Landmark[L2])
        b1=self.plan.Landmark[L].lengthFromPoint(self.plan.Landmark[L3])
        b2=self.real.Landmark[L].lengthFromPoint(self.real.Landmark[L3])
        c1=self.plan.Landmark[L3].lengthFromPoint(self.plan.Landmark[L2])
        c2=self.real.Landmark[L3].lengthFromPoint(self.real.Landmark[L2])

        costheta1=((a1**2+b1**2-c1**2)/(2*b1*c1))
        costheta2=((a2**2+b2**2-c2**2)/(2*b2*c2))
        sintheta1=(1-(costheta1**2))**0.5
        sintheta2=(1-(costheta2**2))**0.5

        sintheta=abs(sintheta1*costheta2-sintheta2*costheta1)
        # print(a1, b1, c1, ':', a2, b2, c2, '::', costheta1, costheta2,sintheta1, sintheta2, sintheta)

        return sintheta

    def isLongerLHelf(self,L,L2,L3):
        a1=self.plan.Landmark[L].lengthFromPoint(self.plan.Landmark[L2])
        a2=self.real.Landmark[L].lengthFromPoint(self.real.Landmark[L2])
        b1=self.plan.Landmark[L].lengthFromPoint(self.plan.Landmark[L3])
        b2=self.real.Landmark[L].lengthFromPoint(self.real.Landmark[L3])
        c1=self.plan.Landmark[L3].lengthFromPoint(self.plan.Landmark[L2])
        c2=self.real.Landmark[L3].lengthFromPoint(self.real.Landmark[L2])
        res=True
        weight=0.5
        if a1<weight*self.LandmrakMean: res=False
        if a2<weight*self.LandmrakMean: res=False
        if b1<weight*self.LandmrakMean: res=False
        if b2<weight*self.LandmrakMean: res=False
        if c1<weight*self.LandmrakMean: res=False
        if c2<weight*self.LandmrakMean: res=False
        return res

    def meanLandmarkLine(self):
        k=0
        sum=0
        for i in range(1,LandmarkN):
            for j in range(i+1,LandmarkN+1):
                k+=2
                sum+=self.plan.Landmark[i].lengthFromPoint(self.plan.Landmark[j])
                sum+=self.real.Landmark[i].lengthFromPoint(self.plan.Landmark[j])

        mean=sum/k
        self.LandmrakMean=mean



    def accuracy(self,HSc=False,Sc=0):
        #update rImplant
        if Sc=='/': Sc=0
        if HSc:
            self.r2Implant.Ia=self.rImplant.Ia
            Rc=self.rImplant.Ic
            Ra=self.rImplant.Ia
            Rlength=self.rImplant.ILength()
            self.r2Implant.Ic.x=Rc.x+(Ra.x-Rc.x)*Sc/Rlength
            self.r2Implant.Ic.y=Rc.y+(Ra.y-Rc.y)*Sc/Rlength
            self.r2Implant.Ic.z=Rc.z+(Ra.z-Rc.z)*Sc/Rlength
        else:
            self.r2Implant=self.rImplant

        Rc = self.r2Implant.Ic
        Ra = self.r2Implant.Ia
        Pc = self.pImplant.Ic
        Pa = self.pImplant.Ia

        self.CGD=Pc.lengthFromPoint(Rc)
        self.AGD=Pa.lengthFromPoint(Ra)
        costheta=((Pa.x-Pc.x)*(Ra.x-Rc.x)+(Pa.y-Pc.y)*(Ra.y-Rc.y)+(Pa.z-Pc.z)*(Ra.z-Rc.z))/(Pc.lengthFromPoint(Pa)*Rc.lengthFromPoint(Ra))
        #print(costheta)
        thetaRad=np.arccos(costheta)
        self.thetaDeg = np.rad2deg(thetaRad)
        self.AD = self.thetaDeg
        #print(self.AD)

        cosbetaC=((Pa.x-Pc.x)*(Rc.x-Pc.x)+(Pa.y-Pc.y)*(Rc.y-Pc.y)+(Pa.z-Pc.z)*(Rc.z-Pc.z))/(Pc.lengthFromPoint(Pa)*Pc.lengthFromPoint(Rc))
        #print(cosbeta)
        self.CVD=Rc.lengthFromPoint(Pc)*cosbetaC
        self.CLD=Rc.lengthFromPoint(Pc)*(1-cosbetaC**2)**0.5

        cosbetaA=((Pc.x-Pa.x)*(Ra.x-Pa.x)+(Pc.y-Pa.y)*(Ra.y-Pa.y)+(Pc.z-Pa.z)*(Ra.z-Pa.z))/(Pa.lengthFromPoint(Pc)*Pa.lengthFromPoint(Ra))
        self.AVD=Ra.lengthFromPoint(Pa)*cosbetaA
        self.ALD=Ra.lengthFromPoint(Pa)*(1-cosbetaA**2)**0.5

        bz= (-self.CVD)/(Pc.lengthFromPoint(Pa)-self.CVD-self.AVD)
        self.RDC = (((Ra.x - Rc.x) * bz + Rc.x - Pc.x) ** 2 + ((Ra.y - Rc.y) * bz + Rc.y - Pc.y) ** 2 + ((Ra.z - Rc.z) * bz + Rc.z - Pc.z) ** 2) ** 0.5
        self.RDA = (((Ra.x - Rc.x) * bz + Ra.x - Pa.x) ** 2 + ((Ra.y - Rc.y) * bz + Ra.y - Pa.y) ** 2 + ((Ra.z - Rc.z) * bz + Ra.z - Pa.z) ** 2) ** 0.5



class XYxlwt:
    def __init__(self,xlFileName,xlSheetName="S1",i=0,j=0):
        self.wb=xlwt.Workbook(encoding = 'ascii')
        self.ws=self.wb.add_sheet(xlSheetName)
        self.wb.save(xlFileName)
        self.xlFN=xlFileName
        self.xlSN=xlSheetName
        self.curri=i
        self.currj=j

    def save(self):
        self.wb.save(self.xlFN)

    def wt(self,*InT,wide=1):
        if InT!=():
            for T in InT:
                self.ws.write(self.curri,self.currj,label=str(T))
                self.currj+=wide

    def wtln(self,*InT):
        self.wt(*InT)
        self.curri+=1
        self.currj=0

    def wtpos(self,i=0,j=0,*InT):
        tmpi=i
        tmpj=j
        if InT!=():
            for T in InT:
                self.ws.write(tmpi,tmpj,label=str(T))
                tmpj+=1

    def setpos(self,i=None,j=None):
        self.curri=self.curri if i==None else i
        self.currj=self.curri if j==None else j

    def wtPoint(self,*pp):
        if pp!=():
            for p in pp:
                self.wt("%.3f"%p.x, "%.3f"%p.y,"%.3f"%p.z)

    def wtPointWithName(self,*pp):
        if pp!=():
            for p in pp:
                self.wt(p.name,"%.3f"%p.x, "%.3f"%p.y,"%.3f"%p.z)

########################################################################
def out1_line(id,PR,of):
    zh=mycase[id].bestZuhe    #zuhe[zhi]
    unzh=list(range(1,LandmarkN+1))
    for izh in range(3): unzh.remove(zh[izh])


    if PR=='P':
        PRnum = 1
        tmpRecord=mycase[id].plan
    else:
        PRnum = 2
        tmpRecord = mycase[id].real

    P1 = tmpRecord.Landmark[zh[0]]
    P2 = tmpRecord.Landmark[zh[1]]
    P3 = tmpRecord.Landmark[zh[2]]

    toN=XYtransfer(P1,P2,P3)

    of.wt(id,PRnum,zh)
    of.wtPoint(toN.p(P1),toN.p(P2),toN.p(P3))
    of.wt(unzh)
    for i in unzh:
        of.wtPoint(toN.p(tmpRecord.Landmark[i]))
    for i in range(1,tmpRecord.Implantpointi+1):
        of.wtPointWithName(toN.p(tmpRecord.Implantpoint[i]))
    of.wtln()

def out1(of):
    #
    #Make TiTle
    of.wt('ID','P/R','Coordinate Landmark Name')
    for i in range(1, 4): of.wt('C' + str(i)+'(x,y,z)', wide=3)
    of.wt('Other Landmark Name')
    for i in range(1, LandmarkN-3+1): of.wt('U' + str(i)+'(x,y,z)', wide=3)
    #for i in range(1, ImplantPointN + 1): of.wt('I_' + mycase[1].plan.Implantpoint[i].sn, wide=3)
    of.wtln('All Implant Points (name,x,y,z)')

    #Content
    for i in range(1,idN+1):
        # for j in range(len(zuhe)):
        out1_line(i,'P',of)
        out1_line(i,'R',of)
        of.save()

def out3(of3):
    # Make TiTle
    of3.wt('ID', 'Chosen', 'deltaSum', 'SysErr[-A-v-e-E-U-]', 'MaxEU', 'MinEU','Screw')
    of3.wtln('AD','CGD','CVD','CLD','AGD','AVD','ALD','RDC','RDA')
    #Content
    for i in range(1,idN+1):
        t=mycase[i]
        of3.wt(i, t.bestZuhe, t.deltaSum, t.AveEU, t.MaxEU, t.MinEU)
        screw=t.rImplant.ILength()-t.pImplant.ILength() if HasScrew else '/'
        #print(t.rImplant.ILength(),t.pImplant.ILength(),screw)
        of3.wt(screw)

        t.accuracy(HasScrew,screw)
        of3.wtln('%.3f'%t.AD,'%.3f'%t.CGD,'%.3f'%t.CVD,'%.3f'%t.CLD,'%.3f'%t.AGD,'%.3f'%t.AVD,'%.3f'%t.ALD,'%.3f'%t.RDC,'%.3f'%t.RDA)
        of3.save()




def killmaxErr(mcasei,mlist,of):
    mcase=mycase[mcasei]
    alldots = mlist
    max=-1
    maxj=0
    of.wt(mcasei,alldots,wide=3)

    for j in alldots:
        tmp=0
        for k in alldots:
            tmp+=mcase.abserr(j,k)
        if tmp>max:
            max=tmp
            maxj=j
        of.wt('%.3f'%tmp)

    anum=maxj
    of.wtln('','','OUT',anum)
    alldots.remove(anum)

    if len(alldots)>3:
        return killmaxErr(mcasei,alldots,of)
    else:
        of.wtln(mcasei,alldots,'GET IT!')
        of.save()
        return alldots





def killmaxtriangledot(mcasei,mlist,of ):
    mcase=mycase[mcasei]
    alldots = mlist
    alldotsErr=[]
    max=-1
    maxj=0
    of.wt(mcasei,alldots,wide=3)

    ii=-1
    for j2 in alldots:
        tmp=0
        ii+=1
        for j in range(len(zuhe2)):
            zh=zuhe2[j]
            if (j2 in zh) and (zh[0] in alldots) and (zh[1] in alldots) and (zh[2] in alldots):
                triangle=[mcase.abserr(zh[0],zh[1]),mcase.abserr(zh[0],zh[2]),mcase.abserr(zh[1],zh[2])]
                tmp+=np.sum(triangle)+(np.max(triangle)-np.min(triangle))
                # tmpzh=list(zh)+[]
                # tmpzh.remove(j2)
                # sintheta=mcase.abstheta(j2,tmpzh[0],tmpzh[1])
                # tmp += sintheta*mcase.LandmrakMean
        alldotsErr+=[tmp]
        if tmp>max:
            max=tmp
            maxj=j2
            maxii=ii
        of.wt('%.3f'%tmp)

    anum=maxj
    of.wtln('','','OUT',anum)
    alldots.remove(anum)
    del alldotsErr[maxii]


    if len(alldots)>3:
        res = killmaxtriangledot(mcasei,alldots,of)
        mcase.minErrPoint=mcase.minErrPoint+[anum]
        # print(mcase.minErrPoint,anum,[21])
        return res
    else:
        for m in range(2):
            for n in range(m+1,3):
                if alldotsErr[m]>alldotsErr[n]:
                    alldotsErr[m],alldotsErr[n]=[alldotsErr[n],alldotsErr[m]]
                    alldots[m],alldots[n]=[alldots[n],alldots[m]]

        of.wtln(mcasei,alldots,'GET IT!')
        mcase.minErrPoint=mcase.minErrPoint+alldots+[anum]
        of.save()




        mcase.bestZuhe=alldots
        zh=alldots

        tmpP = mcase.plan
        tmpR = mcase.real

        P1 = tmpP.Landmark[zh[0]]
        P2 = tmpP.Landmark[zh[1]]
        P3 = tmpP.Landmark[zh[2]]

        R1 = tmpR.Landmark[zh[0]]
        R2 = tmpR.Landmark[zh[1]]
        R3 = tmpR.Landmark[zh[2]]

        PtoN = XYtransfer(P1, P2, P3)
        RtoN = XYtransfer(R1, R2, R3)

        sumPC = P1.lengthFromPoint(P2) + P1.lengthFromPoint(P3) + P2.lengthFromPoint(P3)
        sumRC = R1.lengthFromPoint(R2) + P1.lengthFromPoint(R3) + R2.lengthFromPoint(R3)
        deltaSumC = abs(sumPC - sumRC)


        Cerr = []
        for k in zh:
            tmperr = PtoN.p(tmpP.Landmark[k]).lengthFromPoint(RtoN.p(tmpR.Landmark[k]))

            Cerr = Cerr + [tmperr]



        AveEC = np.mean(Cerr)

        PIc = PtoN.p(tmpP.Ic)
        PIa = PtoN.p(tmpP.Ia)
        RIc = RtoN.p(tmpR.Ic)
        RIa = RtoN.p(tmpR.Ia)
        mindelta = deltaSumC

        sysP= PtoN.p(tmpP.Landmark[anum])
        sysR= RtoN.p(tmpR.Landmark[anum])
        sysErr=sysP.lengthFromPoint(sysR)



        mycase[mcasei].bestZuhe=alldots

        mycase[mcasei].pImplant.Ic=PIc
        mycase[mcasei].pImplant.Ia=PIa
        mycase[mcasei].rImplant.Ic=RIc
        mycase[mcasei].rImplant.Ia=RIa
        mycase[mcasei].deltaSum=mindelta
        mycase[mcasei].AveEU=sysErr    #Sys Err for Dot 4



############temp#######################

        return alldots

def out7(of7):
    for i in range(1, idN + 1):
        mycase[i].meanLandmarkLine()

        alldots = list(range(1, LandmarkN + 1))
        minlist = killmaxtriangledot(i, alldots, of7)
        of7.wtln(mycase[i].minErrPoint)
        of7.save()


###########################################################################
#           XiangYa Implant Accuracy Compute Program
#                   Designed By Dr. Ye Liang
#                        2018.8   10.0
#                 Email: liangye@csu.edu.cn
#  Center of Stomatology, Xiangya Hospital, Central South University, China
###########################################################################

parser = argparse.ArgumentParser(description='-Lm = Landmark Number')
parser.add_argument('--Lm', type=int, default = 0)
args = parser.parse_args()
print(args.Lm)


data=xlrd.open_workbook('implantData.xlsx')
table=data.sheets()[0]
hang= table.nrows
lie= table.ncols

idN=round(float(table.cell(2,0).value))     #14 #总种植体数量

LandmarkN=round(float(table.cell(2,1).value))   #10 # 1..8
if args.Lm != 0:LandmarkN=args.Lm
print('Implants: %.0f'%table.cell(2,0).value,' Landmraks: ',LandmarkN)

ImplantPointN=round(float(table.cell(2,2).value))
HasScrew=(table.cell(2,3).value=='Y')

mycase=[XYcase() for i in range(idN+1)]

print('Reading Data ...')
for i in range(5,hang):
    idi=int(table.cell(i,0).value)
    ssi=int(table.cell(i,1).value)
    mainname=table.cell(i,2).value
    subname=table.cell(i,3).value
    #print(i,hang,idi,ssi,mainname,subname)

    t=mycase[idi].plan if ssi==1 else mycase[idi].real
    t2=''
    t2=t.Ic if mainname=='I' and subname=='C' else t2
    t2=t.Ia if mainname=='I' and subname=='A' else t2
    if mainname=='I' and subname!='A' and subname!='C':
        t.Implantpointi+=1
        t2=t.Implantpoint[t.Implantpointi]

    subname = int(subname) if mainname == 'L' else subname
    if not((mainname=='L') and (subname>LandmarkN)):
        t2=t.Landmark[int(subname)] if mainname=='L' else t2
        t2.x,t2.y,t2.z=[table.cell(i,4).value,table.cell(i,5).value,table.cell(i,6).value]
        t2.mn=mainname
        t2.sn=subname
        t2.setname()

print('Got [',idN,'] implants ...')

zuhe=(list(permutations(range(1,LandmarkN+1), 3)))
unzuhe=[list(range(1,LandmarkN+1)) for i in range(len(zuhe))]
tmp=[unzuhe[i].remove(zuhe[i][j]) for i in range(len(zuhe)) for j in range(3)]

zuhe2=(list(combinations(range(1,LandmarkN+1), 3)))
unzuhe2=[list(range(1,LandmarkN+1)) for i in range(len(zuhe2))]
tmp=[unzuhe2[i].remove(zuhe2[i][j]) for i in range(len(zuhe2)) for j in range(3)]



path=sys.path[0]+'\\'
ex='m8-L'+str(LandmarkN)

of7=XYxlwt(path+'out7'+ex+'.xls')
out7(of7)
print('[out7.xls] File is OK!')

of3=XYxlwt(path+'out3'+ex+'.xls')
out3(of3)
print('[out3.xls] File is OK!')

of1=XYxlwt(path+'out1'+ex+'.xls')
out1(of1)
print('[out1.xls] File is OK!')



pass

