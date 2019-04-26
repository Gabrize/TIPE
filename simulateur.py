import math
import numpy as np
from PIL import Image
import random
import copy
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation
import skfmm

conv = 0.01 #1 unité = conv mètres
#définition de la population (unités : m, s, N/m, kg) 
param = {
            "nb_piétons" : 10,
            "k" : 1.2e5,
            "k_o" : 1.2e5,
             "t_relaxation" : 0.5,
             "h" : 0.02,
             "rayon" : 0.2/conv,
             "taille_carte" : 2000,
             "sortie" : (1000,1), 
             "zone_sortie" : (0,0,2000,130), #(x1,y1,x2,y2))
             "masse" : 60,
             "largeur_porte" : 1/conv,
             "vitesse_moyenne" : 1.34/conv,
             "ecart_type"  : 0.26/conv}
n = param["taille_carte"]
segments = [[(0,0), (0,n-1)], [(0,n-1), (n-1,n-1)], [(n-1,n-1), (n-1,0)], [(n-1,0), (0,0)], [(0,130), ((n-int(param["largeur_porte"]))//2,130)] , [((n+int(param["largeur_porte"]))//2,130), (n-1,130)]]
cercles = []#[(centre), rayon]
obstacle = ()#r, g, p
domaine = [(0,1.5),  (0,3), (0,3)]
def rectifier(ind):
    rep = [[] for i in "aaa"]
    for i in range(len(ind)) :
        rep[i] = max(domaine[i][0], min(ind[i], domaine[i][1]))
    return rep
def optimize():
    F = 0.5
    mutation_rate = 0.1
    lamb = 0.1
    max_generations = 2#20
    taille_pop = 10 #
    rep = [0 for i in range(max_generations)]
    v = np.array([[0,0,0] for i in range(taille_pop)])
    u = np.array([[0,0,0] for i in range(taille_pop)])
    e = np.array([0 for i in range(taille_pop)])
    i = 0
    pop = np.array([ [random.random()*abs(domaine[i][0]-domaine[i][1]) for i in range(3)] for j in range(taille_pop)])
    for i in range(max_generations):
        jrand = random.randrange(taille_pop)
        e_pop = evaluer(pop, aff_image = True)
        print(min(e_pop))
        j_best = 0
        best = np.inf
        for i in range(taille_pop):
            if e[i] < best :
                best = e[i]
                j_best = i
        for i in range(taille_pop):
            r1, r2 = [random.randrange(taille_pop) for i in range(2)]
            v[i] = v[i]  + lamb*(pop[j_best]-pop[i]) + F*(pop[r1]-pop[r2])
        for i in range(taille_pop):
            r = random.random()
            if r < mutation_rate or i == jrand :
                u[i] = rectifier(v[i])
            else :
                u[i] = pop[i]
        e_u = evaluer(u, aff_image = True)
        print("v :", v)
        print("u :", u)
        print("e_u :", e_u)
        print("pop :", pop)
        print("e_p :", e_pop)
        for i in range(i):
            if e_u[i] <= e_pop[i] :
                pop[i] = u[i]
        rep[i] = e_u.min()
    return pop, rep

        
def evaluer(pop, aff_image = False):
    global cercles
    taille = len(pop)
    n = param["taille_carte"]
    set_nb_pietons(20)
    e = np.array([0. for i in range(taille)])
    for i in range(taille):
        print(i+1, "/", taille, end = " ")
        r, g, p = pop[i]
        cercles = [[(n//2+p/conv, 130+g/conv), r/conv]]
        S = init_carte(segments, cercles)
        if aff_image :
            afficher_image(S)
        F = fast_marching(S)
        e[i] = main(F, affiche_im = False)
    print(e)
    return e
        
def load():
    return np.load("./fast_marching_obstacle_2.npy")

def detendre():
    param["t_relaxation"] += 0.2
    

# fonction affichage
def ecretage(M):
    M[M == np.inf] = 0
    return M
def couleur_orange(a):
    if int(a*255) == 0:
        return (255,255,255)
    return(255, int(a*255),0)

def couleur(t):
    return(int(255//2*np.cos(4*np.pi*t)+255//2), int(255//2*np.sin(4*np.pi*t)+255//2), int(255/2/np.pi*t))

# fonctions sur les vecteurs
def entier(v):
    x,y = v
    return (int(x) , int(y))
def produit_scalaire(v1, v2):
    x1, y1 = v1
    x2, y2 = v2
    return x1*x2+y1*y2

def dist_segment_point(extremite_1, extremite_2, point): #renvoie la distance et le vecteur
                                                                                    #directeur du segment P'P avec P' le projeté du point sur le segment
    v = sub(extremite_2, extremite_1)
    t  = produit_scalaire(v, sub(point, extremite_1))/norme(v)**2
    t = min(max(0,t), 1)
    r = sub(add(extremite_1, mult_scal(t, v)), point)
    return norme(r), r
    
def norme(v):
    return np.sqrt((v[0])**2+(v[1])**2)

def add(v1, v2):
    x1,y1 = v1
    x2,y2 = v2
    return (x1+x2,y1+y2)

def mult_scal(x, v):
    return (x*v[0], x*v[1])

def sub(v1, v2):
    return add(v1, mult_scal(-1, v2))

def normalisé(v):
    return mult_scal(1/norme(v), v)

# fonctions sur les positions
def d(i, j, q):
    return norme(sub(q[i], q[j])) - 2*param["rayon"]

def el(coor,t):
    i,j = coord
    return t[i][j]

def el1D(coor, l):
    h, l = M.shape
    x, y = coor
    x, y = int(x), int(y)
    return l[x*l+y]

def gen_unique(D, q, k):
    h,l = D.shape
    i, j = (random.randint(0,h-1) , random.randint(0,l-1))
    while D[i,j] == np.inf or est_sorti((i,j)) or detecter_contact(k, q, (i,j)) :
        i, j = random.randint(0,h-1) , random.randint(0,l-1)
    return (i,j)


def gen(D, n, base = []): #retourne un vecteur de n coordonnées distinctes et alétaoires en dehors des obstacles en complétant base       
    h,l = D.shape
    res = [(-1,-1) for i in range(n)]
    if base != []:
        for i in range(len(base)):
            res[i] = base[i]
    for k in range(len(base), n):
        res[k] = gen_unique(D, res, k)
    return res


# Fast marching
def fast_marching(carte): 
 return skfmm.distance(carte)

#Initialisation
def init_carte(segments, cercles):
    n = param["taille_carte"]
    S = np.ones( (n,n) ) #les états des points de la grille pour l'algorithme
    X, Y = np.meshgrid(np.linspace(0,n-1,n), np.linspace(0,n-1,n))
    s1, s2 = param["sortie"]
    S[s1,s2] = 0
    #obstacles

    for cercle in cercles :
        centre, rayon = cercle
        x, y = centre
        mask = np.sqrt((Y-x)**2+(X-y)**2) <=rayon
        S  = np.ma.MaskedArray(S, mask)
 
    for s  in segments :
        e1, e2 = s
        x1, y1 = e1
        x2, y2 = e2
        mask1 = np.logical_and(np.logical_and(x1<=Y, Y<=x2), np.logical_and(y1<=X, X<= y2))
        mask2 = np.logical_and(np.logical_and(x1>=Y, Y>=x2), np.logical_and(y1>=X, X>= y2))
        mask = np.logical_or(mask1, mask2)
        S  = np.ma.MaskedArray(S, mask)
        
    return S

def set_nb_pietons(n):
    param["nb_piétons"] = n
    

#forces
def f_contact(q, est_sorti): #renvoie le vecteur f_c
    n = param["nb_piétons"]
    k = param["k"]
    f = [(0,0) for i in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j  and not(est_sorti[j]):
                f[i] =  add(f[i], mult_scal(k*min(0, d(i,j,q)), normalisé(sub(q[j],q[i]))))
    return f

def f_c_ob(q, sorti, segment, cercle):
    n = param["nb_piétons"]
    r = param["rayon"]
    f = [(0,0) for i in range(n)]
    k_o= param["k_o"]
    for i in range(n):
        if not(sorti[i]):
            for ob in segment :
                ex1, ex2 = ob
                d, v = dist_segment_point(ex1, ex2, q[i])
                f[i] =  add(f[i], mult_scal(k_o*min(0, d-r), normalisé(v)))
            for ob in cercle :
                centre, rc = ob
                v = sub(centre, q[i])
                d = norme(v)
                f[i] = add(f[i], mult_scal(k_o*min(0, d-r-rc), normalisé(v)))
    return f    
    
def f_a(coor,  G, u_courante): ##coor : (x,y) G gradient de la distance à la sortie
    x,y = coor
    x , y = int(x), int(y)
    v_d = np.random.normal(param["vitesse_moyenne"],param["ecart_type"])
    t_r = param["t_relaxation"]
    m = param["masse"]
    e = normalisé((G[0][x,y],G[1][x,y]))
    return mult_scal(m/t_r, sub(mult_scal(-v_d, e), u_courante))
#def f_int(n, q,  carte):

def est_sorti( q):
    i,j = q
    x1,y1,x2,y2 = param["zone_sortie"]
    if i>= x1 and i <= x2 and j<= y2 and j>= y1 :
        return True
    return False

def afficher_image(D) :  #D matrice
    D = abs(D)
    D[D == np.inf] = 1000000
    im = Image.new("L",(D.shape[1],D.shape[0]))
    im.putdata(list((255*D/D.max()).flat))
    im.show()

def detecter_contact(i, q, coor): #renvoie True si (x,y) est en contact avec qqch, False sinon    
    n = param["nb_piétons"]
    k = param["k"]
    r = param["rayon"]
    for j in range(n):
        if i != j :
            if norme(sub(q[j], coor)) - 2*r < 0 :
                return True
    for ob in segments :
        ex1, ex2 = ob
        d, v = dist_segment_point(ex1, ex2, coor)
        if d-r < 0 :
            return True
    for ob in cercles :
        centre, rc = ob
        v = sub(centre, coor)
        d = norme(v)
        if d-r-rc < 0:
            return True
    return False

def main( M, affiche_pos = False, affiche_im = True, save = True, pos_init = []):
    D = np.copy(M)
    nb_iter = 2500
    compt = 0
    # C est pour l'affichage
    C = np.copy(D)
    C[C == np.inf] = 0
    C = (C/C.max()).flat
    C = list(map(couleur_orange, C))
    vert = (0,255,0)
    
    h = param["h"] #s
    n = param["nb_piétons"]
    m = param["masse"]
    rayon = param["rayon"]
    Q = []
    X = [[] for i in range(n)]
    Y = [[] for i in range(n)]
    chrono = 0
    if len(pos_init) != 0 :
        assert len(pos_init) == n
        q = copy.deepcopy(pos_init)
    else :
        q = gen(D, n) #positions

    u = [mult_scal(param["vitesse_moyenne"], (random.random(),random.random())) for i in range(n)] #vitesses
    sorti = [ False for i in range(n)]
    D[abs(D) == np.inf] = 10000
    G = np.gradient(D)
    temp = [] # init temp
    while False in sorti and compt <50000:
 #       if compt%100 == 0:
 #           print(sorti.count(True))
        Q.append(copy.deepcopy(q))
        if affiche_pos :
            print(q)
        chrono += h
        for k in range(n):
            C[D.shape[1]*int(q[k][0])+int(q[k][1])] = vert
        f_c = f_contact(q,  sorti)
        f_c_o = f_c_ob(q, sorti, segments, cercles)
        for p in range(n):
            if not(sorti[p]):
                u[p] = add(u[p], mult_scal(h/m, add( add(f_a(q[p],  G, u[p]), f_c[p]) , f_c_o[p])))
                q[p] = add(q[p], mult_scal(h,u[p]))
                if est_sorti( q[p]):
                    sorti[p] = True
        compt+=1
    if save or affiche_im :
        im2 = Image.new("RGB",(D.shape[1],D.shape[0]))
        im2.putdata(C)
        if affiche_im :
            im2.show()
        if save :
            im2.save(str(n)+" piétons en "+str(np.round(chrono, 2))+"s"+", "+"couleur avec trajectoire"+str(random.random())+".png", format = 'PNG')
    if affiche_im :
        for pos in Q :
            for i in range(n):
                x,y = pos[i]
                X[i].append(x)
                Y[i].append(y)
        for i in range(n):
            plt.plot(Y[i], X[i])
        plt.show()
        plt.close()
    
    return chrono
        
def moyenne_carte_G(n,G, affiche_pos = False, save = True):
    s = 0
    
    print("|", end = "")
    
    for i in range(n):
        s+= main( G, affiche_pos=affiche_pos, affiche_im = False, save = save)
        
        print("*", end = "")
        if (i+1)%10 == 0 :
            print("|", end = "")
            
    print()
    return s/n

def exhiber_exemples(pietons, av_ob, ss_ob, nb_exemples):
    compt_exemples = 0
    compt_tot = 0
    rep = [[] for i in range(nb_exemples)]
    param["nb_piétons"]= pietons
    while compt_exemples < nb_exemples :
        compt_tot += 1
        p = gen(tri, param["nb_piétons"])
        t1 = main( av_ob, affiche_im = False, pos_init = p)
        t2 = main( ss_ob, affiche_im = False, pos_init = p)
        print(t1, t2)
        if t1<t2 :
            rep[compt_exemples] = [t1, t2, p]
            compt_exemples += 1
    print(compt_exemples/compt_tot)
    return rep

Gao = np.load("fast_marching_petit_obstacle.npy")
Gso = np.load("fast_marching_sans_obstacle_large_porte.npy")
#p_i = gen(Gao, param["nb_piétons"])

            
    
def flux( M, temps, affiche_pos = False, affiche_im = True, save = True, pos_init = []):
    D = np.copy(M)
    nb_iter = 2500
    compt = 0
    # C est pour l'affichage
    C = np.copy(D)
    C[C == np.inf] = 0
    C = (C/C.max()).flat
    C = list(map(couleur_orange, C))
    vert = (0,255,0)
    nb_sorti = 0
    h = param["h"] #s
    n = param["nb_piétons"]
    m = param["masse"]
    rayon = param["rayon"]
    U = [0 for i in range(int(temps/h)+1)]
    Q = []
    X = [[] for i in range(n)]
    Y = [[] for i in range(n)]
    chrono = 0
    if len(pos_init) != 0 :
        assert len(pos_init) == n
        q = copy.deepcopy(pos_init)
    else :
        q = gen(D, n) #positions

    u = [mult_scal(1.34, (random.random(),random.random())) for i in range(n)] #vitesses
    sorti = [ False for i in range(n)]
    D[abs(D) == np.inf] = 10000
    G = np.gradient(D)
    temp = [] # init temp
    while chrono < temps:
        try :
            U[compt] = norme(u[0])
     #       if compt%100 == 0:
     #           print(sorti.count(True))
            Q.append(copy.deepcopy(q))
            if affiche_pos :
                print(q)
            chrono += h
            for k in range(n):
                C[D.shape[1]*int(q[k][0])+int(q[k][1])] = vert
            f_c = f_contact(q,  sorti)
            f_c_o = f_c_ob(q, sorti, segments, cercles)
            for p in range(n):
                if not(sorti[p]):
                    u[p] = add(u[p], mult_scal(h/m, add( add(f_a(q[p],  G, u[p]), f_c[p]) , f_c_o[p])))
                    q[p] = add(q[p], mult_scal(h,u[p]))
                    if est_sorti( q[p]):
                        q[p] = gen_unique(M, q, p)
                        nb_sorti +=1
            compt+=1
        except :
            print("erreur")
            break
    if save or affiche_im :
        im2 = Image.new("RGB",(D.shape[1],D.shape[0]))
        im2.putdata(C)
        if affiche_im :
            im2.show()
        if save :
            im2.save(str(n)+" piétons en "+str(np.round(chrono, 2))+"s"+", "+"couleur avec trajectoire"+str(random.random())+".png", format = 'PNG')
    if affiche_im :
        for pos in Q :
            for i in range(n):
                x,y = pos[i]
                X[i].append(x)
                Y[i].append(y)
        for i in range(n):
            plt.plot(Y[i], X[i])
        plt.show()
        plt.close()
    Q = np.array(Q)
    np.save("Q", Q)
    return nb_sorti/chrono

def flux_ss_im( M, temps, pos_init = []):
    flag = False
    top = 0
    D = np.copy(M)
    compt = 0
    nb_sorti = 0
    h = param["h"] #s
    n = param["nb_piétons"]
    m = param["masse"]
    rayon = param["rayon"]
    sorti = [False for i in range(n)]
    chrono = 0
    if len(pos_init) != 0 :
        assert len(pos_init) == n
        q = copy.deepcopy(pos_init)
    else :
        q = gen(D, n) #positions
    u = [mult_scal(1.34, (random.random(),random.random())) for i in range(n)] #vitesses
    D[abs(D) == np.inf] = 10000
    G = np.gradient(D)
    temp = [] # init temp
    while chrono < temps:
        try :
            chrono += h
            f_c = f_contact(q,  sorti)
            f_c_o = f_c_ob(q, sorti, segments, cercles)
            for p in range(n):
                u[p] = add(u[p], mult_scal(h/m, add( add(f_a(q[p],  G, u[p]), f_c[p]) , f_c_o[p])))
                q[p] = add(q[p], mult_scal(h,u[p]))
                if est_sorti(q[p]):
                    if flag == False :
                        top = chrono
                        flag = True
                    q[p] = gen_unique(M, q, p)
                    nb_sorti +=1
                    print("*", end = "")
            compt+=1
        except :
            print("erreur")
            break
    print()
    print(top, chrono, nb_sorti)
    return nb_sorti/(chrono-top)
def flux_ss_im_echantillon( M, temps, nb_batch, pos_init = []):
    r = [0 for i in range(nb_batch)] 
    D = np.copy(M)
    h = param["h"] #s
    n = param["nb_piétons"]
    m = param["masse"]
    rayon = param["rayon"]
    sorti = [False for i in range(n)]
    if len(pos_init) != 0 :
        assert len(pos_init) == n
        q = copy.deepcopy(pos_init)
    else :
        q = gen(D, n) #positions
    u = [mult_scal(1.34, (random.random(),random.random())) for i in range(n)] #vitesses
    D[abs(D) == np.inf] = 10000
    G = np.gradient(D)

    for i in range(nb_batch):
        print(str(i+1)+"/"+str(nb_batch))
        chrono = 0
        compt = 0
        nb_sorti = 0
        while chrono < temps:
            try :
                chrono += h
                f_c = f_contact(q,  sorti)
                f_c_o = f_c_ob(q, sorti, segments, cercles)
                for p in range(n):
                    u[p] = add(u[p], mult_scal(h/m, add( add(f_a(q[p],  G, u[p]), f_c[p]) , f_c_o[p])))
                    q[p] = add(q[p], mult_scal(h,u[p]))
                    if est_sorti(q[p]):
                        q[p] = gen_unique(M, q, p)
                        nb_sorti +=1
                        print("*", end = "")
                compt+=1
            except :
                print("erreur")
                break
        print()
        print(chrono, nb_sorti)
        r[i] = nb_sorti/chrono
        np.save("r", r)
    return r
def anim(): 
    fig = plt.figure()
    fig.set_dpi(100)
    fig.set_size_inches(7, 6.5)

    q = np.load("Q.npy")
    n = len(q[0])
    ax = plt.axes(xlim=(0, 200), ylim=(0, 200))
    patch = [plt.Circle((q[0][i][0], q[0][i][1]), 3) for i in  range(n)]

    def init():
        for i in range(n) :
            patch[i].center = (q[0, i][0], q[0, i][1])
            ax.add_patch(patch[i])
        return patch,

    def animate(i):
        for j in range(n):
            x, y = patch[j].center
            x = q[i, j][0]
            y = q[i, j][1]
            patch[j].center = (x, y)
            coll = matplotlib.collections.PatchCollection(patch, facecolors='black')
        return coll,

    anim = animation.FuncAnimation(fig, animate, 
                                   init_func=init, 
                                   frames=len(q)-1, 
                                   interval= 1,
                                   blit=False)

    plt.show()
