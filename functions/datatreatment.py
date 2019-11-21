
import numpy as np # linear algebra

def weightedAverage(data, w, l):
    df = data.copy()
    columns = df.columns.values
    wa = np.zeros(df.shape[0])
    for i in range(len(l)):
        wa += (w[i] * df[columns[l[i]]]) / sum(w)
    df['wa'] = wa
    df = df.sort_values(by=['wa'], ascending=False)
    return df
    
def maximin(data, indices):
    
    df = data.copy()
    columns = df.columns.values 
    t = df.shape[0] 
    mn = np.zeros(t) 
    for i in range(t): 
        if i in df.index:
            mn[i] = df[columns[indices[0]]][i]
            for j in range(1, len(indices)):
                if mn[i] > df[columns[indices[j]]][i]:
                    mn[i] = df[columns[indices[j]]][i]
    df['minVal'] = mn
    df = df.sort_values(by=['minVal'], ascending=False)
    return df

def minimax(data, indices):
    
    df = data.copy()
    columns = df.columns.values 
    t = df.shape[0] 
    mn = np.zeros(t) 
    for i in range(t): 
        if i in df.index:
            mn[i] = df[columns[indices[0]]][i]
            for j in range(1, len(indices)):
                if mn[i] < df[columns[indices[j]]][i]:
                    mn[i] = df[columns[indices[j]]][i]
    df['maxVal'] = mn
    df = df.sort_values(by=['maxVal'], ascending=False)
    return df

#Leximin

def leximin(data, indices):
    df = data.copy()
    columns = df.columns.values
    t = df.shape[0]
    lex = [np.zeros(t) for i in range(len(indices))]
    a = [[] for i in range(len(indices))]
    for i in range(t):
        for j in range(len(indices)):
            if i in df.index:
                a[j] = df[columns[indices[j]]][i]
        a.sort()
        for j in range(len(indices)):
            lex[j][i] = a[j]
    for j in range(len(indices)):
        df['c' + str(j)] = lex[j]
    c = ['c' + str(i) for i in range(len(indices))]
    df = df.sort_values(by=c, ascending=False)
    return df

#Leximax

def leximax(data, indices):
    df = data.copy()
    columns = df.columns.values
    t = df.shape[0]
    lex = [np.zeros(t) for i in range(len(indices))]
    a = [[] for i in range(len(indices))]
    for i in range(t):
        for j in range(len(indices)):
            if i in df.index:
                a[j] = df[columns[indices[j]]][i]
        a.sort(reverse=True)
        for j in range(len(indices)):
            lex[j][i] = a[j]
    for j in range(len(indices)):
        df['c' + str(j)] = lex[j]
    c = ['c' + str(i) for i in range(len(indices))]
    df = df.sort_values(by=c, ascending=False)
    return df

def ParetoDomina(a,b):
    mi = len([1 for i in range(len(a)) if a[i] >= b[i]])
    my = len([1 for i in range(len(a)) if a[i] > b[i]])
    if mi == len(a):
        if my > 0:
            return True
    return False

def skylines(data, l):
    df = data.head(5000).copy()
    columns = df.columns.values
    t = df.shape[0]
    for i in range(t):
        if i in df.index:
            a = [0] * len(l)
            for j in range(i + 1, t):
                if j in df.index:
                    b = [0] * len(l)
                    for k in range(len(l)):
                        a[k] = df[columns[l[k]]][i]
                        b[k] = df[columns[l[k]]][j]
                    if ParetoDomina(a,b):
                        df = df.drop(j)
                    elif ParetoDomina(b,a):
                        df = df.drop(i)
                        break
    return df