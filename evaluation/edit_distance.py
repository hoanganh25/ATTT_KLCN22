"""Memory-efficient Levenshtein distance used by WER and CER metrics."""


def distance(a,b):
    if len(a)<len(b):a,b=b,a
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        next_row=[i]+[0]*len(b)
        for j,y in enumerate(b,1):
            next_row[j]=min(row[j]+1,next_row[j-1]+1,row[j-1]+(x!=y))
        row=next_row
    return row[-1]
